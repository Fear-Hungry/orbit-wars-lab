use std::collections::{HashMap, HashSet};

use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::combat::resolve_planet_combat;
use crate::config::{
    Config, BOARD_SIZE, CENTER, COMET_PRODUCTION, COMET_RADIUS, COMET_SPAWN_STEPS,
    ROTATION_RADIUS_LIMIT, SUN_RADIUS,
};
use crate::generator::generate_training_game;
use crate::geometry::{
    distance, orbital_radius, point_to_segment_distance, rotate_about_center, swept_pair_hit,
};
use crate::types::{CometGroup, Fleet, GameState, Move, Planet, StepOutcome};

/// A planet's motion over a single tick: position before (`old`) and after
/// (`new`) rotation / comet movement, plus whether it takes part in fleet
/// collision this tick. `check` is false for a comet's first on-board placement
/// (it appears mid-tick from off-board), which the official interpreter skips.
struct PlanetPath {
    id: i32,
    old: [f64; 2],
    new: [f64; 2],
    radius: f64,
    check: bool,
}

#[derive(Clone, Debug)]
pub struct Game {
    pub cfg: Config,
    pub state: GameState,
}

impl Game {
    pub fn new_training(cfg: Config, seed: u64, num_players: usize) -> Self {
        Self {
            cfg,
            state: generate_training_game(seed, num_players),
        }
    }

    pub fn from_state(cfg: Config, state: GameState) -> Self {
        Self { cfg, state }
    }

    pub fn step(&mut self, actions: &[Vec<Move>]) -> StepOutcome {
        if self.state.done {
            return self.outcome();
        }

        self.spawn_comets();
        self.process_actions(actions);
        self.produce();

        // Compute each planet's (old -> new) motion for this tick up front, move
        // fleets against those swept paths, then commit the motion. This mirrors
        // the official interpreter, which resolves fleet/planet collision
        // continuously over the planet's movement instead of against a stale
        // single position (which made rotating planets register phantom hits).
        let planet_paths = self.compute_planet_paths();
        let mut combat_lists: HashMap<i32, Vec<(i8, i32)>> = HashMap::new();
        self.move_fleets_and_collect_collisions(&planet_paths, &mut combat_lists);
        self.commit_planet_paths(&planet_paths);
        // Expired comets are removed after movement but before combat resolution,
        // matching the official interpreter: a just-expired comet is still
        // collidable during movement, but a fleet that hits it finds no combat
        // target, and a captured comet can still launch this tick before removal.
        self.remove_expired_comets();
        self.resolve_combat(combat_lists);

        self.state.step += 1;
        self.update_done();
        self.outcome()
    }

    pub fn scores(&self) -> Vec<i32> {
        let mut scores = vec![0_i32; self.state.num_players];
        for p in &self.state.planets {
            if p.owner >= 0 {
                let idx = p.owner as usize;
                if idx < scores.len() {
                    scores[idx] += p.ships.max(0);
                }
            }
        }
        for f in &self.state.fleets {
            if f.owner >= 0 {
                let idx = f.owner as usize;
                if idx < scores.len() {
                    scores[idx] += f.ships.max(0);
                }
            }
        }
        scores
    }

    fn outcome(&self) -> StepOutcome {
        let scores = self.scores();
        let mut rewards = vec![0.0_f32; self.state.num_players];
        if self.state.done {
            let max_score = scores.iter().copied().max().unwrap_or(0);
            for (i, score) in scores.iter().enumerate() {
                rewards[i] = if max_score > 0 && *score == max_score {
                    1.0
                } else {
                    -1.0
                };
            }
        }
        StepOutcome {
            rewards,
            done: self.state.done,
            scores,
        }
    }

    fn process_actions(&mut self, actions: &[Vec<Move>]) {
        for player_id in 0..self.state.num_players {
            let Some(player_actions) = actions.get(player_id) else {
                continue;
            };
            for mv in player_actions {
                if !mv.angle.is_finite() || mv.ships <= 0 {
                    continue;
                }
                let Some(idx) = self
                    .state
                    .planets
                    .iter()
                    .position(|p| p.id == mv.from_planet_id)
                else {
                    continue;
                };
                if self.state.planets[idx].owner != player_id as i8
                    || self.state.planets[idx].ships < mv.ships
                {
                    continue;
                }

                let (start_x, start_y, from_id, ships) = {
                    let p = &mut self.state.planets[idx];
                    p.ships -= mv.ships;
                    (
                        p.x + mv.angle.cos() * (p.radius + 0.1),
                        p.y + mv.angle.sin() * (p.radius + 0.1),
                        p.id,
                        mv.ships,
                    )
                };

                self.state.fleets.push(Fleet {
                    id: self.state.next_fleet_id,
                    owner: player_id as i8,
                    x: start_x,
                    y: start_y,
                    angle: mv.angle,
                    from_planet_id: from_id,
                    ships,
                });
                self.state.next_fleet_id += 1;
            }
        }
    }

    fn produce(&mut self) {
        for p in &mut self.state.planets {
            if p.owner != -1 {
                p.ships += p.production;
            }
        }
    }

    fn fleet_speed(&self, ships: i32) -> f64 {
        let s = (ships.max(1) as f64).ln() / 1000.0_f64.ln();
        let speed = 1.0 + (self.cfg.ship_speed - 1.0) * s.powf(1.5);
        speed.min(self.cfg.ship_speed).max(1.0)
    }

    fn move_fleets_and_collect_collisions(
        &mut self,
        planet_paths: &[PlanetPath],
        combat_lists: &mut HashMap<i32, Vec<(i8, i32)>>,
    ) {
        let max_fleets = self.cfg.max_fleets;
        let mut survivors = Vec::with_capacity(self.state.fleets.len().min(max_fleets));
        let drained: Vec<Fleet> = self.state.fleets.drain(..).collect();

        for mut fleet in drained {
            let old = [fleet.x, fleet.y];
            let speed = self.fleet_speed(fleet.ships);
            fleet.x += fleet.angle.cos() * speed;
            fleet.y += fleet.angle.sin() * speed;
            let new = [fleet.x, fleet.y];

            // Order matches the official interpreter: check planets FIRST, via a
            // swept-pair test over each planet's own motion this tick, so fast
            // fleets that would overshoot the board bounds or graze the sun still
            // get credit for hitting a planet along the way. Only fleets that hit
            // nothing are then tested against bounds and the sun.
            let mut collided = false;
            for path in planet_paths {
                if !path.check {
                    continue;
                }
                if swept_pair_hit(old, new, path.old, path.new, path.radius) {
                    combat_lists
                        .entry(path.id)
                        .or_default()
                        .push((fleet.owner, fleet.ships));
                    collided = true;
                    break;
                }
            }
            if collided {
                continue;
            }

            if !(0.0..=BOARD_SIZE).contains(&fleet.x) || !(0.0..=BOARD_SIZE).contains(&fleet.y) {
                continue;
            }
            if point_to_segment_distance([CENTER, CENTER], old, new) < SUN_RADIUS {
                continue;
            }

            if survivors.len() < max_fleets {
                survivors.push(fleet);
            }
        }
        self.state.fleets = survivors;
    }

    /// Compute each planet's (old -> new) motion for this tick without committing
    /// it, so fleet movement can use a continuous swept-pair collision check.
    /// Planet order matches `self.state.planets` (== official `obs0.planets`).
    fn compute_planet_paths(&self) -> Vec<PlanetPath> {
        let comet_ids: HashSet<i32> = self.state.comet_planet_ids.iter().copied().collect();
        let initial_by_id: HashMap<i32, Planet> = self
            .state
            .initial_planets
            .iter()
            .map(|planet| (planet.id, *planet))
            .collect();
        let rotation_angle = self.state.angular_velocity * self.state.step as f64;

        let mut paths: Vec<PlanetPath> = Vec::with_capacity(self.state.planets.len());

        for planet in &self.state.planets {
            if comet_ids.contains(&planet.id) {
                continue;
            }
            let old = [planet.x, planet.y];
            let mut new = old;
            if let Some(initial) = initial_by_id.get(&planet.id) {
                if orbital_radius(initial.x, initial.y) + planet.radius < ROTATION_RADIUS_LIMIT {
                    new = rotate_about_center(initial.x, initial.y, rotation_angle);
                }
            }
            paths.push(PlanetPath {
                id: planet.id,
                old,
                new,
                radius: planet.radius,
                check: true,
            });
        }

        for group in &self.state.comets {
            let idx = (group.path_index + 1) as usize;
            for (k, pid) in group.planet_ids.iter().copied().enumerate() {
                let Some(planet) = self.state.planets.iter().find(|p| p.id == pid) else {
                    continue;
                };
                let Some(path) = group.paths.get(k) else {
                    continue;
                };
                let old = [planet.x, planet.y];
                // Exhausted path: comet stays put (still collidable). Stepping in
                // from the off-board placeholder: appears mid-tick, no collision.
                let (new, check) = if idx >= path.len() {
                    (old, true)
                } else {
                    ([path[idx][0], path[idx][1]], old[0] >= 0.0)
                };
                paths.push(PlanetPath {
                    id: pid,
                    old,
                    new,
                    radius: planet.radius,
                    check,
                });
            }
        }

        paths
    }

    /// Apply the motion computed by `compute_planet_paths` and advance comet
    /// path indices. Called after fleet movement, like the official interpreter.
    fn commit_planet_paths(&mut self, planet_paths: &[PlanetPath]) {
        let new_positions: HashMap<i32, [f64; 2]> =
            planet_paths.iter().map(|path| (path.id, path.new)).collect();
        for planet in &mut self.state.planets {
            if let Some(position) = new_positions.get(&planet.id) {
                planet.x = position[0];
                planet.y = position[1];
            }
        }
        for group in &mut self.state.comets {
            group.path_index += 1;
        }
    }

    fn resolve_combat(&mut self, combat_lists: HashMap<i32, Vec<(i8, i32)>>) {
        for p in &mut self.state.planets {
            if let Some(arrivals) = combat_lists.get(&p.id) {
                resolve_planet_combat(p, arrivals);
            }
        }
    }

    fn update_done(&mut self) {
        if self.state.step >= self.cfg.episode_steps {
            self.state.done = true;
            return;
        }
        let mut alive = vec![false; self.state.num_players];
        for p in &self.state.planets {
            if p.owner >= 0 && (p.owner as usize) < alive.len() {
                alive[p.owner as usize] = true;
            }
        }
        for f in &self.state.fleets {
            if f.owner >= 0 && (f.owner as usize) < alive.len() {
                alive[f.owner as usize] = true;
            }
        }
        if alive.iter().filter(|x| **x).count() <= 1 {
            self.state.done = true;
        }
    }

    fn remove_expired_comets(&mut self) {
        let mut expired = HashSet::new();
        for group in &self.state.comets {
            for (i, pid) in group.planet_ids.iter().copied().enumerate() {
                let Some(path) = group.paths.get(i) else {
                    continue;
                };
                // Expired once the (post-increment) path index has advanced past
                // the last sampled position; the comet then stays put for one tick
                // and is removed this step, before combat (mirrors the official).
                if group.path_index >= path.len() as i32 {
                    expired.insert(pid);
                }
            }
        }
        if expired.is_empty() {
            return;
        }
        self.state.planets.retain(|p| !expired.contains(&p.id));
        self.state
            .initial_planets
            .retain(|p| !expired.contains(&p.id));
        self.state
            .comet_planet_ids
            .retain(|pid| !expired.contains(pid));
        for group in &mut self.state.comets {
            group.planet_ids.retain(|pid| !expired.contains(pid));
        }
        self.state.comets.retain(|g| !g.planet_ids.is_empty());
    }

    fn spawn_comets(&mut self) {
        if !self.cfg.enable_comets || !COMET_SPAWN_STEPS.contains(&(self.state.step + 1)) {
            return;
        }
        let mut rng = ChaCha8Rng::seed_from_u64(self.comet_spawn_seed());
        self.spawn_comets_with_rng(&mut rng);
    }

    fn spawn_comets_with_rng<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        if !self.cfg.enable_comets || !COMET_SPAWN_STEPS.contains(&(self.state.step + 1)) {
            return;
        }
        let Some(paths) = generate_comet_paths_with_rng(
            &self.state.initial_planets,
            self.state.angular_velocity,
            self.state.step + 1,
            &self.state.comet_planet_ids,
            self.cfg.comet_speed,
            rng,
        ) else {
            return;
        };

        let next_id = self
            .state
            .planets
            .iter()
            .map(|planet| planet.id)
            .max()
            .unwrap_or(-1)
            + 1;
        let comet_ships = (0..4).map(|_| rng.gen_range(1..=99)).min().unwrap_or(1);
        let mut group = CometGroup {
            planet_ids: Vec::with_capacity(paths.len()),
            paths,
            path_index: -1,
        };

        for offset in 0..group.paths.len() {
            let pid = next_id + offset as i32;
            group.planet_ids.push(pid);
            self.state.comet_planet_ids.push(pid);
            let comet = Planet {
                id: pid,
                owner: -1,
                x: -99.0,
                y: -99.0,
                radius: COMET_RADIUS,
                ships: comet_ships,
                production: COMET_PRODUCTION,
            };
            self.state.planets.push(comet);
            self.state.initial_planets.push(comet);
        }

        self.state.comets.push(group);
    }

    fn comet_spawn_seed(&self) -> u64 {
        let mut seed = 0x9E37_79B9_7F4A_7C15_u64 ^ (self.state.step as u64);
        seed = mix_seed(seed, self.state.num_players as u64);
        seed = mix_seed(seed, self.state.angular_velocity.to_bits());
        seed = mix_seed(seed, self.state.comet_planet_ids.len() as u64);
        for planet in &self.state.initial_planets {
            seed = mix_seed(seed, planet.id as u64);
            seed = mix_seed(seed, (planet.owner as i64) as u64);
            seed = mix_seed(seed, planet.x.to_bits());
            seed = mix_seed(seed, planet.y.to_bits());
            seed = mix_seed(seed, planet.radius.to_bits());
            seed = mix_seed(seed, planet.ships as u64);
            seed = mix_seed(seed, planet.production as u64);
        }
        seed
    }
}

fn mix_seed(seed: u64, value: u64) -> u64 {
    seed.rotate_left(7) ^ value.wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

fn generate_comet_paths_with_rng<R: Rng + ?Sized>(
    initial_planets: &[Planet],
    angular_velocity: f64,
    spawn_step: i32,
    comet_planet_ids: &[i32],
    comet_speed: f64,
    rng: &mut R,
) -> Option<Vec<Vec<[f64; 2]>>> {
    let comet_ids: HashSet<i32> = comet_planet_ids.iter().copied().collect();

    for _attempt in 0..300 {
        let eccentricity = rng.gen_range(0.75..0.93);
        let major_axis = rng.gen_range(60.0..150.0);
        let perihelion = major_axis * (1.0 - eccentricity);
        if perihelion < SUN_RADIUS + COMET_RADIUS {
            continue;
        }

        let minor_axis = major_axis * (1.0 - eccentricity * eccentricity).sqrt();
        let focus_offset = major_axis * eccentricity;
        let phi = rng.gen_range(std::f64::consts::PI / 6.0..std::f64::consts::PI / 3.0);

        let mut dense = Vec::with_capacity(5000);
        for idx in 0..5000 {
            let t = 0.3 * std::f64::consts::PI + 1.4 * std::f64::consts::PI * (idx as f64) / 4999.0;
            let ex = focus_offset + major_axis * t.cos();
            let ey = minor_axis * t.sin();
            let x = CENTER + ex * phi.cos() - ey * phi.sin();
            let y = CENTER + ex * phi.sin() + ey * phi.cos();
            dense.push([x, y]);
        }

        let path = resample_comet_path(&dense, comet_speed);
        let Some((board_start, board_end)) = visible_segment_bounds(&path) else {
            continue;
        };
        let visible = path[board_start..=board_end].to_vec();
        if !(5..=40).contains(&visible.len()) {
            continue;
        }

        let paths = vec![
            visible.clone(),
            visible
                .iter()
                .map(|point| [BOARD_SIZE - point[0], point[1]])
                .collect(),
            visible
                .iter()
                .map(|point| [point[0], BOARD_SIZE - point[1]])
                .collect(),
            visible
                .iter()
                .map(|point| [BOARD_SIZE - point[0], BOARD_SIZE - point[1]])
                .collect(),
        ];

        if comet_paths_are_valid(
            &visible,
            initial_planets,
            &comet_ids,
            angular_velocity,
            spawn_step,
        ) {
            return Some(paths);
        }
    }

    None
}

fn resample_comet_path(dense: &[[f64; 2]], comet_speed: f64) -> Vec<[f64; 2]> {
    if dense.is_empty() {
        return Vec::new();
    }
    let mut path = vec![dense[0]];
    let mut accumulated = 0.0;
    let mut target = comet_speed;
    for window in dense.windows(2) {
        accumulated += distance(window[0], window[1]);
        if accumulated >= target {
            path.push(window[1]);
            target += comet_speed;
        }
    }
    path
}

fn visible_segment_bounds(path: &[[f64; 2]]) -> Option<(usize, usize)> {
    let mut board_start = None;
    let mut board_end = None;
    for (idx, point) in path.iter().enumerate() {
        if (0.0..=BOARD_SIZE).contains(&point[0]) && (0.0..=BOARD_SIZE).contains(&point[1]) {
            if board_start.is_none() {
                board_start = Some(idx);
            }
            board_end = Some(idx);
        }
    }
    match (board_start, board_end) {
        (Some(start), Some(end)) => Some((start, end)),
        _ => None,
    }
}

fn comet_paths_are_valid(
    visible: &[[f64; 2]],
    initial_planets: &[Planet],
    comet_ids: &HashSet<i32>,
    angular_velocity: f64,
    spawn_step: i32,
) -> bool {
    let mut static_planets = Vec::new();
    let mut orbiting_planets = Vec::new();
    for planet in initial_planets {
        if comet_ids.contains(&planet.id) {
            continue;
        }
        let orbital = distance([planet.x, planet.y], [CENTER, CENTER]);
        if orbital + planet.radius < ROTATION_RADIUS_LIMIT {
            orbiting_planets.push(*planet);
        } else {
            static_planets.push(*planet);
        }
    }

    let comet_buffer = COMET_RADIUS + 0.5;
    for (step_idx, point) in visible.iter().enumerate() {
        if distance(*point, [CENTER, CENTER]) < SUN_RADIUS + COMET_RADIUS {
            return false;
        }

        let symmetric = [
            *point,
            [BOARD_SIZE - point[0], point[1]],
            [point[0], BOARD_SIZE - point[1]],
            [BOARD_SIZE - point[0], BOARD_SIZE - point[1]],
        ];

        for planet in &static_planets {
            if symmetric
                .iter()
                .any(|sym| distance(*sym, [planet.x, planet.y]) < planet.radius + comet_buffer)
            {
                return false;
            }
        }

        let game_step = spawn_step - 1 + step_idx as i32;
        for planet in &orbiting_planets {
            let dx = planet.x - CENTER;
            let dy = planet.y - CENTER;
            let orbital = (dx * dx + dy * dy).sqrt();
            let initial_angle = dy.atan2(dx);
            let current_angle = initial_angle + angular_velocity * game_step as f64;
            let px = CENTER + orbital * current_angle.cos();
            let py = CENTER + orbital * current_angle.sin();
            if symmetric
                .iter()
                .any(|sym| distance(*sym, [px, py]) < planet.radius + COMET_RADIUS)
            {
                return false;
            }
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Planet;
    use rand_chacha::rand_core::SeedableRng;

    fn test_config() -> Config {
        Config {
            enable_comets: false,
            ..Config::default()
        }
    }

    fn comet_config() -> Config {
        Config {
            enable_comets: true,
            ..Config::default()
        }
    }

    fn planet(id: i32, owner: i8, x: f64, y: f64, ships: i32) -> Planet {
        Planet {
            id,
            owner,
            x,
            y,
            radius: 2.0,
            ships,
            production: 1,
        }
    }

    fn base_state() -> GameState {
        let planets = vec![planet(0, 0, 20.0, 20.0, 20), planet(1, 1, 80.0, 80.0, 20)];
        GameState {
            step: 0,
            num_players: 2,
            angular_velocity: 0.0,
            initial_planets: planets.clone(),
            planets,
            fleets: Vec::new(),
            next_fleet_id: 0,
            comets: Vec::new(),
            comet_planet_ids: Vec::new(),
            done: false,
        }
    }

    #[test]
    fn invalid_actions_are_sanitized() {
        let mut game = Game::from_state(test_config(), base_state());
        let actions = vec![
            vec![
                Move {
                    from_planet_id: 999,
                    angle: 0.0,
                    ships: 5,
                },
                Move {
                    from_planet_id: 0,
                    angle: f64::NAN,
                    ships: 5,
                },
                Move {
                    from_planet_id: 0,
                    angle: 0.0,
                    ships: 0,
                },
                Move {
                    from_planet_id: 0,
                    angle: 0.0,
                    ships: 50,
                },
            ],
            vec![],
        ];

        game.step(&actions);

        assert!(game.state.fleets.is_empty());
        assert_eq!(game.state.next_fleet_id, 0);
        assert_eq!(game.state.planets[0].ships, 21);
        assert_eq!(game.state.planets[1].ships, 21);
    }

    #[test]
    fn launched_fleet_starts_outside_planet_radius() {
        let mut game = Game::from_state(test_config(), base_state());
        let angle = 0.0;
        let launched = 5;
        let actions = vec![
            vec![Move {
                from_planet_id: 0,
                angle,
                ships: launched,
            }],
            vec![],
        ];

        game.step(&actions);

        let fleet = game
            .state
            .fleets
            .iter()
            .find(|f| f.owner == 0)
            .expect("fleet launched");
        let source = &game.state.planets[0];
        let dist = ((fleet.x - source.x).powi(2) + (fleet.y - source.y).powi(2)).sqrt();
        let min_expected = source.radius + 0.1 + game.fleet_speed(launched);

        assert!(
            (dist - min_expected).abs() < 1e-9,
            "dist={dist} expected={min_expected}"
        );
    }

    #[test]
    fn production_happens_after_launch() {
        let mut game = Game::from_state(test_config(), base_state());
        let actions = vec![
            vec![Move {
                from_planet_id: 0,
                angle: 0.0,
                ships: 5,
            }],
            vec![],
        ];

        game.step(&actions);

        assert_eq!(game.state.planets[0].ships, 16);
        assert_eq!(game.state.planets[1].ships, 21);
    }

    #[test]
    fn fleet_speed_matches_official_formula() {
        let game = Game::from_state(test_config(), base_state());
        let expected = |ships: i32| {
            let s = (ships.max(1) as f64).ln() / 1000.0_f64.ln();
            let speed = 1.0 + (game.cfg.ship_speed - 1.0) * s.powf(1.5);
            speed.min(game.cfg.ship_speed).max(1.0)
        };

        for ships in [1, 2, 10, 100, 1000, 100_000] {
            let observed = game.fleet_speed(ships);
            let target = expected(ships);
            assert!(
                (observed - target).abs() < 1e-12,
                "ships={ships} observed={observed} target={target}"
            );
        }
    }

    #[test]
    fn rotating_planets_follow_official_step_index() {
        let mut state = base_state();
        state.planets[0].x = 70.0;
        state.planets[0].y = 50.0;
        state.initial_planets = state.planets.clone();
        state.angular_velocity = 0.1;
        let initial = state.planets[0].xy();

        let expected = rotate_about_center(
            state.planets[0].x,
            state.planets[0].y,
            state.angular_velocity,
        );
        let mut game = Game::from_state(test_config(), state);

        game.step(&[vec![], vec![]]);

        assert!((game.state.planets[0].x - initial[0]).abs() < 1e-12);
        assert!((game.state.planets[0].y - initial[1]).abs() < 1e-12);

        game.step(&[vec![], vec![]]);

        assert!((game.state.planets[0].x - expected[0]).abs() < 1e-12);
        assert!((game.state.planets[0].y - expected[1]).abs() < 1e-12);
    }

    #[test]
    fn terminal_score_rewards_winner_and_loser() {
        let mut state = base_state();
        state.planets[1].owner = -1;
        state.initial_planets = state.planets.clone();
        let mut game = Game::from_state(test_config(), state);

        let outcome = game.step(&[vec![], vec![]]);

        assert!(outcome.done);
        assert_eq!(outcome.scores, vec![21, 0]);
        assert_eq!(outcome.rewards, vec![1.0, -1.0]);
    }

    #[test]
    fn fleets_colliding_with_border_are_removed() {
        let mut state = base_state();
        state.fleets.push(Fleet {
            id: 0,
            owner: 0,
            x: 99.5,
            y: 50.0,
            angle: 0.0,
            from_planet_id: 0,
            ships: 1,
        });
        let mut game = Game::from_state(test_config(), state);

        game.step(&[vec![], vec![]]);

        assert!(game.state.fleets.is_empty());
    }

    #[test]
    fn fleets_colliding_with_sun_are_removed() {
        let mut state = base_state();
        state.fleets.push(Fleet {
            id: 0,
            owner: 0,
            x: 45.0,
            y: 50.0,
            angle: 0.0,
            from_planet_id: 0,
            ships: 1,
        });
        let mut game = Game::from_state(test_config(), state);

        game.step(&[vec![], vec![]]);

        assert!(game.state.fleets.is_empty());
    }

    #[test]
    fn fleets_colliding_with_planets_are_resolved_continuously() {
        let mut state = base_state();
        state.planets[1].x = 30.0;
        state.planets[1].y = 20.0;
        state.initial_planets = state.planets.clone();
        state.fleets.push(Fleet {
            id: 0,
            owner: 0,
            x: 26.0,
            y: 20.0,
            angle: 0.0,
            from_planet_id: 0,
            ships: 1000,
        });
        let mut game = Game::from_state(test_config(), state);

        game.step(&[vec![], vec![]]);

        assert!(game.state.fleets.is_empty());
        let target = game
            .state
            .planets
            .iter()
            .find(|p| p.id == 1)
            .expect("target planet exists");
        assert_eq!(target.owner, 0);
        assert!(target.ships > 0);
    }

    #[test]
    fn moving_planet_sweep_collision_hits_stationary_path_point() {
        let mut state = base_state();
        state.planets[0].x = 97.0;
        state.planets[0].y = 50.0;
        state.planets[0].owner = 1;
        state.planets[0].ships = 5;
        state.planets[1].owner = -1;
        state.initial_planets = state.planets.clone();
        state.angular_velocity = 0.2;
        state.step = 1;
        state.fleets.push(Fleet {
            id: 0,
            owner: 0,
            x: 95.5315551687,
            y: 54.6682550269,
            angle: 0.0,
            from_planet_id: 1,
            ships: 3,
        });
        let mut game = Game::from_state(test_config(), state);

        game.step(&[vec![], vec![]]);

        assert!(game.state.fleets.is_empty());
        let swept_planet = game
            .state
            .planets
            .iter()
            .find(|p| p.id == 0)
            .expect("planet exists");
        assert_eq!(swept_planet.owner, 1);
        assert_eq!(swept_planet.ships, 3);
    }

    #[test]
    fn generated_comet_paths_follow_configured_speed() {
        let state = base_state();
        let mut rng = ChaCha8Rng::seed_from_u64(7);

        let paths =
            generate_comet_paths_with_rng(&state.initial_planets, 0.03, 50, &[], 4.0, &mut rng)
                .expect("comet paths generated");

        assert_eq!(paths.len(), 4);
        for path in &paths {
            assert!((5..=40).contains(&path.len()));
            for segment in path.windows(2) {
                let step = distance(segment[0], segment[1]);
                assert!((step - 4.0).abs() < 0.3, "step={step}");
            }
        }
    }

    #[test]
    fn spawned_comets_move_and_expire() {
        let mut state = base_state();
        state.step = 49;
        state.angular_velocity = 0.03;
        let mut game = Game::from_state(comet_config(), state);
        let mut rng = ChaCha8Rng::seed_from_u64(7);

        game.spawn_comets_with_rng(&mut rng);

        assert_eq!(game.state.comets.len(), 1);
        let spawned = game.state.comets[0].clone();
        assert_eq!(spawned.path_index, -1);
        assert_eq!(spawned.planet_ids.len(), 4);
        for pid in &spawned.planet_ids {
            let comet = game
                .state
                .planets
                .iter()
                .find(|planet| planet.id == *pid)
                .expect("comet planet");
            assert_eq!(comet.x, -99.0);
            assert_eq!(comet.y, -99.0);
            assert_eq!(comet.radius, COMET_RADIUS);
            assert_eq!(comet.production, COMET_PRODUCTION);
        }

        let paths = game.compute_planet_paths();
        game.commit_planet_paths(&paths);
        let after_first_move = game.state.comets[0].clone();
        assert_eq!(after_first_move.path_index, 0);
        for (idx, pid) in after_first_move.planet_ids.iter().enumerate() {
            let comet = game
                .state
                .planets
                .iter()
                .find(|planet| planet.id == *pid)
                .expect("moved comet");
            let expected = after_first_move.paths[idx][0];
            assert!((comet.x - expected[0]).abs() < 1e-12);
            assert!((comet.y - expected[1]).abs() < 1e-12);
        }

        let previous_positions: Vec<[f64; 2]> = after_first_move
            .planet_ids
            .iter()
            .map(|pid| {
                let comet = game
                    .state
                    .planets
                    .iter()
                    .find(|planet| planet.id == *pid)
                    .expect("current comet");
                [comet.x, comet.y]
            })
            .collect();
        let paths = game.compute_planet_paths();
        game.commit_planet_paths(&paths);
        for (idx, pid) in game.state.comets[0].planet_ids.iter().enumerate() {
            let comet = game
                .state
                .planets
                .iter()
                .find(|planet| planet.id == *pid)
                .expect("advanced comet");
            let moved = distance(previous_positions[idx], [comet.x, comet.y]);
            assert!((moved - 4.0).abs() < 0.3, "moved={moved}");
        }

        game.state.comets[0].path_index = game.state.comets[0].paths[0].len() as i32;
        game.remove_expired_comets();

        assert!(game.state.comets.is_empty());
        for pid in spawned.planet_ids {
            assert!(!game.state.planets.iter().any(|planet| planet.id == pid));
            assert!(!game
                .state
                .initial_planets
                .iter()
                .any(|planet| planet.id == pid));
            assert!(!game.state.comet_planet_ids.contains(&pid));
        }
    }
}
