use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};

use crate::config::{BOARD_SIZE, CENTER, PLANET_CLEARANCE, ROTATION_RADIUS_LIMIT, SUN_RADIUS};
use crate::geometry::distance;
use crate::types::{GameState, Planet};

const MIN_PLANET_GROUPS: usize = 5;
const MAX_PLANET_GROUPS: usize = 10;
const MIN_STATIC_GROUPS: usize = 3;

/// Training generator.
///
/// This mirrors the official Orbit Wars opening-map distribution while staying
/// deterministic under an explicit Rust seed. Use official snapshots for
/// parity tests; use this generator for seeded local training.
pub fn generate_training_game(seed: u64, num_players: usize) -> GameState {
    assert!(num_players == 2 || num_players == 4);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let angular_velocity = rng.gen_range(0.025..0.05);
    let mut planets = generate_training_planets(&mut rng);
    assign_home_planets(&mut planets, num_players, &mut rng);
    let initial_planets = planets.clone();
    GameState {
        step: 0,
        num_players,
        angular_velocity,
        planets,
        initial_planets,
        fleets: Vec::new(),
        next_fleet_id: 0,
        comets: Vec::new(),
        comet_planet_ids: Vec::new(),
        done: false,
    }
}

fn generate_training_planets<R: Rng + ?Sized>(rng: &mut R) -> Vec<Planet> {
    let groups = rng.gen_range(MIN_PLANET_GROUPS..=MAX_PLANET_GROUPS);
    let mut planets: Vec<Planet> = Vec::with_capacity(groups * 4);
    let mut id_counter = 0_i32;
    let mut static_groups = 0_usize;

    for _ in 0..5000 {
        if static_groups >= MIN_STATIC_GROUPS {
            break;
        }
        let production = rng.gen_range(1..=5);
        let radius = 1.0 + (production as f64).ln();
        let angle = rng.gen_range(0.0..FRAC_PI_2);
        let min_orbital = ROTATION_RADIUS_LIMIT - radius;
        let max_orbital = (BOARD_SIZE - CENTER - radius) / angle.cos().max(angle.sin());
        if min_orbital > max_orbital {
            continue;
        }
        let orbital_radius = rng.gen_range(min_orbital..=max_orbital);
        let x = CENTER + orbital_radius * angle.cos();
        let y = CENTER + orbital_radius * angle.sin();

        if (x - CENTER) < radius + 5.0 || (y - CENTER) < radius + 5.0 {
            continue;
        }

        let ships = sample_large_group_ships(rng);
        let group = build_symmetric_group(id_counter, x, y, radius, ships, production);
        if group_is_valid(&group, &planets) {
            planets.extend(group);
            id_counter += 4;
            static_groups += 1;
        }
    }

    for _ in 0..1000 {
        let production = rng.gen_range(1..=5);
        let radius = 1.0 + (production as f64).ln();
        let min_orbital = SUN_RADIUS + radius + 10.0;
        let max_orbital = ROTATION_RADIUS_LIMIT - radius;
        if min_orbital >= max_orbital {
            continue;
        }
        let orbital_radius = rng.gen_range(min_orbital..=max_orbital);
        let x = CENTER + orbital_radius * FRAC_PI_4.cos();
        let y = CENTER + orbital_radius * FRAC_PI_4.sin();
        let ships = sample_large_group_ships(rng);
        let group = build_symmetric_group(id_counter, x, y, radius, ships, production);
        if group_is_valid(&group, &planets) {
            planets.extend(group);
            id_counter += 4;
            break;
        }
    }

    let mut attempts = 0;
    let mut has_orbiting = planets.iter().any(is_rotating);
    while planets.len() < groups * 4 || !has_orbiting {
        attempts += 1;
        if attempts >= 5000 {
            break;
        }
        let production = rng.gen_range(1..=5);
        let radius = 1.0 + (production as f64).ln();
        let x = rng.gen_range((CENTER + 15.0)..(BOARD_SIZE - radius - 5.0));
        let y = rng.gen_range((CENTER + 15.0)..(BOARD_SIZE - radius - 5.0));
        let orbital_radius = distance([x, y], [CENTER, CENTER]);
        if orbital_radius < SUN_RADIUS + radius + 10.0 {
            continue;
        }
        if orbital_radius + radius >= ROTATION_RADIUS_LIMIT
            && (x + radius > BOARD_SIZE
                || x - radius < 0.0
                || y + radius > BOARD_SIZE
                || y - radius < 0.0)
        {
            continue;
        }

        let ships = rng.gen_range(5..=30);
        let group = build_symmetric_group(id_counter, x, y, radius, ships, production);
        if group_is_valid(&group, &planets) {
            has_orbiting |= is_rotating(&group[0]);
            planets.extend(group);
            id_counter += 4;
        }
    }

    planets
}

fn sample_large_group_ships<R: Rng + ?Sized>(rng: &mut R) -> i32 {
    rng.gen_range(5..=99).min(rng.gen_range(5..=99))
}

fn build_symmetric_group(
    id_counter: i32,
    x: f64,
    y: f64,
    radius: f64,
    ships: i32,
    production: i32,
) -> [Planet; 4] {
    [
        Planet {
            id: id_counter,
            owner: -1,
            x,
            y,
            radius,
            ships,
            production,
        },
        Planet {
            id: id_counter + 1,
            owner: -1,
            x: BOARD_SIZE - x,
            y,
            radius,
            ships,
            production,
        },
        Planet {
            id: id_counter + 2,
            owner: -1,
            x,
            y: BOARD_SIZE - y,
            radius,
            ships,
            production,
        },
        Planet {
            id: id_counter + 3,
            owner: -1,
            x: BOARD_SIZE - x,
            y: BOARD_SIZE - y,
            radius,
            ships,
            production,
        },
    ]
}

fn group_is_valid(group: &[Planet; 4], existing: &[Planet]) -> bool {
    let mut seen = existing.to_vec();
    for planet in group {
        if !is_valid_new_planet(planet, &seen) {
            return false;
        }
        seen.push(*planet);
    }
    true
}

fn is_rotating(planet: &Planet) -> bool {
    distance([planet.x, planet.y], [CENTER, CENTER]) + planet.radius < ROTATION_RADIUS_LIMIT
}

fn assign_home_planets<R: Rng + ?Sized>(planets: &mut [Planet], num_players: usize, rng: &mut R) {
    let num_groups = planets.len() / 4;
    if num_groups == 0 {
        return;
    }

    let mut home_group = rng.gen_range(0..num_groups);
    if num_players == 4 && is_rotating(&planets[home_group * 4]) {
        if let Some(diagonal_group) = (0..num_groups).find(|group_idx| {
            let planet = planets[group_idx * 4];
            is_rotating(&planet) && ((planet.x - CENTER) - (planet.y - CENTER)).abs() < 0.01
        }) {
            home_group = diagonal_group;
        }
    }

    let base = home_group * 4;
    if num_players == 2 {
        planets[base].owner = 0;
        planets[base].ships = 10;
        planets[base + 3].owner = 1;
        planets[base + 3].ships = 10;
    } else {
        for (owner, planet) in planets[base..base + 4].iter_mut().enumerate() {
            planet.owner = owner as i8;
            planet.ships = 10;
        }
    }
}

fn is_valid_new_planet(np: &Planet, existing: &[Planet]) -> bool {
    if np.x - np.radius < 0.0
        || np.x + np.radius > BOARD_SIZE
        || np.y - np.radius < 0.0
        || np.y + np.radius > BOARD_SIZE
    {
        return false;
    }
    for p in existing {
        if distance([np.x, np.y], [p.x, p.y]) < np.radius + p.radius + PLANET_CLEARANCE {
            return false;
        }
        let np_rotating =
            distance([np.x, np.y], [CENTER, CENTER]) + np.radius < ROTATION_RADIUS_LIMIT;
        let p_rotating = distance([p.x, p.y], [CENTER, CENTER]) + p.radius < ROTATION_RADIUS_LIMIT;
        if np_rotating != p_rotating {
            let r1 = distance([np.x, np.y], [CENTER, CENTER]);
            let r2 = distance([p.x, p.y], [CENTER, CENTER]);
            if (r1 - r2).abs() < np.radius + p.radius + PLANET_CLEARANCE {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_no_overlap(planets: &[Planet]) {
        for (idx, a) in planets.iter().enumerate() {
            for b in planets.iter().skip(idx + 1) {
                let min_distance = a.radius + b.radius + PLANET_CLEARANCE;
                assert!(
                    distance([a.x, a.y], [b.x, b.y]) >= min_distance,
                    "planets {} and {} overlap: {} < {}",
                    a.id,
                    b.id,
                    distance([a.x, a.y], [b.x, b.y]),
                    min_distance
                );
            }
        }
    }

    fn assert_quadrant_symmetry(group: &[Planet]) {
        assert_eq!(group.len(), 4);
        let a = group[0];
        let b = group[1];
        let c = group[2];
        let d = group[3];

        for planet in group {
            assert_eq!(planet.radius, a.radius);
            assert_eq!(planet.production, a.production);
        }

        assert_eq!(b.x, BOARD_SIZE - a.x);
        assert_eq!(b.y, a.y);
        assert_eq!(c.x, a.x);
        assert_eq!(c.y, BOARD_SIZE - a.y);
        assert_eq!(d.x, BOARD_SIZE - a.x);
        assert_eq!(d.y, BOARD_SIZE - a.y);
    }

    fn static_group_count(planets: &[Planet]) -> usize {
        planets.chunks_exact(4).filter(|group| !is_rotating(&group[0])).count()
    }

    fn orbiting_group_count(planets: &[Planet]) -> usize {
        planets.chunks_exact(4).filter(|group| is_rotating(&group[0])).count()
    }

    fn has_diagonal_orbiting_group(planets: &[Planet]) -> bool {
        planets.chunks_exact(4).any(|group| {
            let planet = group[0];
            is_rotating(&planet) && ((planet.x - CENTER) - (planet.y - CENTER)).abs() < 0.01
        })
    }

    fn assert_training_distribution(seed: u64, num_players: usize) {
        let state = generate_training_game(seed, num_players);

        assert_eq!(state.step, 0);
        assert_eq!(state.num_players, num_players);
        assert!(!state.done);
        assert!(state.fleets.is_empty());
        assert!(state.comets.is_empty());
        assert!(state.comet_planet_ids.is_empty());
        assert_eq!(state.next_fleet_id, 0);
        assert_eq!(state.initial_planets, state.planets);
        assert!(state.planets.len() >= 20);
        assert!(state.planets.len() <= 40);
        assert_eq!(state.planets.len() % 4, 0);
        assert!((0.025..0.050).contains(&state.angular_velocity));

        assert_no_overlap(&state.planets);
        assert!(static_group_count(&state.planets) >= MIN_STATIC_GROUPS);
        assert!(orbiting_group_count(&state.planets) >= 1);
        assert!(has_diagonal_orbiting_group(&state.planets));

        for chunk in state.planets.chunks_exact(4) {
            assert_quadrant_symmetry(chunk);
        }

        if num_players == 2 {
            let home_groups: Vec<_> = state
                .planets
                .chunks_exact(4)
                .filter(|group| group.iter().any(|planet| planet.owner != -1))
                .collect();
            assert_eq!(home_groups.len(), 1);
            let group = home_groups[0];
            assert_eq!(group[0].owner, 0);
            assert_eq!(group[3].owner, 1);
            assert_eq!(group[0].ships, 10);
            assert_eq!(group[3].ships, 10);
            assert_eq!(group[1].owner, -1);
            assert_eq!(group[2].owner, -1);
        } else {
            let home_groups: Vec<_> = state
                .planets
                .chunks_exact(4)
                .filter(|group| group.iter().any(|planet| planet.owner != -1))
                .collect();
            assert_eq!(home_groups.len(), 1);
            for (owner, planet) in home_groups[0].iter().enumerate() {
                assert_eq!(planet.owner, owner as i8);
                assert_eq!(planet.ships, 10);
            }
        }

        for group in state.planets.chunks_exact(4) {
            let owned = group.iter().filter(|planet| planet.owner != -1).count();
            assert!(owned == 0 || owned == 2 || owned == 4);
        }
    }

    #[test]
    fn generator_is_deterministic_for_same_seed() {
        assert_eq!(generate_training_game(7, 2), generate_training_game(7, 2));
        assert_eq!(generate_training_game(19, 4), generate_training_game(19, 4));
    }

    #[test]
    fn generator_changes_layout_for_different_seeds() {
        let a = generate_training_game(1, 2);
        let b = generate_training_game(2, 2);
        assert_ne!(a.planets, b.planets);
        assert_ne!(a.angular_velocity, b.angular_velocity);
    }

    #[test]
    fn generator_matches_expected_training_distribution() {
        for seed in [0_u64, 1, 7, 42, 99] {
            assert_training_distribution(seed, 2);
            assert_training_distribution(seed, 4);
        }
    }
}
