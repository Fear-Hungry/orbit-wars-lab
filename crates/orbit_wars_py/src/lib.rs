use numpy::{PyArray1, PyReadonlyArray2};
use orbit_wars_core::{BatchSimulator, Config, Fleet, GameState, Move, Planet};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::types::PyDict;

const BOARD_SIZE: f64 = 100.0;
const CENTER: f64 = 50.0;

#[pyclass]
#[derive(Clone)]
struct PyConfig {
    #[pyo3(get, set)]
    episode_steps: i32,
    #[pyo3(get, set)]
    act_timeout: f64,
    #[pyo3(get, set)]
    ship_speed: f64,
    #[pyo3(get, set)]
    comet_speed: f64,
    #[pyo3(get, set)]
    enable_comets: bool,
    #[pyo3(get, set)]
    max_planets: usize,
    #[pyo3(get, set)]
    max_fleets: usize,
}

impl From<PyConfig> for Config {
    fn from(value: PyConfig) -> Self {
        Self {
            episode_steps: value.episode_steps,
            act_timeout: value.act_timeout,
            ship_speed: value.ship_speed,
            comet_speed: value.comet_speed,
            enable_comets: value.enable_comets,
            max_planets: value.max_planets,
            max_fleets: value.max_fleets,
        }
    }
}

#[pymethods]
impl PyConfig {
    #[new]
    #[pyo3(signature = (episode_steps=500, act_timeout=1.0, ship_speed=6.0, comet_speed=4.0, enable_comets=true, max_planets=96, max_fleets=4096))]
    fn new(
        episode_steps: i32,
        act_timeout: f64,
        ship_speed: f64,
        comet_speed: f64,
        enable_comets: bool,
        max_planets: usize,
        max_fleets: usize,
    ) -> Self {
        Self {
            episode_steps,
            act_timeout,
            ship_speed,
            comet_speed,
            enable_comets,
            max_planets,
            max_fleets,
        }
    }
}

#[pyclass]
struct PyBatchSimulator {
    inner: BatchSimulator,
}

#[pymethods]
impl PyBatchSimulator {
    #[new]
    #[pyo3(signature = (num_envs, num_players=2, seed=0, config=None))]
    fn new(
        num_envs: usize,
        num_players: usize,
        seed: u64,
        config: Option<PyConfig>,
    ) -> PyResult<Self> {
        if num_players != 2 && num_players != 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Orbit Wars supports only 2 or 4 players",
            ));
        }
        let cfg: Config = config
            .unwrap_or_else(|| PyConfig::new(500, 1.0, 6.0, 4.0, true, 96, 4096))
            .into();
        Ok(Self {
            inner: BatchSimulator::new(num_envs, num_players, cfg, seed),
        })
    }

    fn reset_json(&mut self, seed: u64) -> PyResult<String> {
        let states = self.inner.reset(seed);
        serde_json::to_string(&states)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn reset_msgpack<'py>(&mut self, py: Python<'py>, seed: u64) -> PyResult<Bound<'py, PyBytes>> {
        let states = py.detach(|| self.inner.reset(seed));
        let encoded = rmp_serde::to_vec_named(&states)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PyBytes::new(py, &encoded))
    }

    fn reset_from_states_json(&mut self, states_json: &str) -> PyResult<String> {
        let states: Vec<GameState> = serde_json::from_str(states_json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let states = self.inner.reset_from_states(states);
        serde_json::to_string(&states)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn reset_from_states_msgpack<'py>(
        &mut self,
        py: Python<'py>,
        states_bytes: &[u8],
    ) -> PyResult<Bound<'py, PyBytes>> {
        let states: Vec<GameState> = rmp_serde::from_slice(states_bytes)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let states = py.detach(|| self.inner.reset_from_states(states));
        let encoded = rmp_serde::to_vec_named(&states)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PyBytes::new(py, &encoded))
    }

    fn states_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner.states())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn states_msgpack<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let states = py.detach(|| self.inner.states());
        let encoded = rmp_serde::to_vec_named(&states)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PyBytes::new(py, &encoded))
    }

    /// Fast training API: returns a flat float32 array with one encoded row per env.
    ///
    /// Shape is `(num_envs, 8 + max_planets * 14 + max_fleets * 10)` after
    /// reshaping on the Python side. This avoids serializing GameState into
    /// Python dicts when the learner only needs the PPO observation vector.
    #[pyo3(signature = (player, max_planets=96, max_fleets=256, include_fleets=true))]
    fn encoded_states<'py>(
        &self,
        py: Python<'py>,
        player: i8,
        max_planets: usize,
        max_fleets: usize,
        include_fleets: bool,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let encoded = py.detach(|| {
            let states = self.inner.states();
            let dim = encoded_state_dim(max_planets, max_fleets);
            let mut out = Vec::with_capacity(states.len() * dim);
            for state in &states {
                encode_state_into(
                    state,
                    player,
                    max_planets,
                    max_fleets,
                    include_fleets,
                    &mut out,
                );
            }
            out
        });
        Ok(PyArray1::from_vec(py, encoded))
    }

    /// Debug-oriented API: actions[env][player][move] = [from_id, angle, ships].
    ///
    /// For high-throughput training, replace this with a NumPy zero-copy API.
    fn step_json(&mut self, py: Python<'_>, actions_json: &str) -> PyResult<String> {
        let raw: Vec<Vec<Vec<[f64; 3]>>> = serde_json::from_str(actions_json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let actions: Vec<Vec<Vec<Move>>> = raw
            .into_iter()
            .map(|env_actions| {
                env_actions
                    .into_iter()
                    .map(|player_actions| {
                        player_actions
                            .into_iter()
                            .map(|m| Move {
                                from_planet_id: m[0] as i32,
                                angle: m[1],
                                ships: m[2] as i32,
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let outcomes = py.detach(|| self.inner.step(actions));
        serde_json::to_string(&outcomes)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn step_msgpack<'py>(
        &mut self,
        py: Python<'py>,
        actions_bytes: &[u8],
    ) -> PyResult<Bound<'py, PyBytes>> {
        let actions = parse_actions_binary(actions_bytes)?;
        let outcomes = py.detach(|| self.inner.step(actions));
        let encoded = rmp_serde::to_vec_named(&outcomes)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PyBytes::new(py, &encoded))
    }

    /// Debug-oriented API returning outcomes and post-step states in one JSON payload.
    fn step_with_states_json(&mut self, py: Python<'_>, actions_json: &str) -> PyResult<String> {
        let raw: Vec<Vec<Vec<[f64; 3]>>> = serde_json::from_str(actions_json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let actions: Vec<Vec<Vec<Move>>> = raw
            .into_iter()
            .map(|env_actions| {
                env_actions
                    .into_iter()
                    .map(|player_actions| {
                        player_actions
                            .into_iter()
                            .map(|m| Move {
                                from_planet_id: m[0] as i32,
                                angle: m[1],
                                ships: m[2] as i32,
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let payload = py.detach(|| self.inner.step_with_states(actions));
        serde_json::to_string(&payload)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Fast API: binary actions in, MessagePack `(outcomes, states)` out.
    fn step_with_states_msgpack<'py>(
        &mut self,
        py: Python<'py>,
        actions_bytes: &[u8],
    ) -> PyResult<Bound<'py, PyBytes>> {
        let actions = parse_actions_binary(actions_bytes)?;
        let payload = py.detach(|| self.inner.step_with_states(actions));
        let encoded = rmp_serde::to_vec_named(&payload)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PyBytes::new(py, &encoded))
    }

    /// Fast API: flat NumPy actions in, MessagePack `(outcomes, states)` out.
    ///
    /// `actions_rows` must have shape `(n, 5)` with rows:
    /// `[env_index, player_index, from_planet_id, angle, ships]`.
    #[pyo3(signature = (actions_rows))]
    fn step_flat_with_states_msgpack<'py>(
        &mut self,
        py: Python<'py>,
        actions_rows: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let actions =
            parse_flat_actions_array(actions_rows, self.inner.games.len(), self.inner.num_players)?;
        let payload = py.detach(|| self.inner.step_with_states(actions));
        let encoded = rmp_serde::to_vec_named(&payload)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PyBytes::new(py, &encoded))
    }

    /// Fast training API: binary actions in, `(outcomes, encoded_states)` out.
    ///
    /// This keeps the post-step state on the Rust side and only materializes the
    /// flat PPO observation array in Python. Use `step_with_states_msgpack` for
    /// debug paths that need full dict-like GameState payloads.
    #[pyo3(signature = (actions_bytes, player, max_planets=96, max_fleets=256, include_fleets=true))]
    fn step_with_encoded_states_msgpack<'py>(
        &mut self,
        py: Python<'py>,
        actions_bytes: &[u8],
        player: i8,
        max_planets: usize,
        max_fleets: usize,
        include_fleets: bool,
    ) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyArray1<f32>>)> {
        let actions = parse_actions_binary(actions_bytes)?;
        let (outcomes, encoded_states) = py.detach(|| {
            let (outcomes, states) = self.inner.step_with_states(actions);
            let dim = encoded_state_dim(max_planets, max_fleets);
            let mut encoded = Vec::with_capacity(states.len() * dim);
            for state in &states {
                encode_state_into(
                    state,
                    player,
                    max_planets,
                    max_fleets,
                    include_fleets,
                    &mut encoded,
                );
            }
            (outcomes, encoded)
        });
        let encoded_outcomes = rmp_serde::to_vec_named(&outcomes)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok((
            PyBytes::new(py, &encoded_outcomes),
            PyArray1::from_vec(py, encoded_states),
        ))
    }

    /// Fastest training API: flat NumPy actions in, `(outcomes, encoded_states)` out.
    ///
    /// `actions_rows` must have shape `(n, 5)` with rows:
    /// `[env_index, player_index, from_planet_id, angle, ships]`.
    #[pyo3(signature = (actions_rows, player, max_planets=96, max_fleets=256, include_fleets=true))]
    fn step_flat_with_encoded_states_msgpack<'py>(
        &mut self,
        py: Python<'py>,
        actions_rows: PyReadonlyArray2<'_, f64>,
        player: i8,
        max_planets: usize,
        max_fleets: usize,
        include_fleets: bool,
    ) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyArray1<f32>>)> {
        let actions =
            parse_flat_actions_array(actions_rows, self.inner.games.len(), self.inner.num_players)?;
        let (outcomes, encoded_states) = py.detach(|| {
            let (outcomes, states) = self.inner.step_with_states(actions);
            let dim = encoded_state_dim(max_planets, max_fleets);
            let mut encoded = Vec::with_capacity(states.len() * dim);
            for state in &states {
                encode_state_into(
                    state,
                    player,
                    max_planets,
                    max_fleets,
                    include_fleets,
                    &mut encoded,
                );
            }
            (outcomes, encoded)
        });
        let encoded_outcomes = rmp_serde::to_vec_named(&outcomes)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok((
            PyBytes::new(py, &encoded_outcomes),
            PyArray1::from_vec(py, encoded_states),
        ))
    }
}

fn read_u32(raw: &[u8], offset: &mut usize) -> PyResult<u32> {
    let end = *offset + 4;
    let bytes = raw
        .get(*offset..end)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("truncated action buffer"))?;
    *offset = end;
    Ok(u32::from_le_bytes(
        bytes.try_into().expect("slice length checked"),
    ))
}

fn read_i32(raw: &[u8], offset: &mut usize) -> PyResult<i32> {
    let end = *offset + 4;
    let bytes = raw
        .get(*offset..end)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("truncated action buffer"))?;
    *offset = end;
    Ok(i32::from_le_bytes(
        bytes.try_into().expect("slice length checked"),
    ))
}

fn read_f64(raw: &[u8], offset: &mut usize) -> PyResult<f64> {
    let end = *offset + 8;
    let bytes = raw
        .get(*offset..end)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("truncated action buffer"))?;
    *offset = end;
    Ok(f64::from_le_bytes(
        bytes.try_into().expect("slice length checked"),
    ))
}

fn parse_actions_binary(raw: &[u8]) -> PyResult<Vec<Vec<Vec<Move>>>> {
    let mut offset = 0usize;
    let env_count = read_u32(raw, &mut offset)? as usize;
    let mut actions = Vec::with_capacity(env_count);
    for _ in 0..env_count {
        let player_count = read_u32(raw, &mut offset)? as usize;
        let mut env_actions = Vec::with_capacity(player_count);
        for _ in 0..player_count {
            let move_count = read_u32(raw, &mut offset)? as usize;
            let mut player_actions = Vec::with_capacity(move_count);
            for _ in 0..move_count {
                let from_planet_id = read_i32(raw, &mut offset)?;
                let angle = read_f64(raw, &mut offset)?;
                let ships = read_i32(raw, &mut offset)?;
                player_actions.push(Move {
                    from_planet_id,
                    angle,
                    ships,
                });
            }
            env_actions.push(player_actions);
        }
        actions.push(env_actions);
    }
    if offset != raw.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "action buffer has trailing bytes",
        ));
    }
    Ok(actions)
}

fn parse_flat_actions_array(
    actions_rows: PyReadonlyArray2<'_, f64>,
    num_envs: usize,
    num_players: usize,
) -> PyResult<Vec<Vec<Vec<Move>>>> {
    let rows = actions_rows.as_array();
    if rows.shape().get(1).copied() != Some(5) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "flat action array must have shape (n, 5)",
        ));
    }

    let mut actions = vec![vec![Vec::new(); num_players]; num_envs];
    for row in rows.outer_iter() {
        let env_index = row[0] as isize;
        let player_index = row[1] as isize;
        if env_index < 0 || env_index as usize >= num_envs {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "flat action env index out of range",
            ));
        }
        if player_index < 0 || player_index as usize >= num_players {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "flat action player index out of range",
            ));
        }
        actions[env_index as usize][player_index as usize].push(Move {
            from_planet_id: row[2] as i32,
            angle: row[3],
            ships: row[4] as i32,
        });
    }
    Ok(actions)
}

fn encoded_state_dim(max_planets: usize, max_fleets: usize) -> usize {
    8 + max_planets * 14 + max_fleets * 10
}

fn owner_rel(owner: i8, player: i8) -> [f32; 4] {
    if owner == -1 {
        [0.0, 0.0, 1.0, 0.0]
    } else if owner == player {
        [1.0, 0.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0, 0.0]
    }
}

fn encode_state_into(
    state: &GameState,
    player: i8,
    max_planets: usize,
    max_fleets: usize,
    include_fleets: bool,
    out: &mut Vec<f32>,
) {
    let own_ships: i32 = state
        .planets
        .iter()
        .filter(|p| p.owner == player)
        .map(|p| p.ships)
        .sum();
    let enemy_ships: i32 = state
        .planets
        .iter()
        .filter(|p| p.owner != -1 && p.owner != player)
        .map(|p| p.ships)
        .sum();
    let own_prod: i32 = state
        .planets
        .iter()
        .filter(|p| p.owner == player)
        .map(|p| p.production)
        .sum();
    let enemy_prod: i32 = state
        .planets
        .iter()
        .filter(|p| p.owner != -1 && p.owner != player)
        .map(|p| p.production)
        .sum();

    out.extend_from_slice(&[
        state.step as f32 / 500.0,
        state.angular_velocity as f32,
        state.planets.len() as f32 / max_planets.max(1) as f32,
        state.fleets.len() as f32 / max_fleets.max(1) as f32,
        ((own_ships.max(0) as f64).ln_1p() / 8.0) as f32,
        ((enemy_ships.max(0) as f64).ln_1p() / 8.0) as f32,
        own_prod as f32 / 64.0,
        enemy_prod as f32 / 64.0,
    ]);

    let planet_limit = state.planets.len().min(max_planets);
    for planet in state.planets.iter().take(planet_limit) {
        encode_planet_into(planet, player, out);
    }
    out.resize(out.len() + (max_planets - planet_limit) * 14, 0.0);

    let fleet_limit = if include_fleets {
        state.fleets.len().min(max_fleets)
    } else {
        0
    };
    for fleet in state.fleets.iter().take(fleet_limit) {
        encode_fleet_into(fleet, player, out);
    }
    out.resize(out.len() + (max_fleets - fleet_limit) * 10, 0.0);
}

fn encode_planet_into(planet: &Planet, player: i8, out: &mut Vec<f32>) {
    let [owner_self, owner_enemy, owner_neutral, owner_other] = owner_rel(planet.owner, player);
    let dx = (planet.x - CENTER) / CENTER;
    let dy = (planet.y - CENTER) / CENTER;
    let dist_center = (dx * dx + dy * dy).sqrt();
    out.extend_from_slice(&[
        1.0,
        owner_self,
        owner_enemy,
        owner_neutral,
        owner_other,
        (planet.x / BOARD_SIZE) as f32,
        (planet.y / BOARD_SIZE) as f32,
        dx as f32,
        dy as f32,
        dist_center as f32,
        (planet.radius / 10.0) as f32,
        ((planet.ships.max(0) as f64).ln_1p() / 8.0) as f32,
        planet.production as f32 / 5.0,
        planet.id as f32 / 512.0,
    ]);
}

fn encode_fleet_into(fleet: &Fleet, player: i8, out: &mut Vec<f32>) {
    let [owner_self, owner_enemy, owner_neutral, _owner_other] = owner_rel(fleet.owner, player);
    out.extend_from_slice(&[
        1.0,
        owner_self,
        owner_enemy,
        owner_neutral,
        (fleet.x / BOARD_SIZE) as f32,
        (fleet.y / BOARD_SIZE) as f32,
        fleet.angle.cos() as f32,
        fleet.angle.sin() as f32,
        ((fleet.ships.max(0) as f64).ln_1p() / 8.0) as f32,
        fleet.from_planet_id as f32 / 512.0,
    ]);
}

#[pyfunction]
fn official_defaults(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let d = PyDict::new(py);
    d.set_item("episode_steps", 500)?;
    d.set_item("act_timeout", 1)?;
    d.set_item("ship_speed", 6.0)?;
    d.set_item("comet_speed", 4.0)?;
    d.set_item("agents", vec![2, 4])?;
    Ok(d.into())
}

#[pymodule]
fn orbit_wars_rs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyConfig>()?;
    m.add_class::<PyBatchSimulator>()?;
    m.add_function(wrap_pyfunction!(official_defaults, m)?)?;
    Ok(())
}
