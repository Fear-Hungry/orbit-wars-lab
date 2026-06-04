use orbit_wars_core::{BatchSimulator, Config, Move};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::types::PyDict;

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
