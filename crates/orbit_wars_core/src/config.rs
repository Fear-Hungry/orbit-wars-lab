use serde::{Deserialize, Serialize};

pub const BOARD_SIZE: f64 = 100.0;
pub const CENTER: f64 = BOARD_SIZE / 2.0;
pub const SUN_RADIUS: f64 = 10.0;
pub const ROTATION_RADIUS_LIMIT: f64 = 50.0;
pub const COMET_RADIUS: f64 = 1.0;
pub const COMET_PRODUCTION: i32 = 1;
pub const PLANET_CLEARANCE: f64 = 7.0;
pub const COMET_SPAWN_STEPS: [i32; 5] = [50, 150, 250, 350, 450];

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    pub episode_steps: i32,
    pub act_timeout: f64,
    pub ship_speed: f64,
    pub comet_speed: f64,
    pub enable_comets: bool,
    pub max_planets: usize,
    pub max_fleets: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            episode_steps: 500,
            act_timeout: 1.0,
            ship_speed: 6.0,
            comet_speed: 4.0,
            enable_comets: true,
            max_planets: 96,
            max_fleets: 4096,
        }
    }
}
