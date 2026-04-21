use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct Planet {
    pub id: i32,
    pub owner: i8,
    pub x: f64,
    pub y: f64,
    pub radius: f64,
    pub ships: i32,
    pub production: i32,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct Fleet {
    pub id: i32,
    pub owner: i8,
    pub x: f64,
    pub y: f64,
    pub angle: f64,
    pub from_planet_id: i32,
    pub ships: i32,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct Move {
    pub from_planet_id: i32,
    pub angle: f64,
    pub ships: i32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CometGroup {
    pub planet_ids: Vec<i32>,
    pub paths: Vec<Vec<[f64; 2]>>,
    pub path_index: i32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GameState {
    pub step: i32,
    pub num_players: usize,
    pub angular_velocity: f64,
    pub planets: Vec<Planet>,
    pub initial_planets: Vec<Planet>,
    pub fleets: Vec<Fleet>,
    pub next_fleet_id: i32,
    pub comets: Vec<CometGroup>,
    pub comet_planet_ids: Vec<i32>,
    pub done: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StepOutcome {
    pub rewards: Vec<f32>,
    pub done: bool,
    pub scores: Vec<i32>,
}

impl Planet {
    pub fn xy(&self) -> [f64; 2] {
        [self.x, self.y]
    }
}

impl Fleet {
    pub fn xy(&self) -> [f64; 2] {
        [self.x, self.y]
    }
}
