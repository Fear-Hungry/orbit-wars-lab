use rayon::prelude::*;

use crate::config::Config;
use crate::step::Game;
use crate::types::{GameState, Move, StepOutcome};

#[derive(Clone, Debug)]
pub struct BatchSimulator {
    pub cfg: Config,
    pub games: Vec<Game>,
    pub num_players: usize,
    base_seed: u64,
}

impl BatchSimulator {
    pub fn new(num_envs: usize, num_players: usize, cfg: Config, seed: u64) -> Self {
        assert!(
            num_players == 2 || num_players == 4,
            "Orbit Wars supports only 2 or 4 players"
        );
        let games = (0..num_envs)
            .map(|i| Game::new_training(cfg.clone(), seed + i as u64, num_players))
            .collect();
        Self {
            cfg,
            games,
            num_players,
            base_seed: seed,
        }
    }

    pub fn reset(&mut self, seed: u64) -> Vec<GameState> {
        self.base_seed = seed;
        self.games = (0..self.games.len())
            .map(|i| Game::new_training(self.cfg.clone(), seed + i as u64, self.num_players))
            .collect();
        self.states()
    }

    pub fn reset_from_states(&mut self, states: Vec<GameState>) -> Vec<GameState> {
        assert_eq!(
            states.len(),
            self.games.len(),
            "reset_from_states requires exactly one state per environment"
        );
        for state in &states {
            assert_eq!(
                state.num_players, self.num_players,
                "state player count must match simulator player count"
            );
        }
        self.games = states
            .into_iter()
            .map(|state| Game::from_state(self.cfg.clone(), state))
            .collect();
        self.states()
    }

    pub fn states(&self) -> Vec<GameState> {
        self.games.iter().map(|g| g.state.clone()).collect()
    }

    pub fn step(&mut self, actions: Vec<Vec<Vec<Move>>>) -> Vec<StepOutcome> {
        self.games
            .par_iter_mut()
            .enumerate()
            .map(|(i, game)| {
                let empty: Vec<Vec<Move>> = vec![Vec::new(); self.num_players];
                let a = actions.get(i).unwrap_or(&empty);
                game.step(a)
            })
            .collect()
    }

    pub fn step_with_states(
        &mut self,
        actions: Vec<Vec<Vec<Move>>>,
    ) -> (Vec<StepOutcome>, Vec<GameState>) {
        self.games
            .par_iter_mut()
            .enumerate()
            .map(|(i, game)| {
                let empty: Vec<Vec<Move>> = vec![Vec::new(); self.num_players];
                let a = actions.get(i).unwrap_or(&empty);
                let outcome = game.step(a);
                let state = game.state.clone();
                (outcome, state)
            })
            .unzip()
    }
}
