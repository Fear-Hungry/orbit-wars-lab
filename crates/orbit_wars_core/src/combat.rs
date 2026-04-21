use std::collections::HashMap;

use crate::types::Planet;

/// Resolve all incoming fleet forces against a planet.
///
/// Forces are first aggregated by owner. The strongest owner fights the
/// second-strongest. If no force survives, the planet is unchanged. If a force
/// survives, it reinforces a friendly planet or attacks an enemy/neutral one.
pub fn resolve_planet_combat(planet: &mut Planet, arrivals: &[(i8, i32)]) {
    if arrivals.is_empty() {
        return;
    }

    let mut by_owner: HashMap<i8, i32> = HashMap::new();
    for (owner, ships) in arrivals.iter().copied() {
        if ships > 0 {
            *by_owner.entry(owner).or_insert(0) += ships;
        }
    }
    if by_owner.is_empty() {
        return;
    }

    let mut forces: Vec<(i8, i32)> = by_owner.into_iter().collect();
    forces.sort_by(|a, b| b.1.cmp(&a.1));

    let (winner_owner, top) = forces[0];
    let second = forces.get(1).map(|x| x.1).unwrap_or(0);
    let survivor = top - second;

    if survivor <= 0 {
        return;
    }

    if planet.owner == winner_owner {
        planet.ships += survivor;
    } else if survivor > planet.ships {
        planet.owner = winner_owner;
        planet.ships = survivor - planet.ships;
    } else {
        planet.ships -= survivor;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stronger_enemy_captures() {
        let mut p = Planet {
            id: 0,
            owner: 0,
            x: 0.0,
            y: 0.0,
            radius: 1.0,
            ships: 10,
            production: 1,
        };
        resolve_planet_combat(&mut p, &[(1, 15)]);
        assert_eq!(p.owner, 1);
        assert_eq!(p.ships, 5);
    }

    #[test]
    fn friendly_reinforces() {
        let mut p = Planet {
            id: 0,
            owner: 0,
            x: 0.0,
            y: 0.0,
            radius: 1.0,
            ships: 10,
            production: 1,
        };
        resolve_planet_combat(&mut p, &[(0, 5)]);
        assert_eq!(p.owner, 0);
        assert_eq!(p.ships, 15);
    }

    #[test]
    fn strongest_force_beats_second_strongest() {
        let mut p = Planet {
            id: 0,
            owner: -1,
            x: 0.0,
            y: 0.0,
            radius: 1.0,
            ships: 2,
            production: 1,
        };
        resolve_planet_combat(&mut p, &[(1, 10), (2, 7), (3, 4)]);
        assert_eq!(p.owner, 1);
        assert_eq!(p.ships, 1);
    }
}
