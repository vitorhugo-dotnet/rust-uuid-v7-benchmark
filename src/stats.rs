use crate::{fair, reduce_entropy};
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct Stat {
    pub volume: usize,
    pub trials: usize,
    pub collisions: usize,
    pub trials_with_collision: usize,
    pub theoretical_probability: f64,
}

/// Monte Carlo demonstration at ONE fixed millisecond. Not a prediction of
/// real-world collision rates, and not a test of cryptographic randomness.
pub fn simulate(volumes: &[usize], trials: usize, bits: u8, seed: u64, timestamp_ms: u64) -> Vec<Stat> {
    let space = 2f64.powi(bits as i32);
    let mut results = Vec::new();
    for &volume in volumes {
        let mut collisions = 0;
        let mut trials_with_collision = 0;
        for trial in 0..trials {
            let mut seen = HashSet::<u128>::with_capacity(volume);
            let trial_seed = fair::splitmix64(seed ^ (trial as u64));
            let mut this_trial = 0;
            for i in 0..volume {
                let id = fair::synthetic_uuid(trial_seed, i as u64, timestamp_ms);
                if !seen.insert(reduce_entropy(id, bits)) {
                    collisions += 1;
                    this_trial += 1;
                }
            }
            if this_trial > 0 { trials_with_collision += 1; }
        }
        let lambda = (volume as f64 * (volume as f64 - 1.0)) / (2.0 * space);
        results.push(Stat {
            volume, trials, collisions, trials_with_collision,
            theoretical_probability: -(-lambda).exp_m1(),
        });
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tiny_space_collides() {
        let data = simulate(&[1_000], 3, 8, 42, 1_729_280_000_000);
        assert_eq!(data[0].trials_with_collision, 3);
        assert!(data[0].collisions > 0);
    }
    #[test]
    fn no_collisions_with_one_value() {
        assert_eq!(simulate(&[1], 5, 16, 42, 1).pop().unwrap().collisions, 0);
    }
}