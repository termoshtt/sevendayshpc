extern crate rand;
extern crate sfmt;
use rand::distributions::*;
use rand::FromEntropy;

const TRIAL: usize = 100000;

fn calc_pi() -> f64 {
    let mut rng = sfmt::SFMT::from_entropy();
    let dist = Uniform::new(0.0, 1.0);
    let mut n = 0;
    for _ in 0..TRIAL {
        let x = dist.sample(&mut rng);
        let y = dist.sample(&mut rng);
        if x * x + y * y < 1.0 {
            n += 1;
        }
    }
    4.0 * n as f64 / TRIAL as f64
}

fn main() {
    println!("PI = {}", calc_pi());
}
