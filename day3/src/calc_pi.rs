use rand::distributions::{Distribution, Uniform};
use rand_core::SeedableRng;

const TRIAL: usize = 100_000;

fn calc_pi(seed: u8) -> f64 {
    let mut rng: rand::rngs::StdRng = SeedableRng::from_seed([seed; 32]);
    let ud = Uniform::<f64>::new(0.0, 1.0);
    let mut n: usize = 0;
    for _ in 0..TRIAL {
        let x = ud.sample(&mut rng);
        let y = ud.sample(&mut rng);
        if x * x + y * y < 1.0 {
            n += 1;
        }
    }
    4.0 * n as f64 / TRIAL as f64
}

fn main() {
    let pi = calc_pi(0);
    println!("{}", pi);
}
