use mpi::collective::SystemOperation;
use mpi::traits::*;
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
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();
    let procs = world.size();
    let pi = calc_pi(rank as u8);
    let pi2 = pi * pi;
    let mut pi_sum = 0.0;
    let mut pi2_sum = 0.0;
    println!("{}", pi);
    world.all_reduce_into(&pi, &mut pi_sum, &SystemOperation::sum());
    world.all_reduce_into(&pi2, &mut pi2_sum, &SystemOperation::sum());
    let pi_ave = pi_sum / procs as f64;
    let pi_var = pi2_sum / (procs - 1) as f64 - pi_sum * pi_sum / procs as f64 / (procs - 1) as f64;
    let pi_stdev = pi_var.sqrt();
    world.barrier();
    if rank == 0 {
        println!("pi = {} +- {}", pi_ave, pi_stdev);
    }
}
