extern crate mpi;
use mpi::traits::*;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();
    println!("Hello, My rank is {}!", rank);
}
