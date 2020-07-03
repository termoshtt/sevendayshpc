use mpi::traits::*;

fn process_file(index: i32, rank: i32) {
    println!("Rank={:03} File={:03}", rank, index);
}

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();
    let procs = world.size();
    let max_file = 100;
    let mut i = rank;
    while i < max_file {
        process_file(i, rank);
        i += procs;
    }
}
