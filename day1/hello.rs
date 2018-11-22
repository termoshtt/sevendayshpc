extern crate mpi;

fn main() {
    let _universe = mpi::initialize().unwrap();
    println!("Hello, MPI world!");
}
