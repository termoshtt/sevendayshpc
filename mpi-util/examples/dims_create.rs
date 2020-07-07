use mpi_util::*;

fn main() {
    let universe = mpi::initialize().unwrap();
    let _ = universe.world();
    let mut v = vec![0; 4];
    let _ = mpi_dims_create(16, 4, &mut v);
    println!("{:?}", v); // [2, 2, 2, 2]
}