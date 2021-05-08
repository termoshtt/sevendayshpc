pub mod stdout;

use libc::c_int;

pub fn mpi_dims_create(nnodes: i32, ndims: i32, dims: &mut [i32]) -> i32 {
    unsafe { mpi_sys::MPI_Dims_create(nnodes as c_int, ndims as c_int, dims.as_mut_ptr()) }
}
