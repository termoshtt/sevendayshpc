use mpi_util::*;

fn main() {
    printf("Hello World!\n");
    printf(&format!("{}st, {}nd, {}rd\n", 1, 2, 3));

    printf!("{}", 1);
}