use mpi::point_to_point as p2p;
use mpi::topology::*;
use mpi::traits::*;
use std::fs::File;
use std::io::{BufWriter, Write};

const L: usize = 128;
const STEP: usize = 100_000;
const DUMP: usize = 1_000;

fn dump(data: &Vec<f64>, index: &mut usize) -> Result<(), Box<dyn std::error::Error>> {
    let filename = format!("data{:03}.dat", index);
    let mut f = BufWriter::new(File::create(&filename)?);
    for i in 0..data.len() {
        f.write_all(format!("{} {}\n", i, data[i]).as_bytes())?;
    }
    *index += 1;
    Ok(())
}

fn dump_mpi(
    local: &Vec<f64>,
    rank: i32,
    procs: i32,
    root_process: &Process<SystemCommunicator>,
    global: &mut Vec<f64>,
    index: &mut usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if rank == 0 {
        root_process.gather_into_root(&local[1..(L / procs as usize) + 1], &mut global[..]);
        dump(global, index)?;
    } else {
        root_process.gather_into(&local[1..(L / procs as usize) + 1]);
    }
    Ok(())
}

#[allow(unused_must_use)]
fn onestep(
    lattice: &mut Vec<f64>,
    orig: &mut Vec<f64>,
    h: f64,
    rank: i32,
    procs: i32,
    world: &SystemCommunicator,
) {
    let size = lattice.len();
    *orig = lattice.clone();
    // ここから通信のためのコード
    let left = (rank - 1 + procs) % procs; // 左のランク番号
    let right = (rank + 1) % procs; // 右のランク番号
    let left_process = world.process_at_rank(left);
    let right_process = world.process_at_rank(right);
    // 右端を右に送って、左端を左から受け取る
    p2p::send_receive_into(
        &lattice[size - 2],
        &right_process,
        &mut orig[0],
        &left_process,
    );
    // 左端を左に送って、右端を右から受け取る
    p2p::send_receive_into(
        &lattice[1],
        &left_process,
        &mut orig[size - 1],
        &right_process,
    );

    //あとはシリアル版と同じ
    for i in 1..size - 1 {
        lattice[i] += (orig[i - 1] - 2.0 * orig[i] + orig[i + 1]) * 0.5 * h;
    }
}

#[allow(dead_code)]
fn uniform_heating(
    lattice: &mut Vec<f64>,
    rank: i32,
    procs: i32,
    world: &SystemCommunicator,
) -> Result<(), Box<dyn std::error::Error>> {
    let h = 0.2;
    let q = 1.0;
    let mut index = 0;
    let mut orig = lattice.clone();
    let root_process = world.process_at_rank(0);
    let mut global = if rank == 0 { vec![0.0; L] } else { vec![] };
    for i in 0..STEP {
        onestep(lattice, &mut orig, h, rank, procs, world);
        for s in lattice.iter_mut() {
            *s += q * h;
        }
        if rank == 0 {
            lattice[0] = 0.0;
        }
        if rank == procs - 1 {
            let size = lattice.len();
            lattice[size - 2] = 0.0;
        }
        if i % DUMP == 0 {
            dump_mpi(lattice, rank, procs, &root_process, &mut global, &mut index)?;
        }
    }
    Ok(())
}

#[allow(dead_code)]
fn fixed_temperature(
    lattice: &mut Vec<f64>,
    rank: i32,
    procs: i32,
    world: &SystemCommunicator,
) -> Result<(), Box<dyn std::error::Error>> {
    let h = 0.01;
    let q = 1.0;
    let s = L / procs as usize;
    let mut index = 0;
    let mut orig = lattice.clone();
    let root_process = world.process_at_rank(0);
    let mut global = if rank == 0 { vec![0.0; L] } else { vec![] };
    for i in 0..STEP {
        onestep(lattice, &mut orig, h, rank, procs, world);
        if rank == (L / 4 / s) as i32 {
            lattice[L / 4 + 1 - rank as usize * s] = q;
        }
        if rank == (3 * L / 4 / s) as i32 {
            lattice[3 * L / 4 + 1 - rank as usize * s] = -q;
        }
        if i % DUMP == 0 {
            dump_mpi(lattice, rank, procs, &root_process, &mut global, &mut index)?;
        }
    }
    Ok(())
}

#[allow(unused_must_use)]
fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();
    let procs = world.size();

    let mysize = L / procs as usize + 2;
    let mut local = vec![0.0; mysize];
    uniform_heating(&mut local, rank, procs, &world);
    //fixed_temperature(&mut local, rank, procs, &world);
}
