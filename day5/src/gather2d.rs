use mpi::topology::*;
use mpi::traits::*;
use mpi_util::*;
use mpi_util::stdout::StdOutEnv;

const L: usize = 8;

#[allow(dead_code)]
struct MPIinfo {
    rank: i32,
    procs: i32,
    gx: i32,
    gy: i32,
    local_grid_x: i32,
    local_grid_y: i32,
    local_size_x: usize,
    local_size_y: usize,
}

impl MPIinfo {
    pub fn new(world: &SystemCommunicator) -> Self { // void setup_info(MPIinfo &mi);
        let rank = world.rank();
        let procs = world.size();
        let mut d2 = vec![0; 2];
        let _ = mpi_dims_create(procs, 2, &mut d2);
        let gx = d2[0];
        let gy = d2[1];
        let local_grid_x = rank % gx;
        let local_grid_y = rank / gx;
        let local_size_x = L / gx as usize;
        let local_size_y = L / gy as usize;
        Self { rank, procs, gx, gy, local_grid_x, local_grid_y, local_size_x, local_size_y, }
    }

    pub fn init(&self, local_data: &mut Vec<i32>) {
        let offset = self.local_size_x * self.local_size_y * self.rank as usize;
        for iy in 0..self.local_size_y {
            for ix in 0..self.local_size_x {
                let index = ix + 1 + (iy + 1) * (self.local_size_x + 2);
                let value = ix + iy * self.local_size_x + offset;
                local_data[index] = value as i32;
            }
        }
    }

    pub fn dump_local_sub(&self, rank: i32, local_data: &mut Vec<i32>, out: &mut StdOutEnv) {
        out.write(&format!("rank = {}\n", rank), rank);
        for iy in 0..self.local_size_y+2 {
            for ix in 0..self.local_size_x+2 {
                let index = ix + iy * (self.local_size_x + 2);
                out.write(&format!(" {:03}", local_data[index]), rank);
            }
            out.write("\n", rank);
        }
        out.write("\n", rank);
    }

    pub fn dump_local(&self, local_data: &mut Vec<i32>, out: &mut StdOutEnv) {
        for i in 0..self.procs {
            out.world.barrier();
            self.dump_local_sub(i, local_data, out);
        }
    }

    pub fn reordering(&self, v: &mut Vec<i32>) {
        let v2 = v.clone();
        let lx = self.local_size_x;
        let ly = self.local_size_y;
        let mut i = 0;
        for r in 0..self.procs as usize {
            let rx = r % self.gx as usize;
            let ry = r / self.gx as usize;
            let sx = rx * lx;
            let sy = ry * ly;
            for iy in 0..ly {
                for ix in 0..lx {
                    let index = sx + ix + (sy + iy) * L;
                    v[index] = v2[i];
                    i += 1;
                }
            }
        }
    }

    pub fn gather(&self, local_data: &Vec<i32>, out: &mut StdOutEnv) {
        let root_process = out.world.process_at_rank(0);
        let lx = self.local_size_x;
        let ly = self.local_size_y;
        let mut sendbuf = vec![0i32; lx * ly];
        for iy in 0..ly {
            for ix in 0..lx {
                let index_from = (ix + 1) + (iy + 1) * (lx + 2);
                let index_to = ix + iy * lx;
                sendbuf[index_to] = local_data[index_from];
            }
        }
        if self.rank == 0 {
            let mut recvbuf = vec![0i32; lx * ly * self.procs as usize];
            root_process.gather_into_root(&sendbuf[..], &mut recvbuf[..]);
            out.write("Before reordering\n", out.rank);
            dump_global(&recvbuf, out);
            self.reordering(&mut recvbuf);
            out.write("After reordering\n", out.rank);
            dump_global(&recvbuf, out);
        } else {
            root_process.gather_into(&sendbuf[..]);
        }
    }
}

fn dump_global(global_data: &Vec<i32>, out: &mut StdOutEnv) {
    for iy in 0..L {
        for ix in 0..L {
            out.write(&format!(" {:03}", global_data[ix + iy * L]), out.rank);
        }
        out.write("\n", out.rank);
    }
    out.write("\n", out.rank);
}

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let mi = MPIinfo::new(&world);
    let mut out = StdOutEnv::new(0, &world);
    // ローカルデータの確保
    let mut local_data = vec![0; (mi.local_size_x + 2) * (mi.local_size_y + 2)];
    // ローカルデータの初期化
    mi.init(&mut local_data);
    // ローカルデータの表示
    mi.dump_local(&mut local_data, &mut out);
    // ローカルデータを集約してグローバルデータに
    mi.gather(&mut local_data, &mut out);
    out.print();
}