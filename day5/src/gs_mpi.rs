use std::fs::File;
use std::io::{BufWriter, Write};
use mpi::point_to_point as p2p;
use mpi::topology::*;
use mpi::traits::*;
use mpi_util::*;

const L: usize = 128;
const TOTAL_STEP: usize = 20_000;
const INTERVAL: usize = 200;
const F: f64 = 0.04;
const K: f64 = 0.06075;
const DT: f64 = 0.2;
const DU: f64 = 0.05;
const DV: f64 = 0.1;

type VD = Vec<f64>;

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

    // 自分から見て(dx,dy)だけずれたプロセスのrankを返す
    pub fn get_rank(&self, dx: i32, dy: i32) -> i32 {
        let rx = (self.local_grid_x + dx + self.gx) % self.gx;
        let ry = (self.local_grid_y + dy + self.gy) % self.gy;
        rx + ry * self.gx
    }

    // 自分の領域に含まれるか
    pub fn is_inside(&self, x: usize, y: usize) -> bool {
        let sx = self.local_size_x * self.local_grid_x as usize;
        let sy = self.local_size_y * self.local_grid_y as usize;
        let ex = sx + self.local_size_x;
        let ey = sy + self.local_size_y;
        if x < sx || x >= ex || y < sy || y >= ey  { return false; }
        true
    }

    // グローバル座標をローカルインデックスに
    pub fn g2i(&self, gx: usize, gy: usize) -> usize {
        let sx = self.local_size_x * self.local_grid_x as usize;
        let sy = self.local_size_y * self.local_grid_y as usize;
        let x = gx - sx;
        let y = gy - sy;
        x + 1 + (y + 1) * (self.local_size_x + 2)
    }

    pub fn init(&self, u: &mut VD, v: &mut VD) {
        let d = 3;
        let start = L / 2 - d;
        let end = L / 2 + d;
        for i in start..end {
            for j in start..end {
                if !self.is_inside(i, j) { continue; }
                let k = self.g2i(i, j);
                u[k] = 0.7;
            }
        }
        let d = 6;
        let start = L / 2 - d;
        let end = L / 2 + d;
        for i in start..end {
            for j in start..end {
                if !self.is_inside(i, j) { continue; }
                let k = self.g2i(i, j);
                v[k] = 0.9;
            }
        }
    }
    
    fn laplacian(&self, ix: usize, iy: usize, s: &VD) -> f64 {
        let mut ts = 0.0;
        let l = self.local_size_x + 2;
        ts += s[ix - 1 + iy * l];
        ts += s[ix + 1 + iy * l];
        ts += s[ix + (iy - 1) * l];
        ts += s[ix + (iy + 1) * l];
        ts -= 4.0 * s[ix + iy * l];
        ts
    }

    fn calc(&self, u: &mut VD, v: &mut VD, u2: &mut VD, v2: &mut VD) {
        let lx = self.local_size_x + 2;
        let ly = self.local_size_y + 2;
        for iy in 1..ly-1 {
            for ix in 1..lx-1 {
                let mut du;
                let mut dv;
                let i = ix + iy * lx;
                du = DU * self.laplacian(ix, iy, u);
                dv = DV * self.laplacian(ix, iy, v);
                du += calc_u(u[i], v[i]);
                dv += calc_v(u[i], v[i]);
                u2[i] = u[i] + du * DT;
                v2[i] = v[i] + dv * DT;
            }
        }
    }

    // 送られてきたデータを再配置する
    pub fn reordering(&self, v: &mut VD) {
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

    #[allow(unused_must_use)]
    pub fn save_as_dat_mpi(&self, local_data: &VD, index: &mut usize, world: &SystemCommunicator) {
        let root_process = world.process_at_rank(0);
        let lx = self.local_size_x;
        let ly = self.local_size_y;
        let mut sendbuf = vec![0.0; lx * ly];
        for iy in 0..ly {
            for ix in 0..lx {
                let index_from = (ix + 1) + (iy + 1) * (lx + 2);
                let index_to = ix + iy * lx;
                sendbuf[index_to] = local_data[index_from];
            }
        }
        if self.rank == 0 {
            let mut recvbuf = vec![0.0; lx * ly * self.procs as usize];
            root_process.gather_into_root(&sendbuf[..], &mut recvbuf[..]);
            self.reordering(&mut recvbuf);
            save_as_dat(&recvbuf, index);
        } else {
            root_process.gather_into(&sendbuf[..]);
        }
    }

    pub fn sendrecv_x(&self, local_data: &mut VD, world: &SystemCommunicator) {
        let lx = self.local_size_x;
        let ly = self.local_size_y;
        let mut sendbuf = vec![0.0; ly];
        let mut recvbuf = vec![0.0; ly];
        let left = self.get_rank(-1, 0);
        let right = self.get_rank(1, 0);
        let left_process = world.process_at_rank(left);
        let right_process = world.process_at_rank(right);
        for i in 0..ly {
            let index = lx + (i + 1) * (lx + 2);
            sendbuf[i] = local_data[index];
        }
        p2p::send_receive_into(&sendbuf[..], &right_process, &mut recvbuf[..], &left_process);
        for i in 0..ly {
            let index = (i + 1) * (lx + 2);
            local_data[index] = recvbuf[i];
        }

        for i in 0..ly {
            let index = 1 + (i + 1) * (lx + 2);
            sendbuf[i] = local_data[index];
        }
        p2p::send_receive_into(&sendbuf[..], &left_process, &mut recvbuf[..], &right_process);
        for i in 0..ly {
            let index = lx + 1 + (i + 1) * (lx + 2);
            local_data[index] = recvbuf[i];
        }
    }

    pub fn sendrecv_y(&self, local_data: &mut VD, world: &SystemCommunicator) {
        let lx = self.local_size_x;
        let ly = self.local_size_y;
        let mut sendbuf = vec![0.0; lx + 2];
        let mut recvbuf = vec![0.0; lx + 2];
        let up = self.get_rank(0, -1);
        let down = self.get_rank(0, 1);
        let up_process = world.process_at_rank(up);
        let down_process = world.process_at_rank(down);
        // 上に投げて下から受け取る
        for i in 0..lx+2 {
            let index = i + 1 * (lx + 2);
            sendbuf[i] = local_data[index];
        }
        p2p::send_receive_into(&sendbuf[..], &up_process, &mut recvbuf[..], &down_process);
        for i in 0..lx+2 {
            let index = i + (ly + 1) * (lx + 2);
            local_data[index] = recvbuf[i];
        }
        // 下に投げて上から受け取る
        for i in 0..lx+2 {
            let index = i + ly * (lx + 2);
            sendbuf[i] = local_data[index];
        }
        p2p::send_receive_into(&sendbuf[..], &down_process, &mut recvbuf[..], &up_process);
        for i in 0..lx+2 {
            let index = i + 0 * (lx + 2);
            local_data[index] = recvbuf[i];
        }
    }

    pub fn sendrecv(&self, u: &mut VD, v: &mut VD, world: &SystemCommunicator) {
        self.sendrecv_x(u, world);
        self.sendrecv_y(u, world);
        self.sendrecv_x(v, world);
        self.sendrecv_y(v, world);
    }
}

fn calc_u(tu: f64, tv: f64) -> f64 {
    tu * tu * tv - (F + K) * tu
}

fn calc_v(tu: f64, tv: f64) -> f64 {
    -tu * tu * tv + F * (1.0 - tv)
}

fn save_as_dat(u: &VD, index: &mut usize) -> Result<(), Box<dyn std::error::Error>> {
    let filename = format!("conf{:03}.dat", index);
    println!("{}", filename);
    let mut f = BufWriter::new(File::create(&filename)?);
    for i in 0..L*L {
        if i == L*L - 1 {
            f.write_all(format!("{:.5}", u[i]).as_bytes())?;
        } else {
            f.write_all(format!("{:.5},", u[i]).as_bytes())?;
        }
    }
    *index += 1;
    Ok(())
}

fn main() {
    let mut index = 0;

    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let mi = MPIinfo::new(&world);
    let v_size =  (mi.local_size_x + 2) * (mi.local_size_y + 2);
    let mut u = vec![0.0; v_size];
    let mut v = vec![0.0; v_size];
    let mut u2 = vec![0.0; v_size];
    let mut v2 = vec![0.0; v_size];
    mi.init(&mut u, &mut v);
    for i in 0..TOTAL_STEP {
        if i % 2 == 1 {
            mi.sendrecv(&mut u2, &mut v2, &world);
            mi.calc(&mut u2, &mut v2, &mut u, &mut v);
        } else {
            mi.sendrecv(&mut u, &mut v, &world);
            mi.calc(&mut u, &mut v, &mut u2, &mut v2);
        }
        if i % INTERVAL == 0 { mi.save_as_dat_mpi(&u, &mut index, &world); }
    }
}