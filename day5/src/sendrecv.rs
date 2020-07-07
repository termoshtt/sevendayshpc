use mpi::point_to_point as p2p;
use mpi::topology::*;
use mpi::traits::*;
use mpi_util::*;
use mpi_util::stdout::StdOutEnv;

const L: usize = 8;

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

    pub fn sendrecv_x(&self, local_data: &mut Vec<i32>, world: &SystemCommunicator) {
        let lx = self.local_size_x;
        let ly = self.local_size_y;
        let mut sendbuf = vec![0; ly];
        let mut recvbuf = vec![0; ly];
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

    pub fn sendrecv_y(&self, local_data: &mut Vec<i32>, world: &SystemCommunicator) {
        let lx = self.local_size_x;
        let ly = self.local_size_y;
        let mut sendbuf = vec![0; lx + 2];
        let mut recvbuf = vec![0; lx + 2];
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
    if mi.rank == 0 { out.write("# 通信前\n", mi.rank); }
    mi.dump_local(&mut local_data, &mut out);
    // x方向に通信
    mi.sendrecv_x(&mut local_data, &world);
    if mi.rank == 0 { out.write("# 左右の通信後\n", mi.rank); }
    mi.dump_local(&mut local_data, &mut out);
    // y方向に通信
    mi.sendrecv_y(&mut local_data, &world);
    if mi.rank == 0 { out.write("# 上下の通信終了後 (これで斜め方向も完了)\n", mi.rank); }
    mi.dump_local(&mut local_data, &mut out);
    out.print();
}