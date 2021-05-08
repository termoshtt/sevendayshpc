use mpi::point_to_point as p2p;
use mpi::topology::*;

pub struct StdOutEnv<'a> {
    pub rank: i32,
    pub procs: i32,
    pub root_rank: i32,
    pub world: &'a SystemCommunicator,
    pub stdout: Vec<u8>,
}

impl<'a> StdOutEnv<'a> {
    pub fn new(root_rank: i32, world: &'a SystemCommunicator) -> Self {
        let rank = world.rank();
        let procs = world.size();
        let stdout = Vec::new();
        Self {
            rank,
            procs,
            root_rank,
            world,
            stdout,
        }
    }

    pub fn print(&self) {
        if self.root_rank == self.rank {
            print!("{}", std::str::from_utf8(&self.stdout).unwrap());
        }
    }

    pub fn write(&mut self, s: &str, rank: i32) {
        let s_bytes = s.as_bytes();
        let s_len = s_bytes.len();
        if self.rank == self.root_rank {
            if self.rank == rank {
                self.stdout.reserve(s_len);
                for &e in s_bytes.iter() {
                    self.stdout.push(e);
                }
            } else {
                let mut buf_len = 0;
                let process = self.world.process_at_rank(rank);
                p2p::send_receive_into(&true, &process, &mut buf_len, &process);
                let mut buf = vec![0u8; buf_len];
                self.stdout.reserve(buf_len);
                p2p::send_receive_into(&true, &process, &mut buf[..], &process);
                for &e in buf.iter() {
                    self.stdout.push(e);
                }
            }
        } else {
            if self.rank == rank {
                let mut flg = false;
                let root_process = self.world.process_at_rank(self.root_rank);
                p2p::send_receive_into(&s_len, &root_process, &mut flg, &root_process);
                assert!(flg);
                flg = false;
                p2p::send_receive_into(&s_bytes[..], &root_process, &mut flg, &root_process);
                assert!(flg);
            }
        }
    }
}
