use mpi::traits::*;

const L: usize = 8;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();
    let procs = world.size();
    let mysize = L / procs as usize;
    // ローカルなデータ(自分のrank番号で初期化)
    let local = vec![rank; mysize];

    let root_rank = 0;
    let root_process = world.process_at_rank(root_rank);
    if rank == root_rank {
        // 受け取り用のグローバルデータ
        let mut global = vec![0; L];
        // 通信(ランク0番に集める: root)
        root_process.gather_into_root(&local[..], &mut global[..]);

        // ランク0番が代表して表示
        for i in 0..L {
            print!("{}", global[i]);
        }
        println!("");
    } else {
        // 通信(ランク0番に集める: non-root)
        root_process.gather_into(&local[..]);
    }
}
