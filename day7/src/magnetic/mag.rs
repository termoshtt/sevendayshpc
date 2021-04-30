use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;

const N: usize = 100000;

#[derive(Clone, Copy)]
struct V {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

const DT: f64 = 0.01;

fn energy(v: &[V]) -> f64 {
    let mut e = 0.0;
    for i in 0..N {
        e += v[i].x * v[i].x;
        e += v[i].y * v[i].y;
        e += v[i].z * v[i].z;
    }
    e * 0.5 / N as f64
}

#[allow(dead_code)]
fn calc_euler(v: &mut [V], r: &mut [V], bx: f64, by: f64, bz: f64) {
    for i in 0..N {
        let px = v[i].y * bz - v[i].z * by;
        let py = v[i].z * bx - v[i].x * bz;
        let pz = v[i].x * by - v[i].y * bx;
        v[i].x += px * DT;
        v[i].y += py * DT;
        v[i].z += pz * DT;
        r[i].x = r[i].x + v[i].x * DT;
        r[i].y = r[i].y + v[i].y * DT;
        r[i].z = r[i].z + v[i].z * DT;
    }
}

fn calc_rk2(v: &mut [V], r: &mut [V], bx: f64, by: f64, bz: f64) {
    for i in 0..N {
        let px = v[i].y * bz - v[i].z * by;
        let py = v[i].z * bx - v[i].x * bz;
        let pz = v[i].x * by - v[i].y * bx;
        let vcx = v[i].x + px * DT * 0.5;
        let vcy = v[i].y + py * DT * 0.5;
        let vcz = v[i].z + pz * DT * 0.5;
        let px2 = vcy * bz - vcz * by;
        let py2 = vcz * bx - vcx * bz;
        let pz2 = vcx * by - vcy * bx;
        v[i].x += px2 * DT;
        v[i].y += py2 * DT;
        v[i].z += pz2 * DT;
        r[i].x += v[i].x * DT;
        r[i].y += v[i].y * DT;
        r[i].z += v[i].z * DT;
    }
}

fn init(v: &mut [V], r: &mut [V]) -> (f64, f64, f64) {
    let ud = Uniform::new(0.0, 1.0);
    let mut rng = thread_rng();
    for i in 0..N {
        let z = ud.sample(&mut rng) * 2.0 - 1.0;
        let s = ud.sample(&mut rng) * std::f64::consts::PI;
        v[i].x = (1.0 - z * z).sqrt() * s.cos();
        v[i].y = (1.0 - z * z).sqrt() * s.sin();
        v[i].z = z;
        r[i].x = 0.0;
        r[i].y = 0.0;
        r[i].z = 0.0;
    }
    let z = ud.sample(&mut rng) * 2.0 - 1.0;
    let s = ud.sample(&mut rng) * std::f64::consts::PI;
    let bx = (1.0 - z * z).sqrt() * s.cos();
    let by = (1.0 - z * z).sqrt() * s.sin();
    let bz = z;

    (bx, by, bz)
}

fn dump(r: &[V]) {
    for i in 0..N {
        print!("{} ", r[i].x);
        print!("{} ", r[i].y);
        println!("{}", r[i].z);
    }
}

fn main() {
    let mut v: Vec<V> = Vec::with_capacity(N);
    let mut r: Vec<V> = Vec::with_capacity(N);
    unsafe {
        v.set_len(N);
        r.set_len(N);
    }
    let (bx, by, bz) = init(&mut v, &mut r);
    let mut t = 0.0;
    for i in 0..10000 {
        // calc_euler(&mut v, &mut r, bx, by, bz);
        calc_rk2(&mut v, &mut r, bx, by, bz);
        t += DT;
        if i % 1000 == 0 {
            println!("{} {}", t, energy(&v));
        } 
    }
    // dump(&r);
}