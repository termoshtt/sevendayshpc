use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;

const N: usize = 100000;

#[repr(align(32))]
struct A {
    data: Vec<f64>,
}

const DT: f64 = 0.01;

fn energy(vx: &[f64], vy: &[f64], vz: &[f64]) -> f64 {
    let mut e = 0.0;
    for i in 0..N {
        e += vx[i] * vx[i];
        e += vy[i] * vy[i];
        e += vz[i] * vz[i];
    }
    e * 0.5 / N as f64
}

#[allow(dead_code)]
fn calc_euler(
    vx: &mut [f64],
    vy: &mut [f64],
    vz: &mut [f64],
    rx: &mut [f64],
    ry: &mut [f64],
    rz: &mut [f64],
    bx: f64,
    by: f64,
    bz: f64,
) {
    for i in 0..N {
        let px = vy[i] * bz - vz[i] * by;
        let py = vz[i] * bx - vx[i] * bz;
        let pz = vx[i] * by - vy[i] * bx;
        vx[i] += px * DT;
        vy[i] += py * DT;
        vz[i] += pz * DT;
        rx[i] = rx[i] + vx[i] * DT;
        ry[i] = ry[i] + vy[i] * DT;
        rz[i] = rz[i] + vz[i] * DT;
    }
}

#[allow(dead_code)]
fn calc_rk2(
    vx: &mut [f64],
    vy: &mut [f64],
    vz: &mut [f64],
    rx: &mut [f64],
    ry: &mut [f64],
    rz: &mut [f64],
    bx: f64,
    by: f64,
    bz: f64,
) {
    for i in 0..N {
        let px = vy[i] * bz - vz[i] * by;
        let py = vz[i] * bx - vx[i] * bz;
        let pz = vx[i] * by - vy[i] * bx;
        let vcx = vx[i] + px * DT * 0.5;
        let vcy = vy[i] + py * DT * 0.5;
        let vcz = vz[i] + pz * DT * 0.5;
        let px2 = vcy * bz - vcz * by;
        let py2 = vcz * bx - vcx * bz;
        let pz2 = vcx * by - vcy * bx;
        vx[i] += px2 * DT;
        vy[i] += py2 * DT;
        vz[i] += pz2 * DT;
        rx[i] += vx[i] * DT;
        ry[i] += vy[i] * DT;
        rz[i] += vz[i] * DT;
    }
}

fn init(
    vx: &mut [f64],
    vy: &mut [f64],
    vz: &mut [f64],
    rx: &mut [f64],
    ry: &mut [f64],
    rz: &mut [f64],
) -> (f64, f64, f64) {
    let ud = Uniform::new(0.0, 1.0);
    let mut rng = thread_rng();
    for i in 0..N {
        let z = ud.sample(&mut rng) * 2.0 - 1.0;
        let s = ud.sample(&mut rng) * std::f64::consts::PI;
        vx[i] = (1.0 - z * z).sqrt() * s.cos();
        vy[i] = (1.0 - z * z).sqrt() * s.sin();
        vz[i] = z;
        rx[i] = 0.0;
        ry[i] = 0.0;
        rz[i] = 0.0;
    }
    let z = ud.sample(&mut rng) * 2.0 - 1.0;
    let s = ud.sample(&mut rng) * std::f64::consts::PI;
    let bx = (1.0 - z * z).sqrt() * s.cos();
    let by = (1.0 - z * z).sqrt() * s.sin();
    let bz = z;

    (bx, by, bz)
}

fn dump(rx: &[f64], ry: &[f64], rz: &[f64]) {
    for i in 0..N {
        print!("{} ", rx[i]);
        print!("{} ", ry[i]);
        println!("{}", rz[i]);
    }
}

fn main() {
    let mut vx = A {
        data: Vec::with_capacity(N),
    }
    .data;
    let mut vy = A {
        data: Vec::with_capacity(N),
    }
    .data;
    let mut vz = A {
        data: Vec::with_capacity(N),
    }
    .data;
    let mut rx = A {
        data: Vec::with_capacity(N),
    }
    .data;
    let mut ry = A {
        data: Vec::with_capacity(N),
    }
    .data;
    let mut rz = A {
        data: Vec::with_capacity(N),
    }
    .data;

    unsafe {
        vx.set_len(N);
        vy.set_len(N);
        vz.set_len(N);
        rx.set_len(N);
        ry.set_len(N);
        rz.set_len(N);
    }

    let (bx, by, bz) = init(&mut vx, &mut vy, &mut vz, &mut rx, &mut ry, &mut rz);
    let mut t = 0.0;
    for i in 0..10000 {
        // calc_euler(&mut v, &mut r, bx, by, bz);
        calc_rk2(
            &mut vx, &mut vy, &mut vz, &mut rx, &mut ry, &mut rz, bx, by, bz,
        );
        t += DT;
        if i % 1000 == 0 {
            println!("{} {}", t, energy(&vx, &vy, &vz));
        }
    }
    // dump(&rx, &ry, &rz);
}
