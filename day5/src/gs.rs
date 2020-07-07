use std::fs::File;
use std::io::{BufWriter, Write};

const L: usize = 128;
const TOTAL_STEP: usize = 20_000;
const INTERVAL: usize = 200;
const F: f64 = 0.04;
const K: f64 = 0.06075;
const DT: f64 = 0.2;
const DU: f64 = 0.05;
const DV: f64 = 0.1;

const V: usize = L * L;

type VD = Vec<f64>;

fn init(u: &mut VD, v: &mut VD) {
    let d = 3;
    let start = L / 2 - d;
    let end = L / 2 + d;
    for i in start..end {
        for j in start..end {
            u[j + i * L] = 0.7;
        }
    }
    let d = 6;
    let start = L / 2 - d;
    let end = L / 2 + d;
    for i in start..end {
        for j in start..end {
            v[j + i * L] = 0.9;
        }
    }
}

fn calc_u(tu: f64, tv: f64) -> f64 {
    tu * tu * tv - (F + K) * tu
}

fn calc_v(tu: f64, tv: f64) -> f64 {
    -tu * tu * tv + F * (1.0 - tv)
}

fn laplacian(ix: usize, iy: usize, s: &VD) -> f64 {
    let mut ts = 0.0;
    ts += s[ix - 1 + iy * L];
    ts += s[ix + 1 + iy * L];
    ts += s[ix + (iy - 1) * L];
    ts += s[ix + (iy + 1) * L];
    ts -= 4.0 * s[ix + iy * L];
    ts
}

fn calc(u: &mut VD, v: &mut VD, u2: &mut VD, v2: &mut VD) {
    for iy in 1..L-1 {
        for ix in 1..L-1 {
            let mut du;
            let mut dv;
            let i = ix + iy * L;
            du = DU * laplacian(ix, iy, u);
            dv = DV * laplacian(ix, iy, v);
            du += calc_u(u[i], v[i]);
            dv += calc_v(u[i], v[i]);
            u2[i] = u[i] + du * DT;
            v2[i] = v[i] + dv * DT;
        }
    }
}

fn save_as_dat(u: &VD, index: &mut usize) -> Result<(), Box<dyn std::error::Error>> {
    let filename = format!("conf{:03}.dat", index);
    println!("{}", filename);
    let mut f = BufWriter::new(File::create(&filename)?);
    for i in 0..V {
        if i == V - 1 {
            f.write_all(format!("{:.5}", u[i]).as_bytes())?;
        } else {
            f.write_all(format!("{:.5},", u[i]).as_bytes())?;
        }
    }
    *index += 1;
    Ok(())
}

#[allow(unused_must_use)]
fn main() {
    let mut index = 0;

    let mut u = vec![0.0; V];
    let mut v = vec![0.0; V];
    let mut u2 = vec![0.0; V];
    let mut v2 = vec![0.0; V];
    init(&mut u, &mut v);
    for i in 0..TOTAL_STEP {
        if i % 2 == 1 {
            calc(&mut u2, &mut v2, &mut u, &mut v);
        } else {
            calc(&mut u, &mut v, &mut u2, &mut v2);
        }
        if i % INTERVAL == 0 { save_as_dat(&u, &mut index); }
    }
}
