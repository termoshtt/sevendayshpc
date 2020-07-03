use std::fs::File;
use std::io::{BufWriter, Write};

const L: usize = 128;
const STEP: usize = 100_000;
const DUMP: usize = 1_000;

fn onestep(lattice: &mut Vec<f64>, orig: &mut Vec<f64>, h: f64) {
    *orig = lattice.clone();
    for i in 1..L-1 {
        lattice[i] += (orig[i - 1] - 2.0 * orig[i] + orig[i + 1]) * 0.5 * h;
    }
    // For Periodic Boundary
    lattice[0] += (orig[L - 1] - 2.0 * lattice[0] + orig[1]) * 0.5 * h;
    lattice[L - 1] += (orig[L - 2] - 2.0 * lattice[L - 1] + orig[0]) * 0.5 * h;
}

fn dump(data: &Vec<f64>, index: &mut usize) -> Result<(), Box<dyn std::error::Error>> {
    let filename = format!("data{:03}.dat", index);
    let mut f = BufWriter::new(File::create(&filename)?);
    for i in 0..data.len() {
        f.write_all(format!("{} {}\n", i, data[i]).as_bytes())?;
    }
    *index += 1;
    Ok(())
}

#[allow(dead_code)]
fn fixed_temperature(lattice: &mut Vec<f64>) -> Result<(), Box<dyn std::error::Error>> {
    let h = 0.01;
    let q = 1.0;
    let mut index = 0;
    let mut orig = lattice.clone();
    for i in 0..STEP {
        onestep(lattice, &mut orig, h);
        lattice[L / 4] = q;
        lattice[3 * L / 4] = -q;
        if i % DUMP == 0 { dump(lattice, &mut index)?; }
    }
    Ok(())
}

#[allow(dead_code)]
fn uniform_heating(lattice: &mut Vec<f64>) -> Result<(), Box<dyn std::error::Error>> {
    let h = 0.2;
    let q = 1.0;
    let mut index = 0;
    let mut orig = lattice.clone();
    for i in 0..STEP {
        onestep(lattice, &mut orig, h);
        for s in lattice.iter_mut() {
            *s += q * h;
        }
        lattice[0] = 0.0;
        lattice[L - 1] = 0.0;
        if i % DUMP == 0 { dump(lattice, &mut index)?; }
    }
    Ok(())
}

#[allow(unused_must_use)]
fn main() {
    let mut lattice = vec![0.0; L];
    uniform_heating(&mut lattice);
    //fixed_temperature(&mut lattice);
}