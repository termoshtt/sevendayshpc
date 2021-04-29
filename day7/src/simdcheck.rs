use rand::distributions::{Distribution, Uniform};

const N: usize = 10000;

#[repr(align(32))]
struct A {
    data: [f64; N],
}

unsafe fn check(vc: &[f64; N], ans: &[f64; N], t: &str) {
    let x = vc.iter().map(|&e| e.to_string()).collect::<Vec<String>>();
    let y = ans.iter().map(|&e| e.to_string()).collect::<Vec<String>>();
    let mut valid = true;
    for i in 0..N {
        let x_byte = x[i].as_bytes();
        let y_byte = y[i].as_bytes();
        let xb_len = x_byte.len();
        let yb_len = y_byte.len();
        if xb_len >= 8 && yb_len >= 8 {
            for j in 0..8 {
                if x_byte[j] != y_byte[j] {
                    valid = false;
                    break;
                }
            }
        } else {
            if xb_len != yb_len {
                valid = false;
            } else {
                for j in 0..xb_len {
                    if x_byte[j] != y_byte[j] {
                        valid = false;
                        break;
                    }
                }
            }
        }
        if !valid { break; }
    }
    if valid {
        println!("{} is OK", t);
    } else {
        println!("{} is NG", t);
    }
}

mod func;
use func::func;

mod funcsimd;
use funcsimd::func_simd;

fn main() {
    let ud = Uniform::new(0.0, 1.0);
    let mut rng = rand::thread_rng();
    let mut va = A { data: [std::f64::NAN; N] }.data;
    let mut vb = A { data: [std::f64::NAN; N] }.data;
    let _ = va.iter_mut()
                .map(|e| {
                    *e = ud.sample(&mut rng);
                })
                .collect::<()>();
    let _ = vb.iter_mut()
                .map(|e| {
                    *e = ud.sample(&mut rng);
                })
                .collect::<()>();
    let mut vc = A { data: [0.0; N] }.data;
    let mut ans = A { data: [std::f64::NAN; N] }.data;
    let _ = ans.iter_mut()
                .zip(va.iter().zip(vb.iter()))
                .map(|(ansi, (&ai, &bi))| {
                    *ansi = ai + bi;
                }).collect::<()>();
    unsafe {
        func(&va, &vb, &mut vc);
        check(&vc, &ans, "scalar");
        func_simd(&va, &vb, &mut vc);
        check(&vc, &ans, "vector");
    }
}