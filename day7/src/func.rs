const N: usize = 10000;

pub fn func(a: &[f64; N], b: &[f64; N], c: &mut [f64; N]) {
    for i in 0..N {
        c[i] = a[i] + b[i];
    }
}
