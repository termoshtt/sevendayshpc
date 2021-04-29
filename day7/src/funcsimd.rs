#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

const N: usize = 10000;

pub unsafe fn func_simd(a: &[f64; N], b: &[f64; N], c: &mut [f64; N]) {
    let mut i = 0;
    while i < N {
        let va = _mm256_load_pd(&a[i] as *const f64);
        let vb = _mm256_load_pd(&b[i] as *const f64);
        let vc = _mm256_add_pd(va, vb);
        _mm256_store_pd(&mut c[i], vc);
        i += 4;
    }
}