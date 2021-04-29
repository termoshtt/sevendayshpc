#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

unsafe fn m256d_to_slice(x: __m256d) -> [f64; 4] {
    let mut s = [0.0; 4];
    _mm256_storeu_pd(s.as_mut_ptr(), x);
    s
}

unsafe fn print256d(x: __m256d) {
    let s = m256d_to_slice(x);
    println!("{:.6} {:.6} {:.6} {:.6}", s[3], s[2], s[1], s[0]);
}

fn main() {
    unsafe {
        let v1 = _mm256_set_pd(3.0, 2.0, 1.0, 0.0);
        let v2 = _mm256_set_pd(7.0, 6.0, 5.0, 4.0);
        let v3 = _mm256_mul_pd(v1, v2);
        print256d(v3);
    }
}