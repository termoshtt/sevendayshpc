#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[repr(align(32))]
struct A {
    data: [f64; 8],
}

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
    let a = A {
        data: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    };

    unsafe {
        let p: *const f64 = &a.data[0];
        let v1 = _mm256_load_pd(p);
        let v2 = _mm256_load_pd(p.offset(4));
        let v3 = _mm256_add_pd(v1, v2);
        print256d(v3);
    }
}
