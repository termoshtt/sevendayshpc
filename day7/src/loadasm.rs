#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub unsafe fn load(a: *const f64, index: isize) -> __m256d {
    _mm256_load_pd(a.offset(index))
}