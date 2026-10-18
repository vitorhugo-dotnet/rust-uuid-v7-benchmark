//! Reproducible, non-cryptographic UUIDv7-shaped workload shared with CUDA.
//! This is a benchmark generator, NOT a production UUID generator or a CSPRNG.

pub const DEFAULT_SEED: u64 = 42;

#[inline(always)]
pub fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// This function must remain bit-for-bit identical to cuda/uuid_gpu.cu.
#[inline(always)]
pub fn synthetic_uuid(seed: u64, index: u64, unix_ms: u64) -> u128 {
    let a = splitmix64(seed ^ index.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let b = splitmix64(seed ^ index.wrapping_mul(0xBF58_476D_1CE4_E5B9) ^ 0xD1B5_4A32_D192_ED03);
    let hi = ((unix_ms & 0xFFFF_FFFF_FFFF) << 16) | (7u64 << 12) | (a & 0xFFF);
    let lo = (2u64 << 62) | (b & 0x3FFF_FFFF_FFFF_FFFF);
    ((hi as u128) << 64) | lo as u128
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn known_vectors_shared_with_cuda() {
        let ms = 1_729_280_000_000;
        assert_eq!(synthetic_uuid(42, 0, ms), 0x0192a11ec0007e9588ddbbbeab9cba1b);
        assert_eq!(synthetic_uuid(42, 1, ms), 0x0192a11ec0007103a03d2985e40305bd);
        assert_eq!(synthetic_uuid(42, 2, ms), 0x0192a11ec00075e3982863846d2ec95c);
    }
    #[test]
    fn version_variant_and_timestamp() {
        for idx in [0, 1, 10, u32::MAX as u64] {
            let id = synthetic_uuid(42, idx, 1_729_280_000_000);
            assert_eq!((id >> 76) & 0xF, 7);
            assert_eq!((id >> 62) & 0x3, 2);
            assert_eq!(id >> 80, 1_729_280_000_000);
        }
    }
}