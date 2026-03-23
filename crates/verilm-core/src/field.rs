//! Prime field F_p with p = 2^32 - 5.
//!
//! All Freivalds checks operate in this field. INT8 values are lifted
//! into F_p, dot products are accumulated in u128 to avoid overflow.

pub const P: u64 = 4_294_967_291; // 2^32 - 5

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Fp(pub u32);

impl Fp {
    pub const ZERO: Fp = Fp(0);
    pub const ONE: Fp = Fp(1);

    pub fn new(val: u32) -> Self {
        Fp(val % P as u32)
    }

    /// Lift a signed INT8 value into F_p.
    /// Maps -128..127 to F_p by taking val mod p (always positive).
    pub fn from_i8(val: i8) -> Self {
        let v = val as i64;
        let reduced = v.rem_euclid(P as i64) as u32;
        Fp(reduced)
    }

    /// Lift a signed i32 value into F_p.
    pub fn from_i32(val: i32) -> Self {
        let v = val as i64;
        let reduced = v.rem_euclid(P as i64) as u32;
        Fp(reduced)
    }

    pub fn add(self, other: Self) -> Self {
        let sum = self.0 as u64 + other.0 as u64;
        Fp((sum % P) as u32)
    }

    pub fn sub(self, other: Self) -> Self {
        // Add P to avoid underflow
        let diff = self.0 as u64 + P - other.0 as u64;
        Fp((diff % P) as u32)
    }

    pub fn mul(self, other: Self) -> Self {
        let prod = self.0 as u64 * other.0 as u64;
        Fp((prod % P) as u32)
    }

    /// Dot product of two slices in F_p. Accumulates in u128.
    /// Both slices must have the same length.
    pub fn dot(a: &[Fp], b: &[Fp]) -> Fp {
        assert_eq!(a.len(), b.len(), "dot product dimension mismatch");
        let mut acc: u128 = 0;
        for (x, y) in a.iter().zip(b.iter()) {
            acc += x.0 as u128 * y.0 as u128;
        }
        Fp((acc % P as u128) as u32)
    }

    /// Dot product of an Fp slice with an i8 slice.
    /// Lifts i8 values on the fly to avoid allocating a temporary Fp vec.
    pub fn dot_fp_i8(a: &[Fp], b: &[i8]) -> Fp {
        assert_eq!(a.len(), b.len(), "dot product dimension mismatch");
        // We split into positive and negative accumulators to stay in u128.
        // Each a[i] < P < 2^32, and |b[i]| <= 128.
        // Product < 2^32 * 128 = 2^39. Summing up to 2^15 = 32768 terms
        // gives < 2^54, well within u128.
        let mut pos_acc: u128 = 0;
        let mut neg_acc: u128 = 0;
        for (x, y) in a.iter().zip(b.iter()) {
            let yi = *y as i16;
            if yi >= 0 {
                pos_acc += x.0 as u128 * yi as u128;
            } else {
                neg_acc += x.0 as u128 * (-yi) as u128;
            }
        }
        // Reduce: result = pos_acc - neg_acc mod P
        let pos_reduced = (pos_acc % P as u128) as u64;
        let neg_reduced = (neg_acc % P as u128) as u64;
        if pos_reduced >= neg_reduced {
            Fp((pos_reduced - neg_reduced) as u32)
        } else {
            Fp((pos_reduced + P - neg_reduced) as u32)
        }
    }

    /// Dot product of an Fp slice with an i32 slice.
    /// Used for Freivalds checks against i32 accumulators.
    pub fn dot_fp_i32(a: &[Fp], b: &[i32]) -> Fp {
        assert_eq!(a.len(), b.len(), "dot product dimension mismatch");
        let mut pos_acc: u128 = 0;
        let mut neg_acc: u128 = 0;
        for (x, y) in a.iter().zip(b.iter()) {
            let yi = *y as i64;
            if yi >= 0 {
                pos_acc += x.0 as u128 * yi as u128;
            } else {
                neg_acc += x.0 as u128 * (-yi) as u128;
            }
        }
        let pos_reduced = (pos_acc % P as u128) as u64;
        let neg_reduced = (neg_acc % P as u128) as u64;
        if pos_reduced >= neg_reduced {
            Fp((pos_reduced - neg_reduced) as u32)
        } else {
            Fp((pos_reduced + P - neg_reduced) as u32)
        }
    }
}

impl std::fmt::Display for Fp {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// Fp64: prime field with p = 2^61 - 1 (Mersenne prime)
// ---------------------------------------------------------------------------

pub const P64: u64 = (1u64 << 61) - 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Fp64(pub u64);

impl Fp64 {
    pub const ZERO: Fp64 = Fp64(0);
    pub const ONE: Fp64 = Fp64(1);

    /// Mersenne reduction for 2^61 - 1. Input must be < 2^62 * p roughly,
    /// but works for any u64 by splitting high/low.
    #[inline]
    pub(crate) fn reduce(val: u128) -> u64 {
        let lo = (val & (P64 as u128)) as u64;
        let hi = (val >> 61) as u64;
        let sum = lo + hi;
        // One more reduction if sum >= P64
        let lo2 = sum & P64;
        let hi2 = sum >> 61;
        let r = lo2 + hi2;
        if r >= P64 { r - P64 } else { r }
    }

    pub fn new(val: u64) -> Self {
        Fp64(Self::reduce(val as u128))
    }

    pub fn from_i8(val: i8) -> Self {
        let v = val as i64;
        if v >= 0 {
            Fp64(v as u64)
        } else {
            Fp64(P64.wrapping_add(v as u64)) // P64 + v where v is negative
        }
    }

    pub fn from_i32(val: i32) -> Self {
        let v = val as i64;
        if v >= 0 {
            Fp64::new(v as u64)
        } else {
            // P64 + v; since |v| < 2^31 and P64 >> 2^31, no underflow
            Fp64(P64.wrapping_add(v as u64))
        }
    }

    pub fn add(self, other: Self) -> Self {
        let sum = self.0 + other.0;
        if sum >= P64 {
            Fp64(sum - P64)
        } else {
            Fp64(sum)
        }
    }

    pub fn sub(self, other: Self) -> Self {
        if self.0 >= other.0 {
            Fp64(self.0 - other.0)
        } else {
            Fp64(self.0 + P64 - other.0)
        }
    }

    pub fn mul(self, other: Self) -> Self {
        let prod = self.0 as u128 * other.0 as u128;
        Fp64(Self::reduce(prod))
    }

    pub fn dot(a: &[Fp64], b: &[Fp64]) -> Fp64 {
        assert_eq!(a.len(), b.len(), "dot product dimension mismatch");
        // Each product < P64^2 < 2^122. Summing many could overflow u128.
        // We reduce periodically. Batch size: 2^6 = 64 terms keeps acc < 2^128.
        let mut total: u128 = 0;
        for chunk in a.chunks(64).zip(b.chunks(64)).map(|(ca, cb)| {
            let mut acc: u128 = 0;
            for (x, y) in ca.iter().zip(cb.iter()) {
                acc += x.0 as u128 * y.0 as u128;
            }
            acc
        }) {
            total = Self::reduce(total) as u128 + Self::reduce(chunk) as u128;
        }
        Fp64(Self::reduce(total))
    }

    pub fn dot_fp_i8(a: &[Fp64], b: &[i8]) -> Fp64 {
        assert_eq!(a.len(), b.len(), "dot product dimension mismatch");
        let mut pos_acc: u128 = 0;
        let mut neg_acc: u128 = 0;
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            let yi = *y as i16;
            if yi >= 0 {
                pos_acc += x.0 as u128 * yi as u128;
            } else {
                neg_acc += x.0 as u128 * (-yi) as u128;
            }
            // Reduce every 2^20 terms to avoid overflow (each term < 2^61 * 128 = 2^68)
            if (i & 0xFFFFF) == 0xFFFFF {
                pos_acc = Self::reduce(pos_acc) as u128;
                neg_acc = Self::reduce(neg_acc) as u128;
            }
        }
        let pos = Self::reduce(pos_acc);
        let neg = Self::reduce(neg_acc);
        if pos >= neg {
            Fp64(pos - neg)
        } else {
            Fp64(pos + P64 - neg)
        }
    }

    pub fn dot_fp_i32(a: &[Fp64], b: &[i32]) -> Fp64 {
        assert_eq!(a.len(), b.len(), "dot product dimension mismatch");
        let mut pos_acc: u128 = 0;
        let mut neg_acc: u128 = 0;
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            let yi = *y as i64;
            if yi >= 0 {
                pos_acc += x.0 as u128 * yi as u128;
            } else {
                neg_acc += x.0 as u128 * (-yi) as u128;
            }
            // Each term < 2^61 * 2^31 = 2^92. Reduce every 2^20 terms.
            if (i & 0xFFFFF) == 0xFFFFF {
                pos_acc = Self::reduce(pos_acc) as u128;
                neg_acc = Self::reduce(neg_acc) as u128;
            }
        }
        let pos = Self::reduce(pos_acc);
        let neg = Self::reduce(neg_acc);
        if pos >= neg {
            Fp64(pos - neg)
        } else {
            Fp64(pos + P64 - neg)
        }
    }
}

impl std::fmt::Display for Fp64 {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// Fp128: prime field with p = 2^127 - 1 (Mersenne prime)
// ---------------------------------------------------------------------------

pub const P128: u128 = (1u128 << 127) - 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Fp128(pub u128);

/// Helper: 256-bit unsigned integer as (hi, lo) pair of u128.
/// Represents the value hi * 2^128 + lo.
#[derive(Debug, Clone, Copy)]
struct U256 {
    hi: u128,
    lo: u128,
}

impl U256 {
    #[inline]
    fn zero() -> Self {
        U256 { hi: 0, lo: 0 }
    }

    /// Multiply two u128 values, producing a 256-bit result.
    #[inline]
    fn mul128(a: u128, b: u128) -> Self {
        let a_lo = a as u64 as u128;
        let a_hi = a >> 64;
        let b_lo = b as u64 as u128;
        let b_hi = b >> 64;

        let ll = a_lo * b_lo;
        let lh = a_lo * b_hi;
        let hl = a_hi * b_lo;
        let hh = a_hi * b_hi;

        // Combine: result = hh * 2^128 + (lh + hl) * 2^64 + ll
        let (mid, carry1) = lh.overflowing_add(hl);
        let lo = ll.wrapping_add(mid << 64);
        let carry2 = if lo < ll { 1u128 } else { 0 };
        let hi = hh + (mid >> 64) + (if carry1 { 1u128 << 64 } else { 0 }) + carry2;

        U256 { hi, lo }
    }

    /// Add a u128 to this U256.
    #[inline]
    #[allow(dead_code)]
    fn add_u128(&mut self, val: u128) {
        let (new_lo, carry) = self.lo.overflowing_add(val);
        self.lo = new_lo;
        if carry {
            self.hi += 1;
        }
    }

    /// Add another U256 to this U256.
    #[inline]
    fn add_u256(&mut self, other: U256) {
        let (new_lo, carry) = self.lo.overflowing_add(other.lo);
        self.lo = new_lo;
        self.hi += other.hi;
        if carry {
            self.hi += 1;
        }
    }

    /// Reduce mod 2^127 - 1.
    /// x mod (2^127 - 1) = (x >> 127) + (x & (2^127 - 1)), possibly with one more reduction.
    fn reduce_mersenne127(self) -> u128 {
        // self represents hi * 2^128 + lo
        // We need to compute self mod (2^127 - 1).
        // First express as a single large number, then shift by 127.
        //
        // Bit 127 and above: hi * 2^128 means hi * 2^1 * 2^127.
        // So self = (hi * 2 + lo >> 127) * 2^127 + (lo & (2^127-1))
        // And x mod (2^127-1) = (x >> 127) + (x & mask127)
        //
        // upper = hi * 2 + (lo >> 127)
        // lower = lo & P128
        let mask = P128;
        let lower = self.lo & mask;
        let upper_from_lo = self.lo >> 127;
        // hi * 2^128 = hi * 2 * 2^127, so contributes hi * 2 to the upper part
        let upper = self.hi * 2 + upper_from_lo;

        // Now we need to reduce `upper` itself, since it could be large.
        // upper * 2^127 + lower, but we already folded once.
        // upper could be up to ~2^130 or so. We need to fold again.
        // upper mod (2^127-1): split upper the same way.
        let upper_lo = upper & mask;
        let upper_hi = upper >> 127;
        let folded_upper = upper_lo + upper_hi;

        let mut r = lower + folded_upper;
        // At most one more fold
        if r >= (1u128 << 127) {
            r = (r & mask) + (r >> 127);
        }
        if r >= mask {
            r -= mask;
        }
        r
    }
}

impl Fp128 {
    pub const ZERO: Fp128 = Fp128(0);
    pub const ONE: Fp128 = Fp128(1);

    #[inline]
    fn reduce(val: u128) -> u128 {
        let lo = val & P128;
        let hi = val >> 127;
        let r = lo + hi;
        if r >= P128 { r - P128 } else { r }
    }

    pub fn new(val: u128) -> Self {
        Fp128(Self::reduce(val))
    }

    pub fn from_i8(val: i8) -> Self {
        let v = val as i64;
        if v >= 0 {
            Fp128(v as u128)
        } else {
            Fp128(P128.wrapping_add(v as i128 as u128))
        }
    }

    pub fn from_i32(val: i32) -> Self {
        let v = val as i64;
        if v >= 0 {
            Fp128(v as u128)
        } else {
            Fp128(P128.wrapping_add(v as i128 as u128))
        }
    }

    pub fn add(self, other: Self) -> Self {
        let sum = self.0 + other.0;
        if sum >= P128 {
            Fp128(sum - P128)
        } else {
            Fp128(sum)
        }
    }

    pub fn sub(self, other: Self) -> Self {
        if self.0 >= other.0 {
            Fp128(self.0 - other.0)
        } else {
            Fp128(self.0 + P128 - other.0)
        }
    }

    pub fn mul(self, other: Self) -> Self {
        let prod = U256::mul128(self.0, other.0);
        Fp128(prod.reduce_mersenne127())
    }

    pub fn dot(a: &[Fp128], b: &[Fp128]) -> Fp128 {
        assert_eq!(a.len(), b.len(), "dot product dimension mismatch");
        // Accumulate in U256, reduce at the end.
        // Each product is a U256. We accumulate sums.
        // To avoid overflow in the U256 accumulator (hi could overflow u128),
        // we reduce every 2^64 terms. In practice vectors are much smaller.
        let mut acc = U256::zero();
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            let prod = U256::mul128(x.0, y.0);
            acc.add_u256(prod);
            if (i & 0xFFFF) == 0xFFFF {
                let reduced = acc.reduce_mersenne127();
                acc = U256 { hi: 0, lo: reduced };
            }
        }
        Fp128(acc.reduce_mersenne127())
    }

    pub fn dot_fp_i8(a: &[Fp128], b: &[i8]) -> Fp128 {
        assert_eq!(a.len(), b.len(), "dot product dimension mismatch");
        // Products are at most 127 + 7 = 134 bits, but since |b[i]| <= 128,
        // product fits in u128 * 128 which needs U256. Actually a.0 < 2^127
        // and |b| <= 128 < 2^8, so product < 2^135. Use U256 accumulators.
        let mut pos_acc = U256::zero();
        let mut neg_acc = U256::zero();
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            let yi = *y as i16;
            if yi >= 0 {
                let prod = U256::mul128(x.0, yi as u128);
                pos_acc.add_u256(prod);
            } else {
                let prod = U256::mul128(x.0, (-yi) as u128);
                neg_acc.add_u256(prod);
            }
            if (i & 0xFFFF) == 0xFFFF {
                let pr = pos_acc.reduce_mersenne127();
                let nr = neg_acc.reduce_mersenne127();
                pos_acc = U256 { hi: 0, lo: pr };
                neg_acc = U256 { hi: 0, lo: nr };
            }
        }
        let pos = pos_acc.reduce_mersenne127();
        let neg = neg_acc.reduce_mersenne127();
        if pos >= neg {
            Fp128(pos - neg)
        } else {
            Fp128(pos + P128 - neg)
        }
    }

    pub fn dot_fp_i32(a: &[Fp128], b: &[i32]) -> Fp128 {
        assert_eq!(a.len(), b.len(), "dot product dimension mismatch");
        let mut pos_acc = U256::zero();
        let mut neg_acc = U256::zero();
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            let yi = *y as i64;
            if yi >= 0 {
                // product < 2^127 * 2^31 = 2^158, needs U256
                let prod = U256::mul128(x.0, yi as u128);
                pos_acc.add_u256(prod);
            } else {
                let prod = U256::mul128(x.0, (-yi) as u128);
                neg_acc.add_u256(prod);
            }
            if (i & 0xFFFF) == 0xFFFF {
                let pr = pos_acc.reduce_mersenne127();
                let nr = neg_acc.reduce_mersenne127();
                pos_acc = U256 { hi: 0, lo: pr };
                neg_acc = U256 { hi: 0, lo: nr };
            }
        }
        let pos = pos_acc.reduce_mersenne127();
        let neg = neg_acc.reduce_mersenne127();
        if pos >= neg {
            Fp128(pos - neg)
        } else {
            Fp128(pos + P128 - neg)
        }
    }
}

impl std::fmt::Display for Fp128 {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_reduces() {
        let p = P as u32;
        assert_eq!(Fp::new(p), Fp(0));
        assert_eq!(Fp::new(p + 1), Fp(1));
        assert_eq!(Fp::new(0), Fp(0));
    }

    #[test]
    fn test_from_i8() {
        assert_eq!(Fp::from_i8(0), Fp(0));
        assert_eq!(Fp::from_i8(1), Fp(1));
        assert_eq!(Fp::from_i8(127), Fp(127));
        // -1 mod p = p - 1
        assert_eq!(Fp::from_i8(-1), Fp(P as u32 - 1));
        // -128 mod p = p - 128
        assert_eq!(Fp::from_i8(-128), Fp(P as u32 - 128));
    }

    #[test]
    fn test_add() {
        assert_eq!(Fp(0).add(Fp(0)), Fp(0));
        assert_eq!(Fp(1).add(Fp(2)), Fp(3));
        // Wrapping
        let p1 = Fp(P as u32 - 1);
        assert_eq!(p1.add(Fp(1)), Fp(0));
        assert_eq!(p1.add(Fp(2)), Fp(1));
    }

    #[test]
    fn test_sub() {
        assert_eq!(Fp(5).sub(Fp(3)), Fp(2));
        assert_eq!(Fp(0).sub(Fp(1)), Fp(P as u32 - 1));
        assert_eq!(Fp(3).sub(Fp(3)), Fp(0));
    }

    #[test]
    fn test_mul() {
        assert_eq!(Fp(0).mul(Fp(100)), Fp(0));
        assert_eq!(Fp(1).mul(Fp(42)), Fp(42));
        assert_eq!(Fp(2).mul(Fp(3)), Fp(6));
        // Large values
        let a = Fp(P as u32 - 1); // -1
        let b = Fp(P as u32 - 1); // -1
        assert_eq!(a.mul(b), Fp(1)); // (-1)(-1) = 1
    }

    #[test]
    fn test_distributive() {
        let a = Fp(12345);
        let b = Fp(67890);
        let c = Fp(11111);
        // a * (b + c) == a*b + a*c
        let lhs = a.mul(b.add(c));
        let rhs = a.mul(b).add(a.mul(c));
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn test_dot() {
        let a = vec![Fp(1), Fp(2), Fp(3)];
        let b = vec![Fp(4), Fp(5), Fp(6)];
        // 1*4 + 2*5 + 3*6 = 32
        assert_eq!(Fp::dot(&a, &b), Fp(32));
    }

    #[test]
    fn test_dot_fp_i8() {
        let a = vec![Fp(1), Fp(2), Fp(3)];
        let b = vec![4i8, 5, 6];
        assert_eq!(Fp::dot_fp_i8(&a, &b), Fp(32));

        // With negatives: 1*(-1) + 2*3 + 3*(-2) = -1 + 6 - 6 = -1 = P-1
        let a2 = vec![Fp(1), Fp(2), Fp(3)];
        let b2 = vec![-1i8, 3, -2];
        assert_eq!(Fp::dot_fp_i8(&a2, &b2), Fp(P as u32 - 1));
    }

    #[test]
    fn test_dot_fp_i8_matches_dot() {
        // dot_fp_i8 should produce the same result as lifting then dot
        let r = vec![Fp(100), Fp(200), Fp(300), Fp(400)];
        let x = vec![-50i8, 30, -128, 127];
        let x_fp: Vec<Fp> = x.iter().map(|&v| Fp::from_i8(v)).collect();
        assert_eq!(Fp::dot_fp_i8(&r, &x), Fp::dot(&r, &x_fp));
    }

    // -----------------------------------------------------------------------
    // Fp64 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_fp64_new_reduces() {
        assert_eq!(Fp64::new(0), Fp64(0));
        assert_eq!(Fp64::new(P64), Fp64(0));
        assert_eq!(Fp64::new(P64 + 1), Fp64(1));
        assert_eq!(Fp64::new(42), Fp64(42));
    }

    #[test]
    fn test_fp64_from_i8() {
        assert_eq!(Fp64::from_i8(0), Fp64(0));
        assert_eq!(Fp64::from_i8(1), Fp64(1));
        assert_eq!(Fp64::from_i8(127), Fp64(127));
        assert_eq!(Fp64::from_i8(-1), Fp64(P64 - 1));
        assert_eq!(Fp64::from_i8(-128), Fp64(P64 - 128));
    }

    #[test]
    fn test_fp64_from_i32() {
        assert_eq!(Fp64::from_i32(0), Fp64(0));
        assert_eq!(Fp64::from_i32(1000), Fp64(1000));
        assert_eq!(Fp64::from_i32(-1), Fp64(P64 - 1));
        assert_eq!(Fp64::from_i32(i32::MIN), Fp64(P64 - 2147483648));
    }

    #[test]
    fn test_fp64_add() {
        assert_eq!(Fp64(0).add(Fp64(0)), Fp64(0));
        assert_eq!(Fp64(1).add(Fp64(2)), Fp64(3));
        // Wrapping
        assert_eq!(Fp64(P64 - 1).add(Fp64(1)), Fp64(0));
        assert_eq!(Fp64(P64 - 1).add(Fp64(2)), Fp64(1));
    }

    #[test]
    fn test_fp64_sub() {
        assert_eq!(Fp64(5).sub(Fp64(3)), Fp64(2));
        assert_eq!(Fp64(0).sub(Fp64(1)), Fp64(P64 - 1));
        assert_eq!(Fp64(3).sub(Fp64(3)), Fp64(0));
    }

    #[test]
    fn test_fp64_mul() {
        assert_eq!(Fp64(0).mul(Fp64(100)), Fp64(0));
        assert_eq!(Fp64(1).mul(Fp64(42)), Fp64(42));
        assert_eq!(Fp64(2).mul(Fp64(3)), Fp64(6));
        // (-1) * (-1) = 1
        let a = Fp64(P64 - 1);
        let b = Fp64(P64 - 1);
        assert_eq!(a.mul(b), Fp64(1));
    }

    #[test]
    fn test_fp64_distributive() {
        let a = Fp64(12345);
        let b = Fp64(67890);
        let c = Fp64(11111);
        let lhs = a.mul(b.add(c));
        let rhs = a.mul(b).add(a.mul(c));
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn test_fp64_dot() {
        let a = vec![Fp64(1), Fp64(2), Fp64(3)];
        let b = vec![Fp64(4), Fp64(5), Fp64(6)];
        assert_eq!(Fp64::dot(&a, &b), Fp64(32));
    }

    #[test]
    fn test_fp64_dot_fp_i8() {
        let a = vec![Fp64(1), Fp64(2), Fp64(3)];
        let b = vec![4i8, 5, 6];
        assert_eq!(Fp64::dot_fp_i8(&a, &b), Fp64(32));

        let a2 = vec![Fp64(1), Fp64(2), Fp64(3)];
        let b2 = vec![-1i8, 3, -2];
        assert_eq!(Fp64::dot_fp_i8(&a2, &b2), Fp64(P64 - 1));
    }

    #[test]
    fn test_fp64_dot_fp_i8_matches_dot() {
        let r = vec![Fp64(100), Fp64(200), Fp64(300), Fp64(400)];
        let x = vec![-50i8, 30, -128, 127];
        let x_fp: Vec<Fp64> = x.iter().map(|&v| Fp64::from_i8(v)).collect();
        assert_eq!(Fp64::dot_fp_i8(&r, &x), Fp64::dot(&r, &x_fp));
    }

    #[test]
    fn test_fp64_dot_fp_i32() {
        let a = vec![Fp64(1), Fp64(2), Fp64(3)];
        let b = vec![4i32, -5, 6];
        // 1*4 + 2*(-5) + 3*6 = 4 - 10 + 18 = 12
        assert_eq!(Fp64::dot_fp_i32(&a, &b), Fp64(12));
    }

    // -----------------------------------------------------------------------
    // Fp128 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_fp128_new_reduces() {
        assert_eq!(Fp128::new(0), Fp128(0));
        assert_eq!(Fp128::new(P128), Fp128(0));
        assert_eq!(Fp128::new(P128 + 1), Fp128(1));
        assert_eq!(Fp128::new(42), Fp128(42));
    }

    #[test]
    fn test_fp128_from_i8() {
        assert_eq!(Fp128::from_i8(0), Fp128(0));
        assert_eq!(Fp128::from_i8(1), Fp128(1));
        assert_eq!(Fp128::from_i8(127), Fp128(127));
        assert_eq!(Fp128::from_i8(-1), Fp128(P128 - 1));
        assert_eq!(Fp128::from_i8(-128), Fp128(P128 - 128));
    }

    #[test]
    fn test_fp128_from_i32() {
        assert_eq!(Fp128::from_i32(0), Fp128(0));
        assert_eq!(Fp128::from_i32(1000), Fp128(1000));
        assert_eq!(Fp128::from_i32(-1), Fp128(P128 - 1));
        assert_eq!(Fp128::from_i32(i32::MIN), Fp128(P128 - 2147483648));
    }

    #[test]
    fn test_fp128_add() {
        assert_eq!(Fp128(0).add(Fp128(0)), Fp128(0));
        assert_eq!(Fp128(1).add(Fp128(2)), Fp128(3));
        assert_eq!(Fp128(P128 - 1).add(Fp128(1)), Fp128(0));
        assert_eq!(Fp128(P128 - 1).add(Fp128(2)), Fp128(1));
    }

    #[test]
    fn test_fp128_sub() {
        assert_eq!(Fp128(5).sub(Fp128(3)), Fp128(2));
        assert_eq!(Fp128(0).sub(Fp128(1)), Fp128(P128 - 1));
        assert_eq!(Fp128(3).sub(Fp128(3)), Fp128(0));
    }

    #[test]
    fn test_fp128_mul() {
        assert_eq!(Fp128(0).mul(Fp128(100)), Fp128(0));
        assert_eq!(Fp128(1).mul(Fp128(42)), Fp128(42));
        assert_eq!(Fp128(2).mul(Fp128(3)), Fp128(6));
        // (-1) * (-1) = 1
        let a = Fp128(P128 - 1);
        let b = Fp128(P128 - 1);
        assert_eq!(a.mul(b), Fp128(1));
    }

    #[test]
    fn test_fp128_distributive() {
        let a = Fp128(12345);
        let b = Fp128(67890);
        let c = Fp128(11111);
        let lhs = a.mul(b.add(c));
        let rhs = a.mul(b).add(a.mul(c));
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn test_fp128_dot() {
        let a = vec![Fp128(1), Fp128(2), Fp128(3)];
        let b = vec![Fp128(4), Fp128(5), Fp128(6)];
        assert_eq!(Fp128::dot(&a, &b), Fp128(32));
    }

    #[test]
    fn test_fp128_dot_fp_i8() {
        let a = vec![Fp128(1), Fp128(2), Fp128(3)];
        let b = vec![4i8, 5, 6];
        assert_eq!(Fp128::dot_fp_i8(&a, &b), Fp128(32));

        let a2 = vec![Fp128(1), Fp128(2), Fp128(3)];
        let b2 = vec![-1i8, 3, -2];
        assert_eq!(Fp128::dot_fp_i8(&a2, &b2), Fp128(P128 - 1));
    }

    #[test]
    fn test_fp128_dot_fp_i8_matches_dot() {
        let r = vec![Fp128(100), Fp128(200), Fp128(300), Fp128(400)];
        let x = vec![-50i8, 30, -128, 127];
        let x_fp: Vec<Fp128> = x.iter().map(|&v| Fp128::from_i8(v)).collect();
        assert_eq!(Fp128::dot_fp_i8(&r, &x), Fp128::dot(&r, &x_fp));
    }

    #[test]
    fn test_fp128_dot_fp_i32() {
        let a = vec![Fp128(1), Fp128(2), Fp128(3)];
        let b = vec![4i32, -5, 6];
        assert_eq!(Fp128::dot_fp_i32(&a, &b), Fp128(12));
    }

    #[test]
    fn test_fp128_mul_large() {
        // Test with larger values to exercise U256 multiplication
        let a = Fp128((1u128 << 100) + 7);
        let _b = Fp128((1u128 << 100) + 13);
        // Just check it doesn't panic and satisfies (-a)*a = -(a*a)
        let neg_a = Fp128(P128 - a.0);
        let prod1 = neg_a.mul(a);
        let prod2 = a.mul(a);
        let neg_prod2 = Fp128(P128 - prod2.0);
        assert_eq!(prod1, neg_prod2);
    }

    #[test]
    fn test_u256_mul128_basic() {
        // Test the U256 multiply with known values
        let r = U256::mul128(1, 1);
        assert_eq!(r.lo, 1);
        assert_eq!(r.hi, 0);

        let r2 = U256::mul128(1u128 << 64, 1u128 << 64);
        // (2^64)^2 = 2^128, so hi=1, lo=0
        assert_eq!(r2.lo, 0);
        assert_eq!(r2.hi, 1);
    }
}
