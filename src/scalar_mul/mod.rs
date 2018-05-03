// -*- mode: rust; -*-
//
// This file is part of curve25519-dalek.
// Copyright (c) 2016-2018 Isis Lovecruft, Henry de Valence
// See LICENSE for licensing information.
//
// Authors:
// - Isis Agora Lovecruft <isis@patternsinthevoid.net>
// - Henry de Valence <hdevalence@hdevalence.ca>

#![allow(non_snake_case)]

use core::borrow::Borrow;

use clear_on_drop::ClearOnDrop;

use curve_models::{AffineNielsPoint, ProjectiveNielsPoint};
use curve_models::{CompletedPoint, ProjectivePoint};
use edwards::EdwardsPoint;
use scalar::Scalar;
use scalar_mul::window::{LookupTable, NafLookupTable5, NafLookupTable8};
use traits::Identity;
use traits::MultiscalarMul;
use traits::VartimeMultiscalarMul;
use traits::PrecomputedMultiscalarMul;

pub mod window;

pub mod variable_base;

#[cfg(feature = "precomputed_tables")]
pub mod vartime_double_base;

/// Perform multiscalar multiplication by the interleaved window
/// method, also known as Straus' method (since it was apparently
/// [first published][solution] by Straus in 1964, as a solution to [a
/// problem][problem] posted in the American Mathematical Monthly in
/// 1963).
///
/// It is easy enough to reinvent, and has been repeatedly.  The basic
/// idea is that when computing
/// \\[
/// Q = s_1 P_1 + \cdots + s_n P_n
/// \\]
/// by means of additions and doublings, the doublings can be shared
/// across the \\( P_i \\\).
///
/// We implement two versions, a constant-time algorithm using fixed
/// windows and a variable-time algorithm using sliding windows.  They
/// are slight variations on the same idea, and are described in more
/// detail in the respective implementations.
///
/// [solution]: https://www.jstor.org/stable/2310929
/// [problem]: https://www.jstor.org/stable/2312273
pub struct Straus {}

#[cfg(any(feature = "alloc", feature = "std"))]
impl MultiscalarMul for Straus {
    type Point = EdwardsPoint;

    /// Constant-time Straus using a fixed window of size \\(4\\).
    ///
    /// Our goal is to compute
    /// \\[
    /// Q = s_1 P_1 + \cdots + s_n P_n.
    /// \\]
    ///
    /// For each point \\( P_i \\), precompute a lookup table of
    /// \\[
    /// P_i, 2P_i, 3P_i, 4P_i, 5P_i, 6P_i, 7P_i, 8P_i.
    /// \\]
    ///
    /// For each scalar \\( s_i \\), compute its radix-\\(2^4\\)
    /// signed digits \\( s_{i,j} \\), i.e.,
    /// \\[
    ///    s_i = s_{i,0} + s_{i,1} 16^1 + ... + s_{i,63} 16^{63},
    /// \\]
    /// with \\( -8 \leq s_{i,j} < 8 \\).  Since \\( 0 \leq |s_{i,j}|
    /// \leq 8 \\), we can retrieve \\( s_{i,j} P_i \\) from the
    /// lookup table with a conditional negation: using signed
    /// digits halves the required table size.
    ///
    /// Then as in the single-base fixed window case, we have
    /// \\[
    /// \begin{aligned}
    /// s_i P_i &= P_i (s_{i,0} +     s_{i,1} 16^1 + \cdots +     s_{i,63} 16^{63})   \\\\
    /// s_i P_i &= P_i s_{i,0} + P_i s_{i,1} 16^1 + \cdots + P_i s_{i,63} 16^{63}     \\\\
    /// s_i P_i &= P_i s_{i,0} + 16(P_i s_{i,1} + 16( \cdots +16P_i s_{i,63})\cdots )
    /// \end{aligned}
    /// \\]
    /// so each \\( s_i P_i \\) can be computed by alternately adding
    /// a precomputed multiple \\( P_i s_{i,j} \\) of \\( P_i \\) and
    /// repeatedly doubling.
    ///
    /// Now consider the two-dimensional sum
    /// \\[
    /// \begin{aligned}
    /// s\_1 P\_1 &=& P\_1 s\_{1,0} &+& 16 (P\_1 s\_{1,1} &+& 16 ( \cdots &+& 16 P\_1 s\_{1,63}&) \cdots ) \\\\
    ///     +     & &      +        & &      +            & &             & &     +            &           \\\\
    /// s\_2 P\_2 &=& P\_2 s\_{2,0} &+& 16 (P\_2 s\_{2,1} &+& 16 ( \cdots &+& 16 P\_2 s\_{2,63}&) \cdots ) \\\\
    ///     +     & &      +        & &      +            & &             & &     +            &           \\\\
    /// \vdots    & &  \vdots       & &   \vdots          & &             & &  \vdots          &           \\\\
    ///     +     & &      +        & &      +            & &             & &     +            &           \\\\
    /// s\_n P\_n &=& P\_n s\_{n,0} &+& 16 (P\_n s\_{n,1} &+& 16 ( \cdots &+& 16 P\_n s\_{n,63}&) \cdots )
    /// \end{aligned}
    /// \\]
    /// The sum of the left-hand column is the result \\( Q \\); by
    /// computing the two-dimensional sum on the right column-wise,
    /// top-to-bottom, then right-to-left, we need to multiply by \\(
    /// 16\\) only once per column, sharing the doublings across all
    /// of the input points.
    fn multiscalar_mul<I, J>(scalars: I, points: J) -> EdwardsPoint
    where
        I: IntoIterator,
        I::Item: Borrow<Scalar>,
        J: IntoIterator,
        J::Item: Borrow<EdwardsPoint>,
    {
        let lookup_tables: Vec<_> = points
            .into_iter()
            .map(|point| LookupTable::<ProjectiveNielsPoint>::from(point.borrow()))
            .collect();

        // This puts the scalar digits into a heap-allocated Vec.
        // To ensure that these are erased, pass ownership of the Vec into a
        // ClearOnDrop wrapper.
        let scalar_digits_vec: Vec<_> = scalars
            .into_iter()
            .map(|s| s.borrow().to_radix_16())
            .collect();
        let scalar_digits = ClearOnDrop::new(scalar_digits_vec);

        let mut Q = EdwardsPoint::identity();
        for j in (0..64).rev() {
            Q = Q.mul_by_pow_2(4);
            let it = scalar_digits.iter().zip(lookup_tables.iter());
            for (s_i, lookup_table_i) in it {
                // R_i = s_{i,j} * P_i
                let R_i = lookup_table_i.select(s_i[j]);
                // Q = Q + R_i
                Q = (&Q + &R_i).to_extended();
            }
        }
        Q
    }
}

#[cfg(any(feature = "alloc", feature = "std"))]
impl VartimeMultiscalarMul for Straus {
    type Point = EdwardsPoint;

    /// Variable-time Straus using a non-adjacent form of width \\(5\\).
    ///
    /// This is completely similar to the constant-time code, but we
    /// use a non-adjacent form for the scalar, and do not do table
    /// lookups in constant time.
    ///
    /// The non-adjacent form has signed, odd digits.  Using only odd
    /// digits halves the table size (since we only need odd
    /// multiples), or gives fewer additions for the same table size.
    fn vartime_multiscalar_mul<I, J>(scalars: I, points: J) -> EdwardsPoint
    where
        I: IntoIterator,
        I::Item: Borrow<Scalar>,
        J: IntoIterator,
        J::Item: Borrow<EdwardsPoint>,
    {
        let nafs: Vec<_> = scalars
            .into_iter()
            .map(|c| c.borrow().non_adjacent_form(5))
            .collect();
        let lookup_tables: Vec<_> = points
            .into_iter()
            .map(|P| NafLookupTable5::<ProjectiveNielsPoint>::from(P.borrow()))
            .collect();

        let mut r = ProjectivePoint::identity();

        for i in (0..255).rev() {
            let mut t: CompletedPoint = r.double();

            for (naf, lookup_table) in nafs.iter().zip(lookup_tables.iter()) {
                if naf[i] > 0 {
                    t = &t.to_extended() + &lookup_table.select(naf[i] as usize);
                } else if naf[i] < 0 {
                    t = &t.to_extended() - &lookup_table.select(-naf[i] as usize);
                }
            }

            r = t.to_projective();
        }

        r.to_extended()
    }
}

pub struct PrecomputedStraus {
    lookup_tables: Vec<LookupTable<AffineNielsPoint>>,
}

#[cfg(any(feature = "alloc", feature = "std"))]
impl PrecomputedMultiscalarMul for PrecomputedStraus {
    type Point = EdwardsPoint;

    fn new<I>(static_points: I) -> Self
    where
        I: IntoIterator,
        I::Item: Borrow<Self::Point>,
    {
        PrecomputedStraus {
            lookup_tables: static_points
                .into_iter()
                .map(|point| LookupTable::<AffineNielsPoint>::from(point.borrow()))
                .collect(),
        }
    }

    fn mixed_multiscalar_mul<I, J, K>(
        &self,
        static_scalars: I,
        dynamic_scalars: J,
        dynamic_points: K,
    ) -> Self::Point
    where
        I: IntoIterator,
        I::Item: Borrow<Scalar>,
        J: IntoIterator,
        J::Item: Borrow<Scalar>,
        K: IntoIterator,
        K::Item: Borrow<Self::Point>,
    {
        // Compute the scalar digits for the static and dynamic scalars.
        // To ensure that these are erased, pass ownership of the Vec into a
        // ClearOnDrop wrapper.

        let static_scalar_digits_vec: Vec<_> = static_scalars
            .into_iter()
            .map(|s| s.borrow().to_radix_16())
            .collect();
        let static_scalar_digits = ClearOnDrop::new(static_scalar_digits_vec);

        let dynamic_scalar_digits_vec: Vec<_> = dynamic_scalars
            .into_iter()
            .map(|s| s.borrow().to_radix_16())
            .collect();
        let dynamic_scalar_digits = ClearOnDrop::new(dynamic_scalar_digits_vec);

        let dynamic_lookup_tables: Vec<_> = dynamic_points
            .into_iter()
            .map(|point| LookupTable::<ProjectiveNielsPoint>::from(point.borrow()))
            .collect();

        let mut Q = EdwardsPoint::identity();
        for j in (0..64).rev() {
            Q = Q.mul_by_pow_2(4);

            // Add the static points
            let it = static_scalar_digits.iter().zip(self.lookup_tables.iter());
            for (s_i, lookup_table_i) in it {
                // R_i = s_{i,j} * P_i
                let R_i = lookup_table_i.select(s_i[j]);
                // Q = Q + R_i
                Q = (&Q + &R_i).to_extended();
            }

            // Add the dynamic points
            let it = dynamic_scalar_digits
                .iter()
                .zip(dynamic_lookup_tables.iter());
            for (s_i, lookup_table_i) in it {
                // R_i = s_{i,j} * P_i
                let R_i = lookup_table_i.select(s_i[j]);
                // Q = Q + R_i
                Q = (&Q + &R_i).to_extended();
            }
        }

        Q
    }
}
