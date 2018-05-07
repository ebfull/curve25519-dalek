// -*- mode: rust; -*-
//
// This file is part of curve25519-dalek.
// Copyright (c) 2016-2018 Isis Lovecruft, Henry de Valence
// See LICENSE for licensing information.
//
// Authors:
// - Isis Agora Lovecruft <isis@patternsinthevoid.net>
// - Henry de Valence <hdevalence@hdevalence.ca>

use core::borrow::Borrow;

use clear_on_drop::ClearOnDrop;

use backend::avx2::edwards::{CachedPoint, ExtendedPoint};
use edwards::EdwardsPoint;
use scalar::Scalar;
use scalar_mul::window::{LookupTable, NafLookupTable5, NafLookupTable8};
use traits::*;

pub mod variable_base;

#[cfg(feature = "precomputed_tables")]
pub mod vartime_double_base;

/// Multiscalar multiplication using interleaved window / Straus'
/// method.  See the `Straus` struct in the serial backend for more
/// details.
///
/// This exists as a seperate implementation from that one because the
/// AVX2 code uses different curve models (it does not pass between
/// multiple models during scalar mul), and it has to convert the
/// point representation on the fly.
pub struct Straus {}

#[cfg(any(feature = "alloc", feature = "std"))]
impl MultiscalarMul for Straus {
    type Point = EdwardsPoint;

    fn multiscalar_mul<I, J>(scalars: I, points: J) -> EdwardsPoint
    where
        I: IntoIterator,
        I::Item: Borrow<Scalar>,
        J: IntoIterator,
        J::Item: Borrow<EdwardsPoint>,
    {
        // Construct a lookup table of [P,2P,3P,4P,5P,6P,7P,8P]
        // for each input point P
        let lookup_tables: Vec<_> = points
            .into_iter()
            .map(|point| LookupTable::<CachedPoint>::from(point.borrow()))
            .collect();

        let scalar_digits_vec: Vec<_> = scalars
            .into_iter()
            .map(|s| s.borrow().to_radix_16())
            .collect();
        // Pass ownership to a ClearOnDrop wrapper
        let scalar_digits = ClearOnDrop::new(scalar_digits_vec);

        let mut Q = ExtendedPoint::identity();
        for j in (0..64).rev() {
            Q = Q.mul_by_pow_2(4);
            let it = scalar_digits.iter().zip(lookup_tables.iter());
            for (s_i, lookup_table_i) in it {
                // Q = Q + s_{i,j} * P_i
                Q = &Q + &lookup_table_i.select(s_i[j]);
            }
        }
        Q.into()
    }
}

#[cfg(any(feature = "alloc", feature = "std"))]
impl VartimeMultiscalarMul for Straus {
    type Point = EdwardsPoint;

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
            .map(|point| NafLookupTable5::<CachedPoint>::from(point.borrow()))
            .collect();

        let mut Q = ExtendedPoint::identity();

        for i in (0..255).rev() {
            Q = Q.double();

            for (naf, lookup_table) in nafs.iter().zip(lookup_tables.iter()) {
                if naf[i] > 0 {
                    Q = &Q + &lookup_table.select(naf[i] as usize);
                } else if naf[i] < 0 {
                    Q = &Q - &lookup_table.select(-naf[i] as usize);
                }
            }
        }
        Q.into()
    }
}

pub struct PrecomputedStraus {
    lookup_tables: Vec<LookupTable<CachedPoint>>,
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
                .map(|point| LookupTable::<CachedPoint>::from(point.borrow()))
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
            .map(|point| LookupTable::<CachedPoint>::from(point.borrow()))
            .collect();

        let mut Q = ExtendedPoint::identity();
        for j in (0..64).rev() {
            Q = Q.mul_by_pow_2(4);

            // Add the static points
            let it = static_scalar_digits.iter().zip(self.lookup_tables.iter());
            for (s_i, lookup_table_i) in it {
                // Q = Q + s_{i,j} * P_i
                Q = &Q + &lookup_table_i.select(s_i[j]);
            }

            // Add the dynamic points
            let it = dynamic_scalar_digits
                .iter()
                .zip(dynamic_lookup_tables.iter());
            for (s_i, lookup_table_i) in it {
                // Q = Q + s_{i,j} * P_i
                Q = &Q + &lookup_table_i.select(s_i[j]);
            }
        }

        Q.into()
    }
}

pub struct VartimePrecomputedStraus {
    lookup_tables: Vec<NafLookupTable8<CachedPoint>>,
}

#[cfg(any(feature = "alloc", feature = "std"))]
impl VartimePrecomputedMultiscalarMul for VartimePrecomputedStraus {
    type Point = EdwardsPoint;

    fn new<I>(static_points: I) -> Self
    where
        I: IntoIterator,
        I::Item: Borrow<Self::Point>,
    {
        VartimePrecomputedStraus {
            lookup_tables: static_points
                .into_iter()
                .map(|point| NafLookupTable8::<CachedPoint>::from(point.borrow()))
                .collect(),
        }
    }

    fn vartime_mixed_multiscalar_mul<I, J, K>(
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
        let static_scalar_nafs: Vec<_> = static_scalars
            .into_iter()
            .map(|c| c.borrow().non_adjacent_form(8))
            .collect();

        let dynamic_scalar_nafs: Vec<_> = dynamic_scalars
            .into_iter()
            .map(|c| c.borrow().non_adjacent_form(5))
            .collect();

        let dynamic_lookup_tables: Vec<_> = dynamic_points
            .into_iter()
            .map(|P| NafLookupTable5::<CachedPoint>::from(P.borrow()))
            .collect();

        let mut Q = ExtendedPoint::identity();

        for i in (0..255).rev() {
            Q = Q.double();

            // Static points use width-8 NAFs
            let it = static_scalar_nafs.iter().zip(self.lookup_tables.iter());
            for (naf, lookup_table) in it {
                if naf[i] > 0 {
                    Q = &Q + &lookup_table.select(naf[i] as usize);
                } else if naf[i] < 0 {
                    Q = &Q - &lookup_table.select(-naf[i] as usize);
                }
            }

            // Dynamic points use width-5 NAFs
            let it = dynamic_scalar_nafs.iter().zip(dynamic_lookup_tables.iter());
            for (naf, lookup_table) in it {
                if naf[i] > 0 {
                    Q = &Q + &lookup_table.select(naf[i] as usize);
                } else if naf[i] < 0 {
                    Q = &Q - &lookup_table.select(-naf[i] as usize);
                }
            }
        }

        Q.into()
    }
}
