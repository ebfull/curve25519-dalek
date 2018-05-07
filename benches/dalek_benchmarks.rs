#![allow(non_snake_case)]

extern crate rand;
use rand::OsRng;

#[macro_use]
extern crate criterion;

use criterion::Criterion;

extern crate curve25519_dalek;

use curve25519_dalek::constants;
use curve25519_dalek::scalar::Scalar;

static BATCH_SIZES: [usize; 5] = [1, 2, 4, 8, 16];
static MULTISCALAR_SIZES: [usize; 13] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 768, 1024];

mod edwards_benches {
    use super::*;
    use curve25519_dalek::edwards;
    use curve25519_dalek::edwards::EdwardsPoint;
    use curve25519_dalek::traits::MultiscalarMul;
    use curve25519_dalek::traits::VartimeMultiscalarMul;

    fn compress(c: &mut Criterion) {
        let B = &constants::ED25519_BASEPOINT_POINT;
        c.bench_function("EdwardsPoint compression", move |b| {
            b.iter(|| B.compress())
        });
    }

    fn decompress(c: &mut Criterion) {
        let B_comp = &constants::ED25519_BASEPOINT_COMPRESSED;
        c.bench_function("EdwardsPoint decompression", move |b| {
            b.iter(|| B_comp.decompress().unwrap())
        });
    }

    fn consttime_fixed_base_scalar_mul(c: &mut Criterion) {
        let B = &constants::ED25519_BASEPOINT_TABLE;
        let s = Scalar::from_u64(897987897).invert();
        c.bench_function("Constant-time fixed-base scalar mul", move |b| {
            b.iter(|| B * &s)
        });
    }

    fn consttime_variable_base_scalar_mul(c: &mut Criterion) {
        let B = &constants::ED25519_BASEPOINT_POINT;
        let s = Scalar::from_u64(897987897).invert();
        c.bench_function("Constant-time variable-base scalar mul", move |b| {
            b.iter(|| B * &s)
        });
    }

    fn vartime_double_base_scalar_mul(c: &mut Criterion) {
        c.bench_function("Variable-time aA+bB, A variable, B fixed", |bench| {
            let B = &constants::ED25519_BASEPOINT_POINT;
            let a = Scalar::from_u64(298374928).invert();
            let b = Scalar::from_u64(897987897).invert();
            let A = B * (b * a);
            bench.iter(|| edwards::vartime::double_scalar_mul_basepoint(&a, &A, &b));
        });
    }

    criterion_group!{
        name = edwards_benches;
        config = Criterion::default();
        targets =
        compress,
        decompress,
        consttime_fixed_base_scalar_mul,
        consttime_variable_base_scalar_mul,
        vartime_double_base_scalar_mul,
    }
}

mod multiscalar_benches {
    use super::*;
    use curve25519_dalek::edwards;
    use curve25519_dalek::edwards::EdwardsPoint;
    use curve25519_dalek::traits::MultiscalarMul;
    use curve25519_dalek::traits::VartimeMultiscalarMul;

    fn consttime_multiscalar_mul(c: &mut Criterion) {
        c.bench_function_over_inputs(
            "Constant-time variable-base multiscalar multiplication",
            |b, &&size| {
                let mut rng = OsRng::new().unwrap();
                let scalars: Vec<Scalar> = (0..size).map(|_| Scalar::random(&mut rng)).collect();
                let points: Vec<EdwardsPoint> = scalars
                    .iter()
                    .map(|s| s * &constants::ED25519_BASEPOINT_TABLE)
                    .collect();
                b.iter(|| EdwardsPoint::multiscalar_mul(&scalars, &points));
            },
            &MULTISCALAR_SIZES,
        );
    }

    fn vartime_multiscalar_mul(c: &mut Criterion) {
        c.bench_function_over_inputs(
            "Variable-time variable-base multiscalar multiplication",
            |b, &&size| {
                let mut rng = OsRng::new().unwrap();
                let scalars: Vec<Scalar> = (0..size).map(|_| Scalar::random(&mut rng)).collect();
                let points: Vec<EdwardsPoint> = scalars
                    .iter()
                    .map(|s| s * &constants::ED25519_BASEPOINT_TABLE)
                    .collect();
                b.iter(|| EdwardsPoint::vartime_multiscalar_mul(&scalars, &points));
            },
            &MULTISCALAR_SIZES,
        );
    }

    fn precomputed_consttime_multiscalar_mul_helper(c: &mut Criterion, dynamic_fraction: f64) {
        let label = format!(
            "Constant-time multiscalar mul with (1x,{:.2}x) static/dynamic ratio",
            dynamic_fraction
        );
        c.bench_function_over_inputs(
            &label,
            move |b, &&static_size| {
                let mut rng = OsRng::new().unwrap();

                let dynamic_size = ((static_size as f64) * dynamic_fraction) as usize;

                let static_scalars: Vec<Scalar> =
                    (0..static_size).map(|_| Scalar::random(&mut rng)).collect();

                let static_points: Vec<EdwardsPoint> = static_scalars
                    .iter()
                    .map(|s| s * &constants::ED25519_BASEPOINT_TABLE)
                    .collect();

                let dynamic_scalars: Vec<Scalar> = (0..dynamic_size)
                    .map(|_| Scalar::random(&mut rng))
                    .collect();

                let dynamic_points: Vec<EdwardsPoint> = dynamic_scalars
                    .iter()
                    .map(|s| s * &constants::ED25519_BASEPOINT_TABLE)
                    .collect();

                use curve25519_dalek::edwards::PrecomputedStraus;
                use curve25519_dalek::traits::PrecomputedMultiscalarMul;

                let precomp = PrecomputedStraus::new(&static_points);

                b.iter(|| {
                    precomp.mixed_multiscalar_mul(
                        &static_scalars,
                        &dynamic_scalars,
                        &dynamic_points,
                    )
                });
            },
            &MULTISCALAR_SIZES,
        );
    }

    fn precomputed_consttime_multiscalar_mul_0_0x(c: &mut Criterion) {
        precomputed_consttime_multiscalar_mul_helper(c, 0.0);
    }

    fn precomputed_consttime_multiscalar_mul_0_2x(c: &mut Criterion) {
        precomputed_consttime_multiscalar_mul_helper(c, 0.2);
    }

    fn precomputed_consttime_multiscalar_mul_1_0x(c: &mut Criterion) {
        precomputed_consttime_multiscalar_mul_helper(c, 1.0);
    }

    fn precomputed_vartime_multiscalar_mul_helper(c: &mut Criterion, dynamic_fraction: f64) {
        let label = format!(
            "Variable-time multiscalar mul with (1x,{:.2}x) static/dynamic ratio",
            dynamic_fraction
        );
        c.bench_function_over_inputs(
            &label,
            move |b, &&static_size| {
                let mut rng = OsRng::new().unwrap();

                let dynamic_size = ((static_size as f64) * dynamic_fraction) as usize;

                let static_scalars: Vec<Scalar> =
                    (0..static_size).map(|_| Scalar::random(&mut rng)).collect();

                let static_points: Vec<EdwardsPoint> = static_scalars
                    .iter()
                    .map(|s| s * &constants::ED25519_BASEPOINT_TABLE)
                    .collect();

                let dynamic_scalars: Vec<Scalar> = (0..dynamic_size)
                    .map(|_| Scalar::random(&mut rng))
                    .collect();

                let dynamic_points: Vec<EdwardsPoint> = dynamic_scalars
                    .iter()
                    .map(|s| s * &constants::ED25519_BASEPOINT_TABLE)
                    .collect();

                use curve25519_dalek::edwards::VartimePrecomputedStraus;
                use curve25519_dalek::traits::VartimePrecomputedMultiscalarMul;

                let precomp = VartimePrecomputedStraus::new(&static_points);

                b.iter(|| {
                    precomp.vartime_mixed_multiscalar_mul(
                        &static_scalars,
                        &dynamic_scalars,
                        &dynamic_points,
                    )
                });
            },
            &MULTISCALAR_SIZES,
        );
    }

    fn precomputed_vartime_multiscalar_mul_0_0x(c: &mut Criterion) {
        precomputed_vartime_multiscalar_mul_helper(c, 0.0);
    }

    fn precomputed_vartime_multiscalar_mul_0_2x(c: &mut Criterion) {
        precomputed_vartime_multiscalar_mul_helper(c, 0.2);
    }

    fn precomputed_vartime_multiscalar_mul_1_0x(c: &mut Criterion) {
        precomputed_vartime_multiscalar_mul_helper(c, 1.0);
    }

    criterion_group!{
        name = multiscalar_benches;
        // Lower the sample size to run the benchmarks faster
        config = Criterion::default().sample_size(15);
        targets =
        consttime_multiscalar_mul,
        vartime_multiscalar_mul,
        precomputed_consttime_multiscalar_mul_0_0x,
        precomputed_consttime_multiscalar_mul_0_2x,
        precomputed_consttime_multiscalar_mul_1_0x,
        precomputed_vartime_multiscalar_mul_0_0x,
        precomputed_vartime_multiscalar_mul_0_2x,
        precomputed_vartime_multiscalar_mul_1_0x,
    }
}

mod ristretto_benches {
    use super::*;
    use curve25519_dalek::ristretto::RistrettoPoint;

    fn compress(c: &mut Criterion) {
        c.bench_function("RistrettoPoint compression", |b| {
            let B = &constants::RISTRETTO_BASEPOINT_POINT;
            b.iter(|| B.compress())
        });
    }

    fn decompress(c: &mut Criterion) {
        c.bench_function("RistrettoPoint decompression", |b| {
            let B_comp = &constants::RISTRETTO_BASEPOINT_COMPRESSED;
            b.iter(|| B_comp.decompress().unwrap())
        });
    }

    fn double_and_compress_batch(c: &mut Criterion) {
        c.bench_function_over_inputs(
            "Batch Ristretto double-and-encode",
            |b, &&size| {
                let mut rng = OsRng::new().unwrap();
                let points: Vec<RistrettoPoint> = (0..size)
                    .map(|_| RistrettoPoint::random(&mut rng))
                    .collect();
                b.iter(|| RistrettoPoint::double_and_compress_batch(&points));
            },
            &BATCH_SIZES,
        );
    }

    criterion_group!{
        name = ristretto_benches;
        config = Criterion::default();
        targets =
        compress,
        decompress,
        double_and_compress_batch,
    }
}

mod montgomery_benches {
    use super::*;

    fn montgomery_ladder(c: &mut Criterion) {
        c.bench_function("Montgomery pseudomultiplication", |b| {
            let B = constants::X25519_BASEPOINT;
            let s = Scalar::from_u64(897987897).invert();
            b.iter(|| B * s);
        });
    }

    criterion_group!{
        name = montgomery_benches;
        config = Criterion::default();
        targets = montgomery_ladder,
    }
}

mod scalar_benches {
    use super::*;

    fn scalar_inversion(c: &mut Criterion) {
        c.bench_function("Scalar inversion", |b| {
            let s = Scalar::from_u64(897987897).invert();
            b.iter(|| s.invert());
        });
    }

    fn batch_scalar_inversion(c: &mut Criterion) {
        c.bench_function_over_inputs(
            "Batch scalar inversion",
            |b, &&size| {
                let mut rng = OsRng::new().unwrap();
                let scalars: Vec<Scalar> = (0..size).map(|_| Scalar::random(&mut rng)).collect();
                b.iter(|| {
                    let mut s = scalars.clone();
                    Scalar::batch_invert(&mut s);
                });
            },
            &BATCH_SIZES,
        );
    }

    criterion_group!{
        name = scalar_benches;
        config = Criterion::default();
        targets =
        scalar_inversion,
        batch_scalar_inversion,
    }
}

criterion_main!(
    scalar_benches::scalar_benches,
    montgomery_benches::montgomery_benches,
    ristretto_benches::ristretto_benches,
    edwards_benches::edwards_benches,
    multiscalar_benches::multiscalar_benches,
);
