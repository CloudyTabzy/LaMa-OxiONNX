use std::time::Instant;

fn bench_gemm(m: usize, k: usize, n: usize, iters: usize) -> f64 {
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32) * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 13) as f32) * 0.1).collect();
    let mut c = vec![0.0f32; m * n];

    // Warmup
    unsafe {
        matrixmultiply::sgemm(
            m, k, n, 1.0, a.as_ptr(), k as isize, 1, b.as_ptr(), n as isize, 1, 0.0,
            c.as_mut_ptr(), n as isize, 1,
        );
    }

    let start = Instant::now();
    for _ in 0..iters {
        unsafe {
            matrixmultiply::sgemm(
                m, k, n, 1.0, a.as_ptr(), k as isize, 1, b.as_ptr(), n as isize, 1, 0.0,
                c.as_mut_ptr(), n as isize, 1,
            );
        }
    }
    let elapsed = start.elapsed().as_secs_f64() / iters as f64;
    let flops = 2.0 * (m * k * n) as f64;
    flops / elapsed / 1e9
}

/// Mimic the worker's N-split parallel_sgemm: `threads` threads each run
/// matrixmultiply on a disjoint N-chunk.
fn bench_parallel(m: usize, k: usize, n: usize, threads: usize, iters: usize) -> f64 {
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32) * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 13) as f32) * 0.1).collect();
    let mut c = vec![0.0f32; m * n];
    let a_ptr = a.as_ptr() as usize;
    let b_ptr = b.as_ptr() as usize;
    let c_ptr = c.as_mut_ptr() as usize;

    let chunk = n.div_ceil(threads);
    let run = || {
        std::thread::scope(|s| {
            for t in 0..threads {
                let n0 = t * chunk;
                if n0 >= n {
                    break;
                }
                let bn = chunk.min(n - n0);
                s.spawn(move || unsafe {
                    matrixmultiply::sgemm(
                        m,
                        k,
                        bn,
                        1.0,
                        a_ptr as *const f32,
                        k as isize,
                        1,
                        (b_ptr as *const f32).add(n0),
                        n as isize,
                        1,
                        0.0,
                        (c_ptr as *mut f32).add(n0),
                        n as isize,
                        1,
                    );
                });
            }
        });
    };

    run(); // warmup
    let start = Instant::now();
    for _ in 0..iters {
        run();
    }
    let elapsed = start.elapsed().as_secs_f64() / iters as f64;
    let flops = 2.0 * (m * k * n) as f64;
    flops / elapsed / 1e9
}

fn main() {
    // Representative LaMa shapes
    let shapes = [
        ("model.34 (final conv)", 3, 3136, 506 * 506),
        ("model.11 conv (256x256)", 128, 1152, 254 * 254),
        ("model.1 FFC (512x512)", 64, 196, 506 * 506),
        ("big square", 256, 256, 256 * 256),
        ("matmul shape", 512, 512, 512),
        ("convg2l split by 12", 128, 3456, 4096 / 12),
        ("convg2l split by 6", 128, 3456, 4096 / 6),
        ("convg2l split by 4", 128, 3456, 4096 / 4),
        ("convg2l full", 128, 3456, 4096),
        ("convl2g split 12", 384, 1152, 4096 / 12),
        ("1x1 split 12", 384, 384, 4096 / 12),
        ("1x1 full", 384, 384, 4096),
        ("m-split row 8", 8, 3456, 4096),
        ("m-split row 11", 11, 3456, 4096),
    ];
    for (name, m, k, n) in shapes {
        let iters = if m * k * n > 100_000_000 { 3 } else { 10 };
        let gflops = bench_gemm(m, k, n, iters);
        println!("{:30} M={:5} K={:5} N={:7}  {:8.2} GFLOP/s", name, m, k, n, gflops);
    }

    println!();
    println!("=== Parallel N-split (mimics worker) ===");
    for (name, m, k, n) in [
        ("1x1 conv 192 from 384", 192, 384, 4096),
        ("1x1 conv 384 from 192", 384, 384, 4096),
        ("convg2l", 128, 3456, 4096),
        ("convl2g", 384, 1152, 4096),
    ] {
        for threads in [16usize] {
            let gflops = bench_parallel(m, k, n, threads, 5);
            let ms = 2.0 * (m * k * n) as f64 / (gflops * 1e6);
            println!(
                "{:26} M={:4} K={:5} N={:5}  threads={:2}  {:8.1} GFLOP/s  ({:.2} ms)",
                name, m, k, n, threads, gflops, ms
            );
        }
    }
}
