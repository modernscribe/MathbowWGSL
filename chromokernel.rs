#![forbid(unsafe_code)]

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde_json::json;
use std::cmp::{max, min};
use std::env;
use std::io::{self, Read};

#[derive(Clone, Copy, Debug)]
struct Cell {
    t: i32,
    e: u8,
    s: i16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CrossMode {
    Reflect,
    Refract,
    Fold,
}

impl CrossMode {
    fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "reflect" => Some(Self::Reflect),
            "refract" => Some(Self::Refract),
            "fold" | "ophiocul" | "ophiocul_fold" => Some(Self::Fold),
            _ => None,
        }
    }
    fn as_str(&self) -> &'static str {
        match self {
            Self::Reflect => "Reflect",
            Self::Refract => "Refract",
            Self::Fold => "Fold",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EmitMode {
    Json,
    Raw,
    Both,
}

impl EmitMode {
    fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "json" => Some(Self::Json),
            "raw" => Some(Self::Raw),
            "both" => Some(Self::Both),
            _ => None,
        }
    }
    fn as_str(&self) -> &'static str {
        match self {
            Self::Json => "json",
            Self::Raw => "raw",
            Self::Both => "both",
        }
    }
}

fn parse_arg_u64(args: &[String], key: &str, default: u64) -> u64 {
    for i in 0..args.len() {
        if args[i] == key && i + 1 < args.len() {
            if let Ok(v) = args[i + 1].parse::<u64>() {
                return v;
            }
        }
    }
    default
}

fn parse_arg_usize(args: &[String], key: &str, default: usize) -> usize {
    for i in 0..args.len() {
        if args[i] == key && i + 1 < args.len() {
            if let Ok(v) = args[i + 1].parse::<usize>() {
                return v;
            }
        }
    }
    default
}

fn parse_arg_string(args: &[String], key: &str, default: &str) -> String {
    for i in 0..args.len() {
        if args[i] == key && i + 1 < args.len() {
            return args[i + 1].clone();
        }
    }
    default.to_string()
}

fn has_flag(args: &[String], key: &str) -> bool {
    args.iter().any(|s| s == key)
}

fn clamp_i32(v: i64, lo: i32, hi: i32) -> i32 {
    if v < lo as i64 {
        lo
    } else if v > hi as i64 {
        hi
    } else {
        v as i32
    }
}

fn clamp_i16(v: i32, lo: i16, hi: i16) -> i16 {
    if v < lo as i32 {
        lo
    } else if v > hi as i32 {
        hi
    } else {
        v as i16
    }
}

fn rotl8(x: u8, r: u8) -> u8 {
    let rr = (r & 7) as u32;
    x.rotate_left(rr)
}

fn mix_u64(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    x
}

fn stdin_seed_xor(seed: u64, stdin: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &b in stdin {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    mix_u64(seed ^ h ^ ((stdin.len() as u64) << 32))
}

fn init_arena(n: usize, mut rng: StdRng) -> Vec<Cell> {
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        let t = rng.gen_range(-1024i32..=1024i32);
        let e = rng.gen::<u8>();
        let s = rng.gen_range(-16i16..=16i16);
        v.push(Cell { t, e, s });
    }
    v
}

fn octave_bin_e(e: u8) -> usize {
    ((e as usize) * 12) / 256
}

fn octave_gate(cell: Cell, neighbor_mix: i32, octave_strength: i32) -> Cell {
    let bin = octave_bin_e(cell.e) as i32;
    let bias = (bin - 6) * octave_strength;
    let t2 = clamp_i32(
        (cell.t as i64) + (neighbor_mix as i64) + (bias as i64),
        -1_000_000,
        1_000_000,
    );
    let s2 = clamp_i16(
        (cell.s as i32) + (neighbor_mix / 64) - (bias / 256),
        -32768,
        32767,
    );
    let e2 = cell.e.wrapping_add(((neighbor_mix ^ bias) & 0xFF) as u8);
    Cell {
        t: t2,
        e: e2,
        s: s2,
    }
}

fn step_pass(
    mode: CrossMode,
    arena: &[Cell],
    out: &mut [Cell],
    step_idx: u64,
    pass_idx: u64,
    octave_strength: i32,
) {
    let n = arena.len();
    if n == 0 {
        return;
    }
    for i in 0..n {
        let l = arena[(i + n - 1) % n];
        let c = arena[i];
        let r = arena[(i + 1) % n];

        let mix0 = (l.t ^ (r.t.rotate_left(3))) as i64 + (l.s as i64 * 17) - (r.s as i64 * 13)
            + ((l.e as i64) << 1)
            - ((r.e as i64) << 2);

        let k =
            mix_u64(((step_idx + 1) << 32) ^ ((pass_idx + 1) << 16) ^ (i as u64) ^ (c.e as u64));
        let jitter = (k & 0x3FF) as i32 - 512;
        let neighbor_mix = clamp_i32(mix0 + jitter as i64, -2_000_000, 2_000_000);

        let base = octave_gate(c, neighbor_mix, octave_strength);

        let (t2, e2, s2) = match mode {
            CrossMode::Reflect => {
                let dt = (neighbor_mix / 8) as i64;
                let ds = (neighbor_mix / 128) as i32;
                let de = (neighbor_mix as u32 & 0xFF) as u8;
                (
                    clamp_i32(base.t as i64 + dt, -2_000_000, 2_000_000),
                    base.e ^ de,
                    clamp_i16(base.s as i32 + ds, -32768, 32767),
                )
            }
            CrossMode::Refract => {
                let dt = (neighbor_mix / 7) as i64;
                let ds = (neighbor_mix / 96) as i32;
                let rot = (neighbor_mix.unsigned_abs() & 7) as u8;
                (
                    clamp_i32(base.t as i64 - dt, -2_000_000, 2_000_000),
                    rotl8(base.e, rot).wrapping_add(neighbor_mix as u8),
                    clamp_i16(
                        base.s as i32 + ds + (octave_bin_e(base.e) as i32 - 6),
                        -32768,
                        32767,
                    ),
                )
            }
            CrossMode::Fold => {
                let edge_a = ((l.e as i32) - (c.e as i32)) + ((c.e as i32) - (r.e as i32));
                let edge_t = (l.t - c.t) + (c.t - r.t);
                let edge_s = (l.s as i32 - c.s as i32) + (c.s as i32 - r.s as i32);

                let fold = clamp_i32(
                    (edge_t as i64) / 4 + (edge_a as i64) * 3 - (edge_s as i64) * 2,
                    -2_000_000,
                    2_000_000,
                );

                let t = clamp_i32(
                    base.t as i64 + (fold as i64) + ((neighbor_mix / 32) as i64),
                    -2_000_000,
                    2_000_000,
                );
                let s = clamp_i16(base.s as i32 + (fold / 256) + (edge_s / 32), -32768, 32767);

                let e = {
                    let bin = octave_bin_e(base.e) as i32;
                    let lift = ((bin - 6) * 11 + (fold / 1024)) as i32;
                    base.e
                        .wrapping_add((fold & 0xFF) as u8)
                        .wrapping_add((lift & 0xFF) as u8)
                        ^ ((neighbor_mix & 0xFF) as u8)
                };

                (t, e, s)
            }
        };

        out[i] = Cell {
            t: t2,
            e: e2,
            s: s2,
        };
    }
}

#[derive(Clone, Debug)]
struct AxisStatsI32 {
    min: i32,
    max: i32,
    mean: f64,
    count: usize,
    zeros: usize,
    positives: usize,
    negatives: usize,
    hist_mod12: [u64; 12],
}

#[derive(Clone, Debug)]
struct AxisStatsU8 {
    min: u8,
    max: u8,
    mean: f64,
    count: usize,
    zeros: usize,
    positives: usize,
    negatives: usize,
    hist_mod12: [u64; 12],
}

#[derive(Clone, Debug)]
struct AxisStatsI16 {
    min: i16,
    max: i16,
    mean: f64,
    count: usize,
    zeros: usize,
    positives: usize,
    negatives: usize,
    hist_mod12: [u64; 12],
}

fn stats_t(arena: &[Cell]) -> AxisStatsI32 {
    let mut mn = i32::MAX;
    let mut mx = i32::MIN;
    let mut sum: f64 = 0.0;
    let mut zeros = 0usize;
    let mut pos = 0usize;
    let mut neg = 0usize;
    let mut hist = [0u64; 12];
    for c in arena {
        mn = min(mn, c.t);
        mx = max(mx, c.t);
        sum += c.t as f64;
        if c.t == 0 {
            zeros += 1;
        } else if c.t > 0 {
            pos += 1;
        } else {
            neg += 1;
        }
        let m = (c.t.rem_euclid(12)) as usize;
        hist[m] += 1;
    }
    AxisStatsI32 {
        min: mn,
        max: mx,
        mean: sum / (arena.len().max(1) as f64),
        count: arena.len(),
        zeros,
        positives: pos,
        negatives: neg,
        hist_mod12: hist,
    }
}

fn stats_e(arena: &[Cell]) -> AxisStatsU8 {
    let mut mn = u8::MAX;
    let mut mx = u8::MIN;
    let mut sum: f64 = 0.0;
    let mut zeros = 0usize;
    let mut pos = 0usize;
    let neg = 0usize;
    let mut hist = [0u64; 12];
    for c in arena {
        mn = min(mn, c.e);
        mx = max(mx, c.e);
        sum += c.e as f64;
        if c.e == 0 {
            zeros += 1;
        } else {
            pos += 1;
        }
        let m = (c.e as usize) % 12;
        hist[m] += 1;
    }
    AxisStatsU8 {
        min: mn,
        max: mx,
        mean: sum / (arena.len().max(1) as f64),
        count: arena.len(),
        zeros,
        positives: pos,
        negatives: neg,
        hist_mod12: hist,
    }
}

fn stats_s(arena: &[Cell]) -> AxisStatsI16 {
    let mut mn = i16::MAX;
    let mut mx = i16::MIN;
    let mut sum: f64 = 0.0;
    let mut zeros = 0usize;
    let mut pos = 0usize;
    let mut neg = 0usize;
    let mut hist = [0u64; 12];
    for c in arena {
        mn = min(mn, c.s);
        mx = max(mx, c.s);
        sum += c.s as f64;
        if c.s == 0 {
            zeros += 1;
        } else if c.s > 0 {
            pos += 1;
        } else {
            neg += 1;
        }
        let m = (i32::from(c.s).rem_euclid(12)) as usize;
        hist[m] += 1;
    }
    AxisStatsI16 {
        min: mn,
        max: mx,
        mean: sum / (arena.len().max(1) as f64),
        count: arena.len(),
        zeros,
        positives: pos,
        negatives: neg,
        hist_mod12: hist,
    }
}

fn parse_passes(spec: &str) -> Vec<CrossMode> {
    let mut out = Vec::new();
    for tok in spec.split(',').map(|x| x.trim()).filter(|x| !x.is_empty()) {
        if let Some(m) = CrossMode::parse(tok) {
            out.push(m);
        }
    }
    if out.is_empty() {
        out.push(CrossMode::Reflect);
    }
    out
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let steps = parse_arg_usize(&args, "--steps", 64);
    let arena_len = parse_arg_usize(&args, "--arena", 4096);
    let seed = parse_arg_u64(&args, "--seed", 1337);
    let passes_spec = parse_arg_string(&args, "--passes", "reflect");
    let octave_strength = parse_arg_usize(&args, "--octave", 16) as i32;

    let emit = {
        let v = parse_arg_string(&args, "--emit", "json");
        EmitMode::parse(&v).unwrap_or(EmitMode::Json)
    };

    let read_stdin = !has_flag(&args, "--no-stdin");
    let mut stdin_bytes = Vec::new();
    if read_stdin {
        let mut stdin = io::stdin();
        let _ = stdin.read_to_end(&mut stdin_bytes);
    }

    let final_seed = stdin_seed_xor(seed, &stdin_bytes);
    let rng = StdRng::seed_from_u64(final_seed);

    let passes = parse_passes(&passes_spec);
    let mut arena = init_arena(arena_len, rng);
    let mut scratch = vec![Cell { t: 0, e: 0, s: 0 }; arena.len()];

    for step_idx in 0..steps as u64 {
        for (pi, mode) in passes.iter().copied().enumerate() {
            step_pass(
                mode,
                &arena,
                &mut scratch,
                step_idx,
                pi as u64,
                octave_strength,
            );
            std::mem::swap(&mut arena, &mut scratch);
        }
    }

    if emit == EmitMode::Raw || emit == EmitMode::Both {
        let mut out = String::new();
        for c in &arena {
            out.push_str(&format!("{} {} {}\n", c.t, c.e as u32, c.s));
        }
        print!("{}", out);
        if emit == EmitMode::Raw {
            return;
        }
    }

    let sample_n = 8usize.min(arena.len());
    let mut sample = Vec::with_capacity(sample_n);
    for i in 0..sample_n {
        let c = arena[i];
        sample.push(json!({"t": c.t, "e": c.e, "s": c.s}));
    }

    let st = stats_t(&arena);
    let se = stats_e(&arena);
    let ss = stats_s(&arena);

    let out = json!({
        "arena_len": arena.len(),
        "steps": steps,
        "seed": seed,
        "seed_effective": final_seed,
        "passes": passes.iter().map(|m| m.as_str()).collect::<Vec<_>>(),
        "octave_strength": octave_strength,
        "stdin_len": stdin_bytes.len(),
        "emit": emit.as_str(),
        "sample": sample,
        "axes": {
            "t": {
                "min": st.min,
                "max": st.max,
                "mean": (st.mean * 1_000_000.0).round() / 1_000_000.0,
                "count": st.count,
                "zeros": st.zeros,
                "positives": st.positives,
                "negatives": st.negatives,
                "hist_mod12": st.hist_mod12,
            },
            "e": {
                "min": se.min,
                "max": se.max,
                "mean": (se.mean * 1_000_000.0).round() / 1_000_000.0,
                "count": se.count,
                "zeros": se.zeros,
                "positives": se.positives,
                "negatives": se.negatives,
                "hist_mod12": se.hist_mod12,
            },
            "s": {
                "min": ss.min,
                "max": ss.max,
                "mean": (ss.mean * 1_000_000.0).round() / 1_000_000.0,
                "count": ss.count,
                "zeros": ss.zeros,
                "positives": ss.positives,
                "negatives": ss.negatives,
                "hist_mod12": ss.hist_mod12,
            },
        }
    });

    println!("{}", out);
}
