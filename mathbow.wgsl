struct Params {
  n: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> A: array<u64>;
@group(0) @binding(1) var<storage, read> B: array<u64>;
@group(0) @binding(2) var<storage, read_write> OUT: array<u64>;
@group(0) @binding(3) var<uniform> P: Params;

fn rn_tier(x: u64) -> u64 {
  return x / 7u;
}

fn rn_color(x: u64) -> u64 {
  return x % 7u;
}

fn rn_add_index(a: u64, b: u64) -> u64 {
  let t1 = rn_tier(a);
  let c1 = rn_color(a);
  let t2 = rn_tier(b);
  let c2 = rn_color(b);
  let c = c1 + c2;
  let carry = c / 7u;
  let cmod = c % 7u;
  let t = t1 + t2 + carry;
  return t * 7u + cmod;
}

fn rn_mul_index(a: u64, b: u64) -> u64 {
  let t1 = rn_tier(a);
  let c1 = rn_color(a);
  let t2 = rn_tier(b);
  let c2 = rn_color(b);
  let t_tt = 7u * t1 * t2;
  let t_tc = t1 * c2 + t2 * c1;
  let c_prod = c1 * c2;
  let t_cc = c_prod / 7u;
  let c_cc = c_prod % 7u;
  let t = t_tt + t_tc + t_cc;
  return t * 7u + c_cc;
}

fn gcd_u64(a_in: u64, b_in: u64) -> u64 {
  var a = a_in;
  var b = b_in;
  loop {
    if (b == 0u) {
      break;
    }
    let r = a % b;
    a = b;
    b = r;
  }
  return a;
}

@compute @workgroup_size(256)
fn native_add(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) {
    return;
  }
  OUT[i] = A[i] + B[i];
}

@compute @workgroup_size(256)
fn rainbow_add(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) {
    return;
  }
  OUT[i] = rn_add_index(A[i], B[i]);
}

@compute @workgroup_size(256)
fn native_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) {
    return;
  }
  OUT[i] = A[i] * B[i];
}

@compute @workgroup_size(256)
fn rainbow_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) {
    return;
  }
  OUT[i] = rn_mul_index(A[i], B[i]);
}

@compute @workgroup_size(256)
fn native_divmod(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) {
    return;
  }
  let a = A[i];
  let b = B[i];
  let q = a / b;
  let r = a % b;
  OUT[i] = q ^ r;
}

@compute @workgroup_size(256)
fn rainbow_divmod(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) {
    return;
  }
  let a = A[i];
  let b = B[i];
  let q = a / b;
  let r = a % b;
  OUT[i] = q ^ r;
}

@compute @workgroup_size(256)
fn native_gcd(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) {
    return;
  }
  OUT[i] = gcd_u64(A[i], B[i]);
}

@compute @workgroup_size(256)
fn rainbow_gcd(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.n) {
    return;
  }
  let g = gcd_u64(A[i], B[i]);
  OUT[i] = g;
}
