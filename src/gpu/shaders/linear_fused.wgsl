const TILE_M: u32 = 16u;
const TILE_N: u32 = 16u;
const TILE_K: u32 = 16u;

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

struct Params { m: u32, k: u32, n: u32 };
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> tile_x: array<f32, 256>;
var<workgroup> tile_w: array<f32, 256>;

@compute @workgroup_size(TILE_M, TILE_N)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let row = wg_id.y * TILE_M + local_id.y;
    let col = wg_id.x * TILE_N + local_id.x;
    
    var sum: f32 = 0.0;
    let num_k_tiles = (params.k + TILE_K - 1u) / TILE_K;
    
    for (var t: u32 = 0u; t < num_k_tiles; t = t + 1u) {
        let x_col = t * TILE_K + local_id.x;
        if (row < params.m && x_col < params.k) {
            tile_x[local_id.y * TILE_K + local_id.x] = x[row * params.k + x_col];
        } else {
            tile_x[local_id.y * TILE_K + local_id.x] = 0.0;
        }
        
        let w_row = t * TILE_K + local_id.y;
        if (w_row < params.k && col < params.n) {
            tile_w[local_id.y * TILE_N + local_id.x] = w[w_row * params.n + col];
        } else {
            tile_w[local_id.y * TILE_N + local_id.x] = 0.0;
        }
        
        workgroupBarrier();
        
        for (var i: u32 = 0u; i < TILE_K; i = i + 1u) {
            sum = sum + tile_x[local_id.y * TILE_K + i] * tile_w[i * TILE_N + local_id.x];
        }
        
        workgroupBarrier();
    }
    
    // FUSION MAGIC: Add bias and apply ReLU before writing to VRAM!
    if (row < params.m && col < params.n) {
        var val = sum + bias[col];
        if (val < 0.0) { val = 0.0; } // ReLU
        out[row * params.n + col] = val;
    }
}