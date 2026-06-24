// Tiled matrix multiplication using workgroup (shared) memory.
// Each workgroup computes a TILE_M x TILE_N block of the output matrix.
// Tiles of A and B are loaded into fast shared memory to avoid repeated VRAM reads.

const TILE_M: u32 = 16u;
const TILE_N: u32 = 16u;
const TILE_K: u32 = 16u;

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

// Push constants for dynamic dimensions (avoids recompilation for different sizes)
struct Params {
    m: u32,
    k: u32,
    n: u32,
};
@group(0) @binding(3) var<uniform> params: Params;

// Workgroup-shared tiles for A and B
var<workgroup> tile_a: array<f32, 256>; // TILE_M * TILE_K = 16*16
var<workgroup> tile_b: array<f32, 256>; // TILE_K * TILE_N = 16*16

@compute @workgroup_size(TILE_M, TILE_N)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let row = wg_id.y * TILE_M + local_id.y;
    let col = wg_id.x * TILE_N + local_id.x;
    
    var sum: f32 = 0.0;
    
    // Loop over K dimension in tiles
    let num_k_tiles = (params.k + TILE_K - 1u) / TILE_K;
    
    for (var t: u32 = 0u; t < num_k_tiles; t = t + 1u) {
        // Cooperatively load tile of A into shared memory
        let a_col = t * TILE_K + local_id.x;
        if (row < params.m && a_col < params.k) {
            tile_a[local_id.y * TILE_K + local_id.x] = a[row * params.k + a_col];
        } else {
            tile_a[local_id.y * TILE_K + local_id.x] = 0.0;
        }
        
        // Cooperatively load tile of B into shared memory
        let b_row = t * TILE_K + local_id.y;
        if (b_row < params.k && col < params.n) {
            tile_b[local_id.y * TILE_N + local_id.x] = b[b_row * params.n + col];
        } else {
            tile_b[local_id.y * TILE_N + local_id.x] = 0.0;
        }
        
        // Synchronize: ensure all threads have finished loading before computing
        workgroupBarrier();
        
        // Compute partial dot product from shared memory tiles
        for (var i: u32 = 0u; i < TILE_K; i = i + 1u) {
            sum = sum + tile_a[local_id.y * TILE_K + i] * tile_b[i * TILE_N + local_id.x];
        }
        
        // Synchronize: ensure all threads finish computing before next tile load
        workgroupBarrier();
    }
    
    // Write result
    if (row < params.m && col < params.n) {
        out[row * params.n + col] = sum;
    }
}