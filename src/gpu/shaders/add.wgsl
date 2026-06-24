// Storage buffers hold our tensor data in VRAM.
// `read` means the shader can only read from it.
// `read_write` means the shader can both read and write.
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

// @compute marks this as a compute shader (not graphics).
// @workgroup_size(256) means we launch threads in blocks of 256.
// The GPU will dispatch as many workgroups as needed to cover all elements.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    // Guard against out-of-bounds threads (workgroups are always multiples of 256).
    if (idx < arrayLength(&out)) {
        out[idx] = a[idx] + b[idx];
    }
}