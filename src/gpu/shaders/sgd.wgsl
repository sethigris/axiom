@group(0) @binding(0) var<storage, read_write> weight: array<f32>;
@group(0) @binding(1) var<storage, read> grad: array<f32>;

// Uniform buffers in wgpu must be 16-byte aligned, so we pad the f32 to a vec4.
struct Params { lr: f32, _pad0: f32, _pad1: f32, _pad2: f32 };
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&weight)) {
        weight[idx] = weight[idx] - params.lr * grad[idx];
    }
}