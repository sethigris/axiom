@group(0) @binding(0) var<storage, read_write> weight: array<f32>;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read_write> m: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;

struct Params {
    lr: f32, beta1: f32, beta2: f32, eps: f32,
    weight_decay: f32, bc1: f32, bc2: f32, _pad: f32
};
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&weight)) {
        let g = grad[idx];
        let w = weight[idx];

        // 1. Load state into registers
        var m_val = m[idx];
        var v_val = v[idx];

        // 2. Update biased first and second moment estimates
        m_val = params.beta1 * m_val + (1.0 - params.beta1) * g;
        v_val = params.beta2 * v_val + (1.0 - params.beta2) * g * g;

        // 3. Write state back to VRAM
        m[idx] = m_val;
        v[idx] = v_val;

        // 4. Compute bias-corrected estimates
        let m_hat = m_val / params.bc1;
        let v_hat = v_val / params.bc2;

        // 5. Compute Adam update + Decoupled Weight Decay
        var update = m_hat / (sqrt(v_hat) + params.eps);
        update = update + params.weight_decay * w; 

        // 6. Apply update to weight
        weight[idx] = w - params.lr * update;
    }
}