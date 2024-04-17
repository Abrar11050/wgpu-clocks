struct VertexOutput {
    @builtin(position) pos: vec4f,
    @location(0)        uv: vec2f
}

struct MatrixData {
    matrix: mat4x4<f32>
}

struct StateData {
    rotated: u32,
    digits:  u32
}

@group(0) @binding(0)
var tex_2d: texture_2d<f32>;

@group(0) @binding(1)
var tex_sampler: sampler;

@group(1) @binding(0)
var<uniform> transform : MatrixData;

var<push_constant> state: StateData;

fn mirror_odd(val: f32, num: u32) -> f32 {
    if num % 2u == 0u {
        return val;
    } else {
        return 1.0 - val;
    }
}

@vertex
fn vs_main(
    @location(0) pos: vec3<f32>,
    @location(1)  uv: vec2<f32>,
    @builtin(instance_index) InstanceIndex: u32
) -> VertexOutput {
    var vto: VertexOutput;

    var mult: vec4<f32>;

    if bool(state.rotated) {
        if bool(InstanceIndex) {
            mult = vec4f(1.0, -1.0, 1.0, 1.0); // R(180), xflip
        } else {
            mult = vec4f(-1.0, -1.0, 1.0, 1.0); // R(180), xreg
        }
    } else {
        if bool(InstanceIndex) {
            mult = vec4f(-1.0, 1.0, 1.0, 1.0); // R(0), xflip
        } else {
            mult = vec4f(1.0, 1.0, 1.0, 1.0); // R(0), xreg
        }
    }

    var digits = array<u32, 2>((state.digits >> 16u) & 0xFFFFu, state.digits & 0xFFFFu);

    vto.pos = transform.matrix * (vec4f(pos, 1.0) * mult);
    vto.uv  = vec2f(
        mirror_odd(uv.x, InstanceIndex) * 0.1 + f32(digits[InstanceIndex % 2u]) * 0.1,
        uv.y
    );

    return vto;
}

@fragment
fn fs_main(vto: VertexOutput) -> @location(0) vec4f {
    var tint: vec4<f32>;

    if bool(state.rotated) {
        tint = vec4f(1.0, 1.0, 1.0, 1.0);
    } else {
        tint = vec4f(0.0, 0.0, 0.0, 1.0);
    }

    let color = textureSample(tex_2d, tex_sampler, vto.uv);

    return color * tint;
}