struct VertexOutput {
    @builtin(position) pos: vec4f,
    @location(0) screenpos: vec4f
}

struct MatrixData {
    matrix: mat4x4<f32>
}

struct StateData {
    rotated: u32,
    _unused: u32
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

fn calc_screen_pos(pos: vec4f) -> vec4f {
    var o = pos * 0.5;

    return vec4f(vec2f(o.x, -o.y) + vec2f(o.w), pos.zw);
}

@vertex
fn vs_main(
    @location(0) pos: vec3<f32>,
    @location(1)  uv: vec2<f32>
) -> VertexOutput {
    var vto: VertexOutput;

    let p = transform.matrix * vec4f(pos, 1.0);

    vto.pos = p;

    vto.screenpos = calc_screen_pos(p);

    return vto;
}

@fragment
fn fs_main(vto: VertexOutput) -> @location(0) vec4f {
    return textureSample(tex_2d, tex_sampler, vto.screenpos.xy / vto.screenpos.w);
}