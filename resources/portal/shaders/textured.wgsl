struct VertexOutput {
    @builtin(position) pos: vec4f,
    @location(0)        uv: vec2f
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

@vertex
fn vs_main(
    @location(0) pos: vec3<f32>,
    @location(1)  uv: vec2<f32>
) -> VertexOutput {
    var vto: VertexOutput;

    var mult: vec4<f32>;
    if bool(state.rotated) {
        mult = vec4f(-1.0, -1.0, 1.0, 1.0);
    } else {
        mult = vec4f(1.0, 1.0, 1.0, 1.0);
    }

    vto.pos = transform.matrix * (vec4f(pos, 1.0) * mult);
    vto.uv  = uv;

    return vto;
}

@fragment
fn fs_main(vto: VertexOutput) -> @location(0) vec4f {
    return textureSample(tex_2d, tex_sampler, vto.uv);
}