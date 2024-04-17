struct VertexOutput {
    @builtin(position) pos: vec4f,
    @location(0)        uv: vec2f
}

struct DrawspaceScales {
    scale:      vec2<f32>,
    extent:     vec2<f32>,
    resolution: vec2<f32>,
    density:    f32
}

struct BlurWO {
    weight: f32,
    offset: f32
}

@group(0) @binding(0)
var src_tex_2d: texture_2d<f32>; // the image generated from previous render pass

@group(0) @binding(1)
var tex_sampler: sampler;

@group(1) @binding(0)
var<uniform> dscales: DrawspaceScales;

@group(2) @binding(0)
var<storage, read> blur_table: array<BlurWO>;

@group(2) @binding(1)
var<uniform> blur_table_size: u32;

@group(3) @binding(0)
var orig_tex_2d: texture_2d<f32>; // the original image, used for blending (in last pass)

var<push_constant> vertical: u32;

@vertex
fn vs_main(
    @builtin(vertex_index)   VertexIndex  : u32,
    @builtin(instance_index) InstanceIndex: u32
) -> VertexOutput {
    // Simple full screen quad
    var coords = array<vec2f, 4>(
        vec2f(-1.0,  1.0),
        vec2f( 1.0,  1.0),
        vec2f(-1.0, -1.0),
        vec2f( 1.0, -1.0)
    );

    var uvs = array<vec2f, 4>(
        vec2f(0.0, 0.0),
        vec2f(1.0, 0.0),
        vec2f(0.0, 1.0),
        vec2f(1.0, 1.0)
    );

    var vto: VertexOutput;
    vto.pos = vec4f(coords[VertexIndex], 0.0, 1.0);
    vto.uv  = uvs[VertexIndex];

    return vto;
}

fn blur(dir: vec2f, uv: vec2f) -> vec4f {
    var result: vec3f = vec3f(0.0);

    for(var i = 0u; i < blur_table_size; i++) {
        var offset: vec2f = dir * blur_table[i].offset / dscales.resolution;
        var weight = blur_table[i].weight;
        var color = textureSample(src_tex_2d, tex_sampler, uv + offset);

        result += color.rgb * color.a * weight; // mutiply by alpha to enforce glow contribution
    }

    return vec4f(result, 1.0);
}

const BLUR_TINT: vec4f = vec4f(0.2, 0.2, 0.2, 1.0);

@fragment
fn fs_main(vto: VertexOutput) -> @location(0) vec4f {
    // We are considering the vertical blurring pass to be the last pass
    // Hence, we also do extra the blending (addition) after computing the blur
    // In the other one (horizontal blur pass), we just compute the blur and pass it.
    if bool(vertical) {
        let blurring = blur(vec2f(0.0, 1.0), vto.uv);
        let original = textureSample(orig_tex_2d, tex_sampler, vto.uv);
        var combined = blurring * BLUR_TINT + original;
        combined.a = 1.0;
        return combined;
    } else {
        let blurring = blur(vec2f(1.0, 0.0), vto.uv);
        return blurring;
    }
}