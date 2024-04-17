struct VertexOutput {
    @builtin(position) pos: vec4f,
    @location(0)  frag_pos: vec4f,
    @location(1)  frag_uvc: vec2f
}

struct MatrixData {
    matrix: mat4x4<f32>
}

struct RotationAngles {
    angles: array<f32, 6>
}

@group(0) @binding(0)
var<uniform> transform : MatrixData;

var<push_constant> rotation: RotationAngles;

const PI: f32 = 3.141592653589793238;

const PAIR_WIDTH:  f32 = 5.0;
const WHEEL_WIDTH: f32 = 2.25;
const BASE_POS: vec2<f32> = vec2f(-6.125, 0.0);

fn rotate_x(ng: f32) -> mat4x4<f32> {
    let cst = cos(ng);
    let snt = sin(ng);
    return mat4x4(
        vec4f(1.0, 0.0,  0.0, 0.0),
        vec4f(0.0, cst, -snt, 0.0),
        vec4f(0.0, snt,  cst, 0.0),
        vec4f(0.0, 0.0,  0.0, 1.0)
    );
}

fn translate_z(trz: f32) -> mat4x4<f32> {
    return mat4x4(
        vec4f(1.0, 0.0, 0.0, 0.0),
        vec4f(0.0, 1.0, 0.0, 0.0),
        vec4f(0.0, 0.0, 1.0, 0.0),
        vec4f(0.0, 0.0, trz, 1.0)
    );
}

const WHEEL_RADIUS: f32 = 6.0;

@vertex
fn vs_main(
    @builtin(vertex_index)   VertexIndex  : u32,
    @builtin(instance_index) InstanceIndex: u32
) -> VertexOutput {
    let pair_no  = InstanceIndex / 20u; // wheel pair index, there are 3 pairs
    let wheel_no = (InstanceIndex % 20u) / 10u; // wheel index inside pair, either 0 or 1
    let digit_no = InstanceIndex % 10u; // digit index within wheel, goes from 0 to 9

    // The horizontal offset, mainly. Cause the vertical offset is same for all
    let effective_pos = BASE_POS + vec2f((f32(pair_no) * PAIR_WIDTH + f32(wheel_no) * WHEEL_WIDTH), 0.0);

    var vertices = array<vec2f, 4>(
        vec2f(-1.0,  1.5),
        vec2f( 1.0,  1.5),
        vec2f(-1.0, -1.5),
        vec2f( 1.0, -1.5)
    );

    var uvs = array<vec2f, 4>(
        vec2f(0.0, 0.0),
        vec2f(0.1, 0.0),
        vec2f(0.0, 1.0),
        vec2f(0.1, 1.0)
    );

    var trn = translate_z(-WHEEL_RADIUS);
    // Each digit's "card" in a wheel is placed `2π/10` = `0.2π` radians apart, and are placed proportionally to their own value.
    // So, for a digit `n`, the card's angle will be `0.2πn`; this is prior to applying wheel rotation.
    // After applying parent wheel's rotation, the resultant card rotation will be `0.2πn + wheel_rotation`
    var rtn = rotate_x(((2.0 * PI) * (f32(digit_no) / 10.0)) + -rotation.angles[pair_no * 2u + wheel_no]);

    // The wheel's center is (offset.x, offset.y, WHEEL_RADIUS);
    var pos = vec4f(vertices[VertexIndex] + effective_pos, WHEEL_RADIUS, 1.0);

    // Sprite sheet adressing is applied to evalute the quad's UV coordinates
    var uvc = uvs[VertexIndex] + vec2f(0.1 * f32(digit_no), 0.0);

    // apply rotation and translation
    pos = (trn * rtn) * pos;

    var vto: VertexOutput;
    vto.pos      = transform.matrix * pos;
    vto.frag_pos = pos;
    vto.frag_uvc = uvc;

    return vto;
}

@group(0) @binding(1)
var tex_2d: texture_2d<f32>;

@group(0) @binding(2)
var tex_sampler: sampler;

@fragment
fn fs_main(vto: VertexOutput) -> @location(0) vec4f {
    // Arbitrary darkening effect where the cards above and below gets progressively darker
    // based on absolute value of the y-axis
    // Try removing the part after asterisk to disable this effect
    let t = min(abs(vto.frag_pos.y), WHEEL_RADIUS) / WHEEL_RADIUS;
    return textureSample(tex_2d, tex_sampler, vto.frag_uvc) * max(1.0 - 2.0 * pow(t, 3.0), 0.0);
}