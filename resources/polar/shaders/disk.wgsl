struct VertexOutput {
    @builtin(position) pos: vec4f,
    @location(0)  frag_pos: vec4f // unscaled fragment position
}

struct DrawspaceScales {
    scale:      vec2<f32>,
    extent:     vec2<f32>,
    resolution: vec2<f32>,
    density:    f32
}

struct DiskInfo {
    center:    vec2<f32>,
    radius:    f32,
    divisions: u32,
    color:     u32
}

var<push_constant> info: DiskInfo;

@group(0) @binding(0)
var<uniform> dscales: DrawspaceScales;

const PI: f32 = 3.141592653589793238;

const GUARDING_SCALE: f32   = 6.4; // obtained via T&E
const SMOOTHSTEP_SCALE: f32 = 1.6; // obtained via T&E

@vertex
fn vs_main(
    @builtin(vertex_index)   VertexIndex  : u32,
    @builtin(instance_index) InstanceIndex: u32
) -> VertexOutput {
    let guarding: f32 = GUARDING_SCALE / dscales.density; // extra space to accomodate the smoothing falloff
    let angle:    f32 = (2.0 * PI) / f32(info.divisions); // the angle between two adjacent radial line segments
    // the distance from the center of the n-gon to a vertex is calculated via
    // divding effective disk radius (the gurading applied) by the cosine of the half of adjacent angle
    // this slightly enlarges the radius, so that the disk will be fully housed within the n-gon.
    // Otherwise the n-gon will cut through the area of the disk
    let dist: f32 = (info.radius + guarding) / cos(angle * 0.5);

    // as wgpu don't support traingle fans, we need to resort to triangle strips for creating the n-gon.
    // It is a bit tricky. It is done by calculating an effective angle,
    // This effective angle is the multiple of the adjacent angle and the pair number of the current vertex.
    // and it is further multiplied with a sign, where the sign is negative for even vertex.
    // The angle is zero for the first vertex (VertexIndex = 0), and pair counting starts at (VertexIndex = 1)
    // After that, just simply do polar to cartesian conversion and add with the n-gon/disk center.
    // Please note: for drawing disks, the division count must be even.
    var factor = 1.0;
    if VertexIndex % 2u == 0u {
        factor = -1.0;
    }
    factor *= ceil(f32(VertexIndex) * 0.5);
    let pos: vec2f = dist * vec2f(cos(angle * factor), sin(angle * factor)) + info.center;

    var vto: VertexOutput;
    vto.pos      = vec4f(pos * dscales.scale, 0.0, 1.0);
    vto.frag_pos = vec4f(pos, 0.0, 1.0);

    return vto;
}

fn color_u32_to_vec4f(value: u32) -> vec4f {
    let r = f32((value >> 24u) & 255u);
    let g = f32((value >> 16u) & 255u);
    let b = f32((value >>  8u) & 255u);
    let a = f32(value          & 255u);
    
    return vec4f(r, g, b, a) * (1.0 / 255.0);
}

@fragment
fn fs_main(@location(0) frag_pos: vec2f) -> @location(0) vec4f {
    let dist:  f32 = length(frag_pos - info.center);
    let smstp: f32 = SMOOTHSTEP_SCALE / dscales.density;

    // Simple fragment shader based filled-circle drawing code
    // With smoothstepping for anti-aliasing instead of using the less-than operator
    return color_u32_to_vec4f(info.color) * smoothstep(
        info.radius + smstp, info.radius, dist
    );
}