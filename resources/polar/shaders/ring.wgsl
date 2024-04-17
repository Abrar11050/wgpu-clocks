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

struct RingInfo {
    center:    vec2<f32>,
    radius:    f32,
    thickness: f32,
    angle:     f32,
    divisions: u32,
    color:     u32
}

var<push_constant> info: RingInfo;

@group(0) @binding(0)
var<uniform> dscales : DrawspaceScales;

const PI: f32 = 3.141592653589793238;

const GUARDING_SCALE: f32   = 6.4; // obtained via T&E
const SMOOTHSTEP_SCALE: f32 = 1.6; // obtained via T&E

@vertex
fn vs_main(
    @builtin(vertex_index)   VertexIndex  : u32,
    @builtin(instance_index) InstanceIndex: u32
) -> VertexOutput {
    // As this is a hollow n-gon, there will be two vertices lying on each radial line segments,
    // called inner and outer vertices, forming a pair (that starts from VertexIndex = 0)
    // As we are dealig with pairs now, drawing a ring requires twice as much vertices
    // than drawing a disk (+2 more to complete the ring)
    let guarding: f32  = GUARDING_SCALE / dscales.density; // extra space to accomodate the smoothing falloff
    let pair_no:  f32  = f32(i32(VertexIndex) / 2);
    let is_inner: bool = i32(VertexIndex) % 2 == 0;
    let angle:    f32  = (2.0 * PI) / f32(info.divisions); // the angle between two adjacent radial line segments
    var dist:     f32;

    // The effective radius is calculated by adding/subtracing (depending on outer/inner) the half thickness with gurading.
    // The calculation outer vertex's distance from center is same as calculating the one from disk drawing,
    // which is dividing the effective radius by the the cosine of the half of the adjacent angle.
    // Please refer to the disk shader, The inner vertex on the other hand, doesn't require such division.
    if(is_inner) {
        dist = info.radius - (info.thickness * 0.5) - guarding;
    } else {
        dist = (info.radius + (info.thickness * 0.5) + guarding) / cos(angle * 0.5);
    }

    // Apply polar to cartesian and add the n-gon's center
    let pos: vec2f = dist * vec2f(cos(angle * pair_no), sin(angle * pair_no)) + info.center;

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


fn get_linecap_rounding_factor(position: vec2f, center: vec2f, radius: f32) -> f32 {
    let dist:  f32 = length(position - center);
    let smstp: f32 = SMOOTHSTEP_SCALE / dscales.density;
    return smoothstep(radius + smstp, radius, dist);
}

@fragment
fn fs_main(@location(0) frag_pos: vec2f) -> @location(0) vec4f {
    // rotate the frag's coordinate 90 degrees
    // because a clock starts from the top, not from the right
    let angle = atan2(
         (frag_pos.x - info.center.x),
        -(frag_pos.y - info.center.y)
    ); // atan2(y, x) => [rotate(90deg)] => atan2(x, -y)

    let current_ng = PI - info.angle; // the arc's angle, inverted to convert to CW rotation

    if(current_ng < angle) {
        // check if we're inside the stroke of (unrounded) arc
        let dist:  f32 = length(frag_pos - info.center);
        var outer: f32 = info.radius + (info.thickness * 0.5);
        var inner: f32 = info.radius - (info.thickness * 0.5);
        let smstp: f32 = SMOOTHSTEP_SCALE / dscales.density;

        let infactor:  f32 = smoothstep(inner - smstp, inner, dist);
        let outfactor: f32 = smoothstep(outer + smstp, outer, dist);

        // when we multiply infactor and outfactor, the result will be 1.0
        // when we're full inside the stroke of the arc, outside that it is 0.0,
        // and it is [0.0~1.0] in the stroke boundaries
        return color_u32_to_vec4f(info.color) * infactor * outfactor;
    } else {
        // we're not inside the stroke of (unrounded) arc, but check whether we're inside
        // any of the rounded arc endings. There are two endings:
        // ending 0: stationary, centered at the beginning of the arc
        // ending 1: moveable, centered at the ending of the arc, dependant on the arc's angle
        let rounding_radius = info.thickness * 0.5;
        let center_0 = vec2f(0.0, info.radius) + info.center;
        let center_1 = info.radius * vec2f(cos(current_ng - PI * 0.5), sin(current_ng - PI * 0.5)) + info.center;

        let factor_0 = get_linecap_rounding_factor(frag_pos, center_0, rounding_radius);
        let factor_1 = get_linecap_rounding_factor(frag_pos, center_1, rounding_radius);

        return color_u32_to_vec4f(info.color) * min(factor_0 + factor_1, 1.0); // limit to 1, otherwise really bright overlaps
    }
}