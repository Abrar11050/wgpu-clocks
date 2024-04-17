/// The `powered_on` property is actually boolean but set as u32 because bool can't be interpolated
/// However, the value of it won't be lost during the interploation process of rasterization stage. 
struct VertexOutput {
    @builtin(position)  pos: vec4f,
    @location(0)   uv_coord: vec2f,
    @location(1) powered_on: u32
}

struct ClockData {
    flagset:   array<u32, 2>,
    selector:  u32,
    timestamp: f32
}

struct DrawspaceScales {
    scale:      vec2<f32>,
    extent:     vec2<f32>,
    resolution: vec2<f32>,
    density:    f32
}

var<push_constant> cdata: ClockData;

@group(0) @binding(0)
var tex_2d: texture_2d<f32>;

@group(0) @binding(1)
var tex_sampler: sampler;

@group(1) @binding(0)
var<uniform> dscales: DrawspaceScales;

@vertex
fn vs_main(
    @location(0) pos: vec2<f32>,
    @location(1)  id: u32
) -> VertexOutput {

    var vto: VertexOutput;

    vto.pos = vec4(
        pos * dscales.scale,
        0.0,
        1.0
    );

    vto.uv_coord = vec2f(
        ( pos.x / dscales.extent.x) * 0.5 + 0.5,
        (-pos.y / dscales.extent.y) * 0.5 + 0.5
    );

    var flags:  u32 = cdata.flagset[id / 32u]; // select the 1st or 2nd flagset based on island ID
    var island: u32 = id % 32u; // the local island ID relative to the selected flagset
    var is_on: bool = bool(flags & (1u << island)); // check if the corresponding island's bit is enabled or not

    if is_on {
        vto.powered_on = 1u;
    } else {
        vto.powered_on = 0u;
    }

    return vto;
}

fn hsv2rgb(c: vec3f) -> vec3f {
    let k: vec4f = vec4f(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    let p: vec3f = abs(fract(c.xxx + k.xyz) * 6.0 - k.www);
    return c.z * mix(k.xxx, clamp(p - k.xxx, vec3f(0.0), vec3f(1.0)), c.y);
}

// I can't remember who originally wrote this effect, all I know I collected it from shadertoy, but lost the link.
// I converted the GLSL to WGSL, with slight modifications.
// Anyone knowing the original source please inform.
fn waves(uv: vec2f) -> vec3f {
    var p = 2.0 * uv - vec2f(1.0);
    let time = cdata.timestamp;

    for(var i = 1; i < 45; i++) {
        let i_ = f32(i); let i_10 = i_ + 10.0;
        var new_p = p;
        new_p.x += (0.5 / i_) * cos(i_ * p.y + time * 11.0 / 37.0 + 0.03 *   i_) + 1.3;
        new_p.y += (0.5 / i_) * cos(i_ * p.x + time * 17.0 / 41.0 + 0.03 * i_10) + 1.9;
        p = new_p;
    }

    return vec3f(
        0.5 * sin(3.0 * p.x) + 0.5,
        0.5 * sin(3.0 * p.y) + 0.5,
        sin(1.3 * p.x + 1.7 * p.y)
    );
}

const FADE_DURATION: f32 = 5.0;

@fragment
fn fs_main(vto: VertexOutput) -> @location(0) vec4f {
    // Check if the current fragment is within an LED region (white color)
    // specified by the clock layout
    let within_field: bool = textureSample(tex_2d, tex_sampler, vto.uv_coord).x > 0.1;

    var color: vec3f;

    switch cdata.selector {
        case 0u: {
            color = vec3f(0.058, 0.321, 1.0); // blue
        }
        case 1u: {
            color = vec3f(0.1, 0.9, 0.1); // green
        }
        case 2u: {
            color = vec3f(1.0, 0.3, 0.0); // orange
        }
        case 3u: {
            let hue = f32(cdata.timestamp % FADE_DURATION) / FADE_DURATION;
            color = hsv2rgb(vec3f(hue, 1.0, 1.0)); // rgb fading
        }
        case 4u: {
            color = waves(vto.uv_coord) + vec3f(0.1); // waves
        }
        default: {
            color = vec3f(0.85);
        }
    }

    // Here, the alpha channel plays the role of telling the
    // subsequent glow blurring pass that whether this pixel
    // contributes to the glow (1.0) or not (0.0)
    // pixels outside the LED region don't contribute,
    // same for turned off LED regions
    if within_field {
        if bool(vto.powered_on) {
            return vec4f(color, 1.0); // on state, bright color
        } else {
            return vec4f(0.005, 0.005, 0.005, 0.0); // off state, dark color but not fully black
        }
    } else {
        return vec4f(0.0);
    }

    return vec4f(waves(vto.uv_coord), 1.0);
}