#![cfg_attr(
    all(
        target_os = "windows",
        not(feature = "console"),
    ),
    windows_subsystem = "windows"
)]
use std::{borrow::Cow, fs::read_to_string};
use wgpu::{RenderPipelineDescriptor, PushConstantRange};
use clockutils::{
    run, cast_struct_to_u8_slice, get_resource_folder_for,
    ExecDraw, SingleUniformBuffer, DrawspaceScales, RenderTexture, ResourceTexture, BasicFilteringSampler,
    SURFACE_FORMAT
};
use chrono::{Local, Timelike, DateTime, TimeDelta};

#[repr(C, align(8))]
struct MatrixData {
    matrix: glam::Mat4
}

#[repr(C, align(8))]
struct RotationAngles {
    angles: [f32; 6]
}

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Nanosecs. changeable but should not exceed 1s
const ANIM_DURATION: u32 = 500_000_000;

/// We calculate the beginning and ending angles for each wheel.
/// The angles are proportional to the digit itself.
/// 
/// Here, the ending digit set is the digit set of current time.
/// And the beginning digit set is of `ANIM_DURATION` from current time.
/// The animation/transition happens at first `ANIM_DURATION` of the current second.
/// During this time period, the resultant angle is
/// calculated from lerping the beginning and ending angles.
/// The rest of the time, the wheel stays at ending angle.
/// 
/// All wheels must rotate in one direction.
/// To prevent reverse rotation, for high to low digit transition like `9 -> 0`,
/// the ending digit is added with `10` to make the transition look like `9 -> 10`
fn calc_wheel_angles() -> [f32; 6] {
    fn extract_digits_from_time(time: &DateTime<Local>) -> [u8; 6] {
        let hours   = time.hour();
        let minutes = time.minute();
        let seconds = time.second();
    
        [
            (hours   / 10) as u8, (hours   % 10) as u8,
            (minutes / 10) as u8, (minutes % 10) as u8,
            (seconds / 10) as u8, (seconds % 10) as u8,
        ]
    }

    fn angle_for_digit(digit: u8) -> f32 {
        (digit as f32 * 0.1) * std::f32::consts::TAU
    }

    let mut angles: [f32; 6] = [0.0; 6];

    let now = Local::now();
    let now_digits = extract_digits_from_time(&now);

    let nanos = now.nanosecond();

    if nanos > ANIM_DURATION {
        for (i, digit) in now_digits.iter().enumerate() {
            angles[i] = angle_for_digit(*digit);
        }
        return angles;
    }

    let ago = now - TimeDelta::nanoseconds(ANIM_DURATION as i64);
    let ago_digits = extract_digits_from_time(&ago);

    let t = (nanos as f32) / (ANIM_DURATION as f32);
    // t = ease_out_bounce(t);
    // Or use your own favorite easing

    for i in 0..now_digits.len() {
        let digit_ago = ago_digits[i];
        let digit_now = if now_digits[i] < ago_digits[i] {
            now_digits[i] + 10
        } else {
            now_digits[i]
        };

        let angle_ago = angle_for_digit(digit_ago);
        let angle_now = angle_for_digit(digit_now);

        angles[i] = (1.0 - t) * angle_ago + t * angle_now; // lerp
    }

    angles
}

fn calc_matrix(resolution: glam::Vec2, extent: glam::Vec2) -> MatrixData {
    let scale = {
        let dscales = DrawspaceScales::new(resolution, extent);

        // Since we are working in 3D,
        // apply the drawspace scales as a scaling matrix
        // only for x, y axis
        glam::Mat4::from_scale(glam::Vec3 {
            x: dscales.scale.x,
            y: dscales.scale.y,
            z: 1.0
        })
    };

    let proj = glam::Mat4::perspective_rh(0.5_f32.atan() * 2.0, 1.0, 1.0, 100.0);

    let view = glam::Mat4::look_at_rh(
        glam::Vec3::new(0.0, 0.0, 8.0),
        glam::Vec3::ZERO,
        glam::Vec3::Y
    );

    MatrixData { matrix: scale * proj * view }
}

struct MechCounter {
    pipeline:       wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group:     wgpu::BindGroup,

    depth_view:     wgpu::TextureView
}

impl ExecDraw for MechCounter {
    fn setup(
        config:   &wgpu::SurfaceConfiguration,
        _adapter: &wgpu::Adapter,
        device:   &wgpu::Device,
        queue:    &wgpu::Queue
    ) -> Self {
        let resources = get_resource_folder_for("mcounter").unwrap();

        let umatrix = SingleUniformBuffer::new::<MatrixData>(device, wgpu::ShaderStages::VERTEX_FRAGMENT);

        // The digit fonts as a sprite sheet.
        // Each digit occupies 10% of the whole texture's width
        // So pretty easily addressable
        let sprites = ResourceTexture::new(
            resources.join("textures/haettenschweiler_digits.png").as_path().to_str().unwrap(),
            device,
            queue
        );

        let sampler = BasicFilteringSampler::new(device);

        // Not doing anything complicated like,
        // so only one bind group will suffice for all shader resources
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                SingleUniformBuffer::default_layout_entry(0, &umatrix),
                ResourceTexture::default_layout_entry(1),
                BasicFilteringSampler::default_layout_entry(2)
            ]
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   None,
            layout:  &bind_group_layout,
            entries: &[
                umatrix.get_entry(0),
                sprites.get_entry(1),
                sampler.get_entry(2)
            ]
        });

        // In the push constants, we shove in the angles for all six wheels
        // 6 x sizeof(f32) = 6 x 4 = 24
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:                None,
            bind_group_layouts:   &[ &bind_group_layout ],
            push_constant_ranges: &[
                PushConstantRange {
                    stages: wgpu::ShaderStages::VERTEX,
                    range:  0..24
                }
            ]
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
                read_to_string(resources.join("shaders/mcounter.wgsl")).unwrap().as_str()
            ))
        });

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label:         None,
            layout:        Some(&pipeline_layout),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default()
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview:     None,
            vertex: wgpu::VertexState {
                module:      &shader,
                entry_point: "vs_main",
                buffers:     &[]
            },
            fragment: Some(wgpu::FragmentState {
                module:      &shader,
                entry_point: "fs_main",
                targets:     &[ Some(SURFACE_FORMAT.into()) ]
            }),
            primitive: wgpu::PrimitiveState {
                topology:     wgpu::PrimitiveTopology::TriangleStrip,
                cull_mode:    Some(wgpu::Face::Front),
                polygon_mode: wgpu::PolygonMode::Fill,
                ..Default::default()
            }
        });

        let depth_texture = RenderTexture::new(
            (config.width, config.height),
            DEPTH_FORMAT,
            false,
            device
        );

        Self {
            pipeline,
            uniform_buffer: umatrix.buffer,
            bind_group,
            depth_view: depth_texture.view
        }
    }

    fn resize(self: &mut Self, width: u32, height: u32, device: &wgpu::Device, queue: &wgpu::Queue) {
        let ubuffer = calc_matrix(
            glam::Vec2::new(width as f32, height as f32),
            glam::Vec2::new(2.0, 1.0)
        );

        // adapt the drawspace scales to the current resolution
        queue.write_buffer(&self.uniform_buffer, 0, cast_struct_to_u8_slice(&ubuffer));

        // the surface texture will be resized automatically
        // it's our duty to handle the depth buffer manually
        let depth_texture = RenderTexture::new(
            (width, height),
            DEPTH_FORMAT,
            false,
            device
        );

        self.depth_view = depth_texture.view;
    }

    fn draw(self: &mut Self, texview: &wgpu::TextureView, device: &wgpu::Device, queue: &wgpu::Queue) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let rtng = RotationAngles { angles: calc_wheel_angles() };

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label:                    None,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load:  wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store
                    }),
                    stencil_ops: None
                }),
                timestamp_writes:    None,
                occlusion_query_set: None,
                color_attachments:   &[Some(wgpu::RenderPassColorAttachment {
                    view: texview,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load:  wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }),
                        store: wgpu::StoreOp::Store
                    }
                })]
            });

            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, &self.bind_group, &[]);
            rpass.set_push_constants(
                wgpu::ShaderStages::VERTEX,
                0,
                cast_struct_to_u8_slice(&rtng)
            );

            // Issue a single draw call to draw everything via instancing.
            // Each wheel contains 10 digit "cards", and there are six wheels
            // So, 6 x 10 = 60 instances
            // Each card contains 4 vertices, the coordinates are calculated on-the-fly
            // via the vertex shader
            rpass.draw(0..4, 0..60);

        }

        queue.submit(std::iter::once(encoder.finish()));
    }
}

fn main() {
    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    #[allow(unused_mut)]
    let mut builder = winit::window::WindowBuilder::new();
    let window = builder
        .with_inner_size(winit::dpi::LogicalSize { width: 1024.0, height: 512.0 })
        .with_title("Mechanical Counter Clock")
        .build(&event_loop)
        .unwrap();

    pollster::block_on(run::<MechCounter>(
        event_loop, window,
        Some(wgpu::Features::PUSH_CONSTANTS))
    );
}

#[allow(dead_code)]
fn ease_out_bounce(mut x: f32) ->  f32 {
    let n1 = 7.5625;
    let d1 = 2.75;
    
    if x < 1.0 / d1 {
        return n1 * x * x;
    } else if x < 2.0 / d1 {
        x -= 1.5 / d1;
        return n1 * x * x + 0.75;
    } else if x < 2.5 / d1 {
        x -= 2.25 / d1;
        return n1 * x * x + 0.9375;
    } else {
        x -= 2.625 / d1;
        return n1 * x * x + 0.984375;
    }
}