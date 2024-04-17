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
    lerp_u32_color, u32_col_to_wgpu_col,
    ExecDraw, SingleUniformBuffer, DrawspaceScales,
    SURFACE_FORMAT
};
use chrono::{Local, Timelike};
use winit::platform::modifier_supplement::KeyEventExtModifierSupplement;

/// Properties of the "hollowed" n-gon on which the arc/ring will be drawn on.
/// Used for drawing an arc with angle control
#[repr(C, align(8))]
struct RingInfo {
    center:    glam::Vec2,
    radius:    f32,
    thickness: f32, 
    angle:     f32, // the angle of the arc on the ring in radians
    divisions: u32, // the "n" of the n-gon
    color:     u32
}

/// Properties of the n-gon on which the disk will be drawn on.
/// Used for drawing a filled circle
#[repr(C, align(8))]
struct DiskInfo {
    center:    glam::Vec2,
    radius:    f32,
    divisions: u32, // the "n" of the n-gon
    color:     u32
}

struct ColorCombo {
    hour:       u32, // color of hour ring
    minute:     u32, // color of minute ring
    second:     u32, // color of second ring
    disk:       u32, // common color of all disks
    background: u32  // background color
}

struct AnglesAndPositions {
    hours_angle:   f32,
    minutes_angle: f32,
    seconds_angle: f32,

    hours_pos:   (f32, f32),
    minutes_pos: (f32, f32),
    seconds_pos: (f32, f32)
}

fn calc_angles_and_positions() -> AnglesAndPositions {
    use std::f32::consts::{FRAC_PI_2, TAU, PI};

    let now = Local::now();
    let seconds = now.second() as f32 + (now.nanosecond() as f32 / 1_000_000_000.0);
    let minutes = now.minute() as f32 + seconds / 60.0;
    let hours   = (now.hour() % 12) as f32 + minutes / 60.0;

    // angles used for drawing the arcs and calculating disk centers
    let seconds_angle = (seconds / 60.0) * TAU;
    let minutes_angle = (minutes / 60.0) * TAU;
    let hours_angle   = (hours   / 12.0) * TAU;

    // positions of disk centers
    let seconds_pos: (f32, f32) = (
        SECONDS_RADIUS * ((PI + TAU - seconds_angle) - FRAC_PI_2).cos(),
        SECONDS_RADIUS * ((PI + TAU - seconds_angle) - FRAC_PI_2).sin()
    );

    let minutes_pos: (f32, f32) = (
        MINUTES_RADIUS * ((PI + TAU - minutes_angle) - FRAC_PI_2).cos(),
        MINUTES_RADIUS * ((PI + TAU - minutes_angle) - FRAC_PI_2).sin()
    );

    let hours_pos: (f32, f32) = (
        HOURS_RADIUS * ((PI + TAU - hours_angle) - FRAC_PI_2).cos(),
        HOURS_RADIUS * ((PI + TAU - hours_angle) - FRAC_PI_2).sin()
    );

    AnglesAndPositions {
        hours_angle, minutes_angle, seconds_angle,
        hours_pos,   minutes_pos,   seconds_pos
    }
}

struct PolarClock {
    ring_pipeline: wgpu::RenderPipeline,
    disk_pipeline: wgpu::RenderPipeline,

    uniform_buffer: wgpu::Buffer,
    bind_group:     wgpu::BindGroup,

    color_index:    usize,
    last_change_ts: u64 // timestamp of the last color change transition start
}

const EXTENT: f32 = 16.0;
/// Note: cranking up the division count will increase vertex count, resulting in smoother n-gon,
/// thus reducing wasted pixel shader invocation. But it'll also result in thin/small triangles,
/// which are bad and will drastically reduce performance if set to a too high figure.
/// But it's fine for a small value like 12 (dodecagon).
/// More info: https://www.humus.name/index.php?page=News&ID=228
const DIVISION_COUNT: u32 = 12;
const THICKNESS:      f32 = 2.4;

const SECONDS_RADIUS: f32 = 13.0;
const MINUTES_RADIUS: f32 =  9.0;
const HOURS_RADIUS:   f32 =  5.0;
const DISK_RADIUS:    f32 =  0.8;

const ANIM_DURATION: f64 = 500.0;

impl ExecDraw for PolarClock {
    fn setup(
        _config:  &wgpu::SurfaceConfiguration,
        _adapter: &wgpu::Adapter,
        device:   &wgpu::Device,
        _queue:   &wgpu::Queue
    ) -> Self {
        let resources = get_resource_folder_for("polar").unwrap();

        let udspace = SingleUniformBuffer::new::<DrawspaceScales>(device, wgpu::ShaderStages::VERTEX_FRAGMENT);

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[ SingleUniformBuffer::default_layout_entry(0, &udspace) ]
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   None,
            layout:  &bind_group_layout,
            entries: &[ udspace.get_entry(0) ]
        });

        // angle, position, color data sent via push constants
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:                None,
            bind_group_layouts:   &[ &bind_group_layout ],
            push_constant_ranges: &[
                PushConstantRange {
                    stages: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    range:  0..32
                }
            ]
        });
        
        let ring_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
                read_to_string(resources.join("shaders/ring.wgsl")).unwrap().as_str()
            ))
        });

        let disk_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
                read_to_string(resources.join("shaders/disk.wgsl")).unwrap().as_str()
            ))
        });

        // use proper blending, otherwise overlapping shapes won't display correctly
        let color_target_state = wgpu::ColorTargetState {
            format: SURFACE_FORMAT,
            blend:  Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::SrcAlpha,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation:  wgpu::BlendOperation::Add
                },
                alpha: wgpu::BlendComponent::REPLACE
            }),
            write_mask: wgpu::ColorWrites::ALL
        };

        let ring_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label:         None,
            layout:        Some(&pipeline_layout),
            depth_stencil: None,
            multisample:   wgpu::MultisampleState::default(),
            multiview:     None,
            vertex: wgpu::VertexState {
                module:      &ring_shader,
                entry_point: "vs_main",
                buffers:     &[]
            },
            fragment: Some(wgpu::FragmentState {
                module:      &ring_shader,
                entry_point: "fs_main",
                targets:     &[ Some(color_target_state.clone()) ]
            }),
            primitive: wgpu::PrimitiveState {
                topology:     wgpu::PrimitiveTopology::TriangleStrip,
                cull_mode:    None,
                polygon_mode: wgpu::PolygonMode::Fill,
                ..Default::default()
            }
        });

        let disk_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label:         None,
            layout:        Some(&pipeline_layout),
            depth_stencil: None,
            multisample:   wgpu::MultisampleState::default(),
            multiview:     None,
            vertex: wgpu::VertexState {
                module:      &disk_shader,
                entry_point: "vs_main",
                buffers:     &[]
            },
            fragment: Some(wgpu::FragmentState {
                module:      &disk_shader,
                entry_point: "fs_main",
                targets:     &[ Some(color_target_state) ]
            }),
            primitive: wgpu::PrimitiveState {
                topology:     wgpu::PrimitiveTopology::TriangleStrip,
                cull_mode:    None,
                polygon_mode: wgpu::PolygonMode::Fill,
                ..Default::default()
            }
        });

        Self {
            ring_pipeline, disk_pipeline,
            bind_group,
            uniform_buffer: udspace.buffer,
            last_change_ts: 0,
            color_index: PALETTE.len() - 1
        }
    }

    fn resize(self: &mut Self, width: u32, height: u32, _device: &wgpu::Device, queue: &wgpu::Queue) {
        // rewrite the uniform buffer containing the drawspace scales since resolution was changed
        let ubuffer = DrawspaceScales::new(
            glam::Vec2::new(width as f32, height as f32),
            glam::Vec2::new(EXTENT, EXTENT)
        );

        queue.write_buffer(&self.uniform_buffer, 0, cast_struct_to_u8_slice(&ubuffer));
    }

    fn onkey(self: &mut Self, event: winit::event::KeyEvent, _device: &wgpu::Device, _queue: &wgpu::Queue) {
        if event.state == winit::event::ElementState::Pressed && !event.repeat {
            match event.key_without_modifiers().as_ref() {
                winit::keyboard::Key::Named(winit::keyboard::NamedKey::Space) => {
                    // goto the next color index (wrapping) and record the current timestamp as transition starts now
                    self.color_index    = (self.color_index + 1) % PALETTE.len();
                    self.last_change_ts = Local::now().timestamp_millis() as u64;
                },
                _ => {}
            }
        }
    }

    fn draw(self: &mut Self, texview: &wgpu::TextureView, device: &wgpu::Device, queue: &wgpu::Queue) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let ap = calc_angles_and_positions();

        // calculate the diff between current timestamp and the last recorded transition start
        let timestamp_diff = ((Local::now().timestamp_millis() as u64) - self.last_change_ts) as f64;
        // no transition past the ANIM_DURATION so clamp it. Noe divide the resultant diff by ANIM_DURATION to get t
        let t = timestamp_diff.min(ANIM_DURATION) / ANIM_DURATION;

        // starting and ending palette for linear interpolation
        let (palette0, palette1) = {
            let cindex0 = self.color_index;
            let cindex1 = (self.color_index + 1) % PALETTE.len();

            (&PALETTE[cindex0], &PALETTE[cindex1])
        };

        // use your own fav easing function
        fn ease_out_quint(t: f64) -> f64 {
            return 1.0 - (1.0 - t).powf(5.0);
        }

        let hh_color = lerp_u32_color(palette0.hour,       palette1.hour,       ease_out_quint(t));
        let mm_color = lerp_u32_color(palette0.minute,     palette1.minute,     ease_out_quint(t));
        let ss_color = lerp_u32_color(palette0.second,     palette1.second,     ease_out_quint(t));
        let cr_color = lerp_u32_color(palette0.disk,       palette1.disk,       ease_out_quint(t));
        let bg_color = lerp_u32_color(palette0.background, palette1.background, ease_out_quint(t));

        fn draw_ring(rpass: &mut wgpu::RenderPass, center: (f32, f32), radius: f32, angle: f32, color: u32) {
            let ring = RingInfo {
                center:    glam::Vec2::new(center.0, center.1),
                thickness: THICKNESS,
                divisions: DIVISION_COUNT,
                radius,
                angle,
                color
            };
        
            rpass.set_push_constants(
                wgpu::ShaderStages::VERTEX_FRAGMENT,
                0,
                cast_struct_to_u8_slice(&ring)
            );
        
            rpass.draw(0..(DIVISION_COUNT * 2 + 2), 0..1); // vertex count = 2n + 2
        }
        
        fn draw_disk(rpass: &mut wgpu::RenderPass, center: (f32, f32), radius: f32, color: u32) {
            let disk = DiskInfo {
                center:    glam::Vec2::new(center.0, center.1),
                divisions: DIVISION_COUNT,
                radius,
                color
            };
        
            rpass.set_push_constants(
                wgpu::ShaderStages::VERTEX_FRAGMENT,
                0,
                cast_struct_to_u8_slice(&disk)
            );
        
            rpass.draw(0..DIVISION_COUNT, 0..1); // vertex count = n
        }

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label:                    None,
                depth_stencil_attachment: None,
                timestamp_writes:         None,
                occlusion_query_set:      None,
                color_attachments:        &[Some(wgpu::RenderPassColorAttachment {
                    view: texview,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load:  wgpu::LoadOp::Clear(u32_col_to_wgpu_col(bg_color)),
                        store: wgpu::StoreOp::Store
                    }
                })]
            });

            rpass.set_pipeline(&self.ring_pipeline);
            rpass.set_bind_group(0, &self.bind_group, &[]);

            draw_ring(&mut rpass, (0.0, 0.0), HOURS_RADIUS,   ap.hours_angle,   hh_color);
            draw_ring(&mut rpass, (0.0, 0.0), MINUTES_RADIUS, ap.minutes_angle, mm_color);
            draw_ring(&mut rpass, (0.0, 0.0), SECONDS_RADIUS, ap.seconds_angle, ss_color);

            ////////////////////////////////////////

            rpass.set_pipeline(&self.disk_pipeline);
            rpass.set_bind_group(0, &self.bind_group, &[]);

            draw_disk(&mut rpass, ap.hours_pos,   DISK_RADIUS, cr_color);
            draw_disk(&mut rpass, ap.minutes_pos, DISK_RADIUS, cr_color);
            draw_disk(&mut rpass, ap.seconds_pos, DISK_RADIUS, cr_color);

            // Performance improvement notes:
            // This implementation is done via multiple push constant calls, one call for each shape.
            // A better implementation would be uploading the ring and disk properties into one or two instance buffers
            // and draw from those buffers, reducing draw calls.
            // Also, move the constant properties (e.g. radius, thickness) to the shader's (this kills flexibility however)
        }

        queue.submit(std::iter::once(encoder.finish()));
    }
}

fn main() {
    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    #[allow(unused_mut)]
    let mut builder = winit::window::WindowBuilder::new();
    let window = builder
        .with_inner_size(winit::dpi::LogicalSize { width: 512.0, height: 512.0 })
        .with_title("Polar Clock")
        .build(&event_loop)
        .unwrap();

    pollster::block_on(run::<PolarClock>(
        event_loop, window,
        Some(wgpu::Features::PUSH_CONSTANTS))
    );
}

// Generated using: https://coolors.co/
const PALETTE: [ColorCombo; 6] = [
    ColorCombo {
        hour:       0x171738_FF,
        minute:     0x2E1760_FF,
        second:     0x3423A6_FF,
        disk:       0xFFFFFF_FF,
        background: 0x000000_FF
    },
    ColorCombo {
        hour:       0x1B1B3A_FF,
        minute:     0x693668_FF,
        second:     0xA74482_FF,
        disk:       0xFFFFFF_FF,
        background: 0x000000_FF
    },
    ColorCombo {
        hour:       0x576232_FF,
        minute:     0xB06F25_FF,
        second:     0x92531D_FF,
        disk:       0xFFFFFF_FF,
        background: 0xFFFFFF_FF
    },
    ColorCombo {
        hour:       0x152614_FF,
        minute:     0x1E441E_FF,
        second:     0x2A7221_FF,
        disk:       0xFFFFFF_FF,
        background: 0xFFFFFF_FF
    },
    ColorCombo {
        hour:       0x000706_FF,
        minute:     0x5F6083_FF,
        second:     0x4347A5_FF,
        disk:       0xFFFFFF_FF,
        background: 0xFFFFFF_FF
    },
    ColorCombo {
        hour:       0xCFFCFF_FF,
        minute:     0xAAEFDF_FF,
        second:     0x9EE37D_FF,
        disk:       0x000000_FF,
        background: 0x000000_FF
    },
];