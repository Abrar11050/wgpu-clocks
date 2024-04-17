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
    cast_struct_to_u8_slice, run, create_vertex_and_index_buffers, cast_slice_to_u8_slice, get_resource_folder_for,
    ExecDraw, ResourceTexture, BasicFilteringSampler, SingleUniformBuffer,
    RenderTexture, DrawspaceScales, ImmutableStorageBuffer, Vtx2ID,
    SURFACE_FORMAT
};
use chrono::{Local, Timelike, Datelike};
use winit::platform::modifier_supplement::KeyEventExtModifierSupplement;

/// Resources that are recreated on window resize
struct DynamicResources {
    render_texture_view:      wgpu::TextureView, // for writing on
    render_texture_bindgroup: wgpu::BindGroup,   // for reading from
}

/// The Clock's mechanism:
/// This digital clock works very similar to how a real LED 7-segment clock would work.
/// Each LED can be illuminated individually, as if they're being powered via individual pins.
/// 
/// The clock's layout of individual "LED regions" is already made in an image editing program.
/// This is loaded as a read-only texture map.
/// The image is then imported into Blender to place individual sets of polygons (called "islands")
/// on top of individual LED regions. Each island covers only one of those LED regions.
/// Each island is given an integer ID.
/// 
/// The polygons are then exported from Blender into this code as vertex+index buffers, the island IDs are included (per vertex).
/// The full vertex buffer is drawn with the clock layout texture as sampled resource.
/// The islands those need to be illuminated, their IDs are sent encoded into a set of bitflags via push constants.
/// The vertex shader tests the current vertex's island ID against the bitflags, and assigns on/off status depending on the bit status.
/// The fragment shader will then assign light/darker color depending on the on/off status.
/// 
/// Extra two more passes are included for the glow effect using two-pass gaussian blur, this is optional to this clock.
/// This two pass version has a time complexity of O(n), which is fine because the single pass version would have
/// time complexity of O(n^2), which is crazy resource hungry and GPU usage goes out of the roof as you crank up the blur radius.
struct DigiClock {
    forward_pipeline: wgpu::RenderPipeline,
    filter_pipeline:  wgpu::RenderPipeline,

    vertex_buffer:  wgpu::Buffer,
    index_buffer:   wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,

    resource_texture_bindgroup: wgpu::BindGroup,
    uniform_buffer_bindgroup:   wgpu::BindGroup,
    blur_table_bindgroup:       wgpu::BindGroup,

    dynamic_resources: [DynamicResources; 2], // two for two blur passes (horizontal and vertical)

    is_12_hours: bool,
    selector:    u32 // color palette selector
}

#[repr(C, align(8))]
struct ClockData {
    flagset:   [u32; 2], // actual LED on/off states are encoded in these two
    selector:  u32, // color palette selector, unrelated to clock
    timestamp: f32 // for animation, unrelated to clock
}

#[derive(Debug)]
#[repr(C, align(8))]
struct BlurWO {
    weight: f32,
    offset: f32
}

/// While calculating gaussian blur, the same weights will be generated for all pixels,
/// to cut out this redundant calc, we move that to the CPU from the fragment shader.
/// This is only done once. Both weights and pixel offsets are calculated,
/// and then sent to the fragment shader as a read-only storage buffer.
/// The shader treats this buffer as a look-up table.
/// This the rustified version of the JS code found in: https://lisyarus.github.io/blog/graphics/2023/02/24/blur-coefficients-generator.html
/// So this function is not my code.
fn create_blur_weights_and_offsets(
    radius:     i32,
    sigma:      f32,
    linear:     bool,
    correction: bool
) -> Result<Vec<BlurWO>, &'static str> {
    if radius < 1 {
        return Err("Radius must be 1 or up");
    }

    if sigma == 0.0 {
        return Err("Sigma cannot be 0");
    }

    // From https://hewgill.com/picomath/javascript/erf.js.html
    fn erf(x: f32) -> f32 {
        // constants
        let a1: f32 =  0.254829592;
        let a2: f32 = -0.284496736;
        let a3: f32 =  1.421413741;
        let a4: f32 = -1.453152027;
        let a5: f32 =  1.061405429;
        let  p: f32 =  0.3275911;
    
        // Save the sign of x
        let mut sign: f32 = 1.0;
        if x < 0.0 {
            sign = -1.0;
        }

        let x = x.abs();
    
        // A&S formula 7.1.26
        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    
        return sign * y;
    }

    let mut sum_weights: f32 = 0.0;

    let mut weights: Vec<f32> = (-radius..radius+1).map(|i| {
        let i = i as f32;

        let w = if correction {
            (erf((i + 0.5) / sigma / 2.0_f32.sqrt()) - erf((i - 0.5) / sigma / 2.0_f32.sqrt())) / 2.0
        } else {
            (-i * i / sigma / sigma).exp()
        };

        sum_weights += w;

        return w;
    }).collect();

    let inv_sum_weights = 1.0 / sum_weights;
    for i in 0..weights.len() {
        weights[i] *= inv_sum_weights;
    }

    let weights_and_offsets: Vec<BlurWO> = if linear {
        (-radius..radius+1).step_by(2).map(|i| {
            if i == radius {
                BlurWO {
                    offset: i as f32,
                    weight: weights[(i + radius) as usize]
                }
            } else {
                let w0 = weights[(i + radius + 0) as usize];
                let w1 = weights[(i + radius + 1) as usize];
                let w = w0 + w1;

                let o: f32 = if w > 0.0 {
                    (i as f32) + w1 / w
                } else {
                    i as f32
                };

                BlurWO {
                    offset: o,
                    weight: w
                }
            }
        }).collect()
    } else {
        (-radius..radius+1).enumerate().map(|(index, off)| {
            BlurWO {
                offset: off as f32,
                weight: weights[index]
            }
        }).collect()
    };

    Ok(weights_and_offsets)
}

fn create_dynamic_resources(
    texsize: (u32, u32),
    sampler: &BasicFilteringSampler,
    device:  &wgpu::Device
) -> (DynamicResources, wgpu::BindGroupLayout) {
    let render_texture = RenderTexture::new(
        texsize, SURFACE_FORMAT,
        true, device
    );

    let render_texture_bindgroup_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[ RenderTexture::default_layout_entry(0), BasicFilteringSampler::default_layout_entry(1) ]
    });

    let render_texture_bindgroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label:   None,
        layout:  &render_texture_bindgroup_layout,
        entries: &[ render_texture.get_entry(0), sampler.get_entry(1) ]
    });

    let dynamic = DynamicResources {
        render_texture_view: render_texture.view,
        render_texture_bindgroup
    };

    (dynamic, render_texture_bindgroup_layout)
}

/// Generate the gblur look-up table:
/// 1. The actual table storage buffer containing weights and offsets.
/// 2. A single value uniform buffer for the count (table length).
/// Both welded into a single bindgroup.
/// (Could've put the count in the storage buffer at index 0, what was I thinking then? :P)
fn create_blur_table_bindgroup(
    radius:     i32,
    sigma:      f32,
    linear:     bool,
    correction: bool,
    device:     &wgpu::Device,
    queue:      &wgpu::Queue
) -> (wgpu::BindGroup, wgpu::BindGroupLayout) {
    let weights_and_offsets = create_blur_weights_and_offsets(radius, sigma, linear, correction).unwrap();

    let stages = wgpu::ShaderStages::FRAGMENT;

    let storage = ImmutableStorageBuffer::new(
        device, stages,
        cast_slice_to_u8_slice(weights_and_offsets.as_slice())
    );

    let uniform = SingleUniformBuffer::new::<u32>(device, stages);

    let bindgroup_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            ImmutableStorageBuffer::default_layout_entry(0, &storage),
            SingleUniformBuffer::default_layout_entry(1, &uniform)
        ]
    });

    let bindgroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label:   None,
        layout:  &bindgroup_layout,
        entries: &[
            storage.get_entry(0),
            uniform.get_entry(1)
        ]
    });

    let data: u32 = weights_and_offsets.len() as u32;
    queue.write_buffer(&uniform.buffer, 0, cast_struct_to_u8_slice(&data));

    (bindgroup, bindgroup_layout)
}

/// Calculate bit flags from current time
/// 
/// Flagset 0:
/// 
///     * bits [0..6]   => hour tens
/// 
///     * bits [7..13]  => hour ones
/// 
///     * bits [14..20] => minute tens
/// 
///     * bits [21..27] => minute ones
/// 
/// Flagset 1:
/// 
///     * bits [0..6]   => day of week
/// 
///     * bit 7 => AM indicator
/// 
///     * bit 8 => PM indicator
/// 
///     * bit 9 => colon
fn calculate_clock_data(hr12: bool, selector: u32) -> ClockData {
    let now = Local::now();

    let mut hours = now.hour();
    let minutes = now.minute();
    let upper_half_sec = now.nanosecond() > 500_000_000;
    let weekday = now.weekday() as usize;

    let mut am = false;
    let mut pm = false;

    if hr12 {
        if hours >= 12 {
            pm = true;
        } else {
            am = true;
        }

        hours %= 12;

        if hours == 0 {
            hours = 12;
        }
    }

    let mut flags0: u32 = 0;
    let mut flags1: u32 = 0;

    // special case for hour tens digit, turn it off completely when it is zero
    flags0 |= if (hours / 10) != 0 {
        DIGIT_SEGMENT_FLAGS[(hours / 10) as usize] << 0
    } else {
        0
    };
    
    flags0 |= DIGIT_SEGMENT_FLAGS[(hours % 10) as usize] << 7;

    flags0 |= DIGIT_SEGMENT_FLAGS[(minutes / 10) as usize] << 14;
    flags0 |= DIGIT_SEGMENT_FLAGS[(minutes % 10) as usize] << 21;

    // made a mistake while designing the clock layout
    // didn't realize chrono's week starts with different index than mine
    flags1 |= 1 << ((weekday + 1) % 7);
    
    flags1 |= (if am { 1 } else { 0 }) << 7;
    flags1 |= (if pm { 1 } else { 0 }) << 8;

    flags1 |= (if upper_half_sec { 1 } else { 0 }) << 9;

    let timestamp = now.second() as f32 + now.nanosecond() as f32 / 1_000_000_000.0;

    ClockData { flagset: [flags0, flags1], selector, timestamp }
}

const SELECTOR_LENGTH: u32 = 5;

impl ExecDraw for DigiClock {
    fn setup(
        config:   &wgpu::SurfaceConfiguration,
        _adapter: &wgpu::Adapter,
        device:   &wgpu::Device,
        queue:    &wgpu::Queue
    ) -> Self {
        let resources = get_resource_folder_for("digital").unwrap();
        
        let (vertex_buffer, index_buffer) = create_vertex_and_index_buffers(
            device,
            cast_slice_to_u8_slice(&VERTICES),
            cast_slice_to_u8_slice(&INDICES)
        );

        let backtex = ResourceTexture::new(
            resources.join("textures/clock_layout.png").as_path().to_str().unwrap(),
            device,
            queue
        );

        let sampler = BasicFilteringSampler::new(device);

        let udspace = SingleUniformBuffer::new::<DrawspaceScales>(device, wgpu::ShaderStages::VERTEX_FRAGMENT);

        let uniform_buffer_bindgroup_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[ SingleUniformBuffer::default_layout_entry(0, &udspace) ]
        });

        let uniform_buffer_bindgroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   None,
            layout:  &uniform_buffer_bindgroup_layout,
            entries: &[ udspace.get_entry(0) ]
        });

        let resource_texture_bindgroup_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[ ResourceTexture::default_layout_entry(0), BasicFilteringSampler::default_layout_entry(1) ]
        });

        let resource_texture_bindgroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   None,
            layout:  &resource_texture_bindgroup_layout,
            entries: &[ backtex.get_entry(0), sampler.get_entry(1) ]
        });

        let (dynamic_resources_0, render_texture_bindgroup_layout) = create_dynamic_resources(
            (config.width, config.height), &sampler, device
        );
        let (dynamic_resources_1, _) = create_dynamic_resources(
            (config.width, config.height), &sampler, device
        );
        let dynamic_resources = [dynamic_resources_0, dynamic_resources_1];

        let (blur_table_bindgroup, blur_table_bindgroup_layout) = create_blur_table_bindgroup(40, 10.0, true, true, device, queue);

        let forward_pipeline = {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label:  None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
                    read_to_string(resources.join("shaders/forward.wgsl")).unwrap().as_str()
                ))
            });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label:                None,
                bind_group_layouts:   &[ &resource_texture_bindgroup_layout, &uniform_buffer_bindgroup_layout ],
                push_constant_ranges: &[
                    PushConstantRange {
                        stages: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        range:  0..16
                    }
                ]
            });

            device.create_render_pipeline(&RenderPipelineDescriptor {
                label:         None,
                layout:        Some(&pipeline_layout),
                depth_stencil: None,
                multisample:   wgpu::MultisampleState::default(),
                multiview:     None,
                vertex: wgpu::VertexState {
                    module:      &shader,
                    entry_point: "vs_main",
                    buffers:     &[
                        wgpu::VertexBufferLayout {
                            array_stride: std::mem::size_of::<Vtx2ID>() as wgpu::BufferAddress,
                            step_mode:    wgpu::VertexStepMode::Vertex,
                            attributes:   &[
                                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, shader_location: 0, offset: 0 },
                                wgpu::VertexAttribute { format: wgpu::VertexFormat::Uint32,    shader_location: 1, offset: 2 * std::mem::size_of::<f32>() as u64 }
                            ]
                        }
                    ]
                },
                fragment: Some(wgpu::FragmentState {
                    module:      &shader,
                    entry_point: "fs_main",
                    targets:     &[ Some(SURFACE_FORMAT.into()) ]
                }),
                primitive: wgpu::PrimitiveState {
                    topology:     wgpu::PrimitiveTopology::TriangleList,
                    cull_mode:    None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    ..Default::default()
                }
            })
        };

        ///////////////////////////////////////////

        // same pipeline used for both horizontal and vertical blurring,
        // the selection is sent via push constant
        let filter_pipeline = {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label:  None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
                    read_to_string(resources.join("shaders/filter.wgsl")).unwrap().as_str()
                ))
            });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label:                None,
                bind_group_layouts:   &[
                    &render_texture_bindgroup_layout,
                    &uniform_buffer_bindgroup_layout,
                    &blur_table_bindgroup_layout,
                    &render_texture_bindgroup_layout
                ],
                push_constant_ranges: &[
                    wgpu::PushConstantRange {
                        stages: wgpu::ShaderStages::FRAGMENT,
                        range:  0..4
                    }
                ]
            });

            // fullscreen quad drawing pipeline
            device.create_render_pipeline(&RenderPipelineDescriptor {
                label:         None,
                layout:        Some(&pipeline_layout),
                depth_stencil: None,
                multisample:   wgpu::MultisampleState::default(),
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
                    cull_mode:    None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    ..Default::default()
                }
            })
        };

        Self {
            forward_pipeline,
            filter_pipeline,

            vertex_buffer,
            index_buffer,
            uniform_buffer: udspace.buffer,

            resource_texture_bindgroup,
            uniform_buffer_bindgroup,
            blur_table_bindgroup,

            dynamic_resources,

            is_12_hours: false,
            selector: 0
        }
    }

    fn onkey(self: &mut Self, event: winit::event::KeyEvent, _device: &wgpu::Device, _queue: &wgpu::Queue) {
        if event.state == winit::event::ElementState::Pressed && !event.repeat {
            match event.key_without_modifiers().as_ref() {
                winit::keyboard::Key::Named(winit::keyboard::NamedKey::Space) => {
                    self.selector = (self.selector + 1) % SELECTOR_LENGTH;
                },
                winit::keyboard::Key::Character("T") | winit::keyboard::Key::Character("t") => {
                    self.is_12_hours = !self.is_12_hours;
                }
                _ => {}
            }
        }
    }

    fn resize(self: &mut Self, width: u32, height: u32, device: &wgpu::Device, queue: &wgpu::Queue) {
        let sampler = BasicFilteringSampler::new(device);
        let (dynamic_resources_0, _) = create_dynamic_resources((width, height), &sampler, device);
        let (dynamic_resources_1, _) = create_dynamic_resources((width, height), &sampler, device);

        self.dynamic_resources = [dynamic_resources_0, dynamic_resources_1];

        let ubuffer = DrawspaceScales::new(
            glam::Vec2::new(width as f32, height as f32),
            glam::Vec2::new(2.5, 1.40625)
        );

        queue.write_buffer(&self.uniform_buffer, 0, cast_struct_to_u8_slice(&ubuffer));

        let radius_scale: f32 = 1.0;

        // adapt the blur radius according to current pixel density
        // the factors are tuned via T&E
        let blur_radius = ((ubuffer.density as f32 / 204.0) * 40.0 * radius_scale) as i32;
        let blur_sigma  = (blur_radius as f32) * 0.25;

        self.blur_table_bindgroup = create_blur_table_bindgroup(
            blur_radius,
            blur_sigma,
            true, true,
            device, queue
        ).0;
    }

    fn draw(self: &mut Self, texview: &wgpu::TextureView, device: &wgpu::Device, queue: &wgpu::Queue) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // Data flow:
        // [Forward Pass] => [Horizontal Blur Pass] => [Vertical Blur+Compositing Pass] => [Present]

        let triangle_render_dst = &self.dynamic_resources[0].render_texture_view;
        let horzblur_render_src = &self.dynamic_resources[0].render_texture_bindgroup;
        let horzblur_render_dst = &self.dynamic_resources[1].render_texture_view;
        let vertblur_render_src = &self.dynamic_resources[1].render_texture_bindgroup;
        let vertblur_render_dst = texview;

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label:                    None,
                depth_stencil_attachment: None,
                timestamp_writes:         None,
                occlusion_query_set:      None,
                color_attachments:        &[Some(wgpu::RenderPassColorAttachment {
                    view: triangle_render_dst,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load:  wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }),
                        store: wgpu::StoreOp::Store
                    }
                })]
            });

            rpass.set_pipeline(&self.forward_pipeline);
            rpass.set_bind_group(0, &self.resource_texture_bindgroup, &[]);
            rpass.set_bind_group(1, &self.uniform_buffer_bindgroup,   &[]);

            rpass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));

            let cdata = calculate_clock_data(self.is_12_hours, self.selector);
            rpass.set_push_constants(wgpu::ShaderStages::VERTEX_FRAGMENT, 0, cast_struct_to_u8_slice(&cdata));

            rpass.draw_indexed(0..INDEX_COUNT as u32, 0, 0..1);
        }

        let mut apply_blur_pass = |source: &wgpu::BindGroup, destination: &wgpu::TextureView, vertical: bool| {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label:                    None,
                depth_stencil_attachment: None,
                timestamp_writes:         None,
                occlusion_query_set:      None,
                color_attachments:        &[Some(wgpu::RenderPassColorAttachment {
                    view: destination,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load:  wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }),
                        store: wgpu::StoreOp::Store
                    }
                })]
            });

            let vertical: u32 = if vertical { 1 } else { 0 };

            rpass.set_pipeline(&self.filter_pipeline);
            rpass.set_bind_group(0, source, &[]);
            rpass.set_bind_group(1, &self.uniform_buffer_bindgroup, &[]);
            rpass.set_bind_group(2, &self.blur_table_bindgroup, &[]);
            rpass.set_bind_group(3, horzblur_render_src, &[]); // common for both
            rpass.set_push_constants(wgpu::ShaderStages::FRAGMENT, 0, cast_slice_to_u8_slice(&[vertical]));

            rpass.draw(0..4, 0..1);
        };

        apply_blur_pass(horzblur_render_src, horzblur_render_dst, false);
        apply_blur_pass(vertblur_render_src, vertblur_render_dst, true);

        queue.submit(std::iter::once(encoder.finish()));
    }
}

fn main() {
    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    #[allow(unused_mut)]
    let mut builder = winit::window::WindowBuilder::new();
    let window = builder
        .with_inner_size(winit::dpi::LogicalSize { width: 1024.0, height: 576.0 })
        .with_title("Digital Clock")
        .build(&event_loop)
        .unwrap();

    pollster::block_on(run::<DigiClock>(
        event_loop, window,
        Some(wgpu::Features::PUSH_CONSTANTS)
    ));
}

// 7-segment display segment mapping table
const DIGIT_SEGMENT_FLAGS: [u32; 10] = [
    0b1110111,
    0b1000100,
    0b1011011,
    0b1011101,
    0b1101100,
    0b0111101,
    0b0111111,
    0b1010100,
    0b1111111,
    0b1111101
];

// The vertex buffer containing 2D position and island ID
const VERTEX_COUNT: usize = 160;
const VERTICES: [Vtx2ID; VERTEX_COUNT] = [
    Vtx2ID { pos: glam::Vec2::new(-2.095772, -0.643742), id:  0 },
    Vtx2ID { pos: glam::Vec2::new( -1.35851, -0.643742), id:  0 },
    Vtx2ID { pos: glam::Vec2::new(-1.896928, -0.453839), id:  0 },
    Vtx2ID { pos: glam::Vec2::new(-1.524128, -0.453839), id:  0 },
    Vtx2ID { pos: glam::Vec2::new(-2.118776,  -0.59405), id:  1 },
    Vtx2ID { pos: glam::Vec2::new(-1.932165, -0.421071), id:  1 },
    Vtx2ID { pos: glam::Vec2::new(-2.075348,  0.031189), id:  1 },
    Vtx2ID { pos: glam::Vec2::new(-1.906402, -0.154733), id:  1 },
    Vtx2ID { pos: glam::Vec2::new( -1.49785, -0.414179), id:  2 },
    Vtx2ID { pos: glam::Vec2::new(  -1.3106, -0.640586), id:  2 },
    Vtx2ID { pos: glam::Vec2::new(-1.474955, -0.145988), id:  2 },
    Vtx2ID { pos: glam::Vec2::new(-1.254634,  0.029795), id:  2 },
    Vtx2ID { pos: glam::Vec2::new(-1.898295, -0.086802), id:  3 },
    Vtx2ID { pos: glam::Vec2::new( -1.45768, -0.086802), id:  3 },
    Vtx2ID { pos: glam::Vec2::new(-1.885434,   0.09269), id:  3 },
    Vtx2ID { pos: glam::Vec2::new(-1.443306,   0.09269), id:  3 },
    Vtx2ID { pos: glam::Vec2::new(-1.984385,  0.002944), id:  3 },
    Vtx2ID { pos: glam::Vec2::new(-1.362022,  0.002944), id:  3 },
    Vtx2ID { pos: glam::Vec2::new(-1.819779,  0.456748), id:  4 },
    Vtx2ID { pos: glam::Vec2::new(-1.436929,  0.456748), id:  4 },
    Vtx2ID { pos: glam::Vec2::new(-1.984956,  0.642849), id:  4 },
    Vtx2ID { pos: glam::Vec2::new(-1.246313,  0.642849), id:  4 },
    Vtx2ID { pos: glam::Vec2::new(-2.067446,  0.002886), id:  5 },
    Vtx2ID { pos: glam::Vec2::new(-1.878717,  0.160996), id:  5 },
    Vtx2ID { pos: glam::Vec2::new(-2.019454,  0.606703), id:  5 },
    Vtx2ID { pos: glam::Vec2::new(-1.851334,   0.42651), id:  5 },
    Vtx2ID { pos: glam::Vec2::new(  -1.4457,  0.160552), id:  6 },
    Vtx2ID { pos: glam::Vec2::new(-1.270013, -0.009925), id:  6 },
    Vtx2ID { pos: glam::Vec2::new(-1.416551,  0.429421), id:  6 },
    Vtx2ID { pos: glam::Vec2::new( -1.21274,  0.639647), id:  6 },
    Vtx2ID { pos: glam::Vec2::new(-1.077749, -0.643742), id:  7 },
    Vtx2ID { pos: glam::Vec2::new(-0.340487, -0.643742), id:  7 },
    Vtx2ID { pos: glam::Vec2::new(-0.878905, -0.453839), id:  7 },
    Vtx2ID { pos: glam::Vec2::new(-0.506105, -0.453839), id:  7 },
    Vtx2ID { pos: glam::Vec2::new(-1.100754,  -0.59405), id:  8 },
    Vtx2ID { pos: glam::Vec2::new(-0.914142, -0.421071), id:  8 },
    Vtx2ID { pos: glam::Vec2::new(-1.057325,  0.031189), id:  8 },
    Vtx2ID { pos: glam::Vec2::new(-0.888379, -0.154733), id:  8 },
    Vtx2ID { pos: glam::Vec2::new(-0.479828, -0.414179), id:  9 },
    Vtx2ID { pos: glam::Vec2::new(-0.292578, -0.640586), id:  9 },
    Vtx2ID { pos: glam::Vec2::new(-0.456933, -0.145988), id:  9 },
    Vtx2ID { pos: glam::Vec2::new(-0.236612,  0.029795), id:  9 },
    Vtx2ID { pos: glam::Vec2::new(-0.880272, -0.086802), id: 10 },
    Vtx2ID { pos: glam::Vec2::new(-0.439658, -0.086802), id: 10 },
    Vtx2ID { pos: glam::Vec2::new(-0.867411,   0.09269), id: 10 },
    Vtx2ID { pos: glam::Vec2::new(-0.425284,   0.09269), id: 10 },
    Vtx2ID { pos: glam::Vec2::new(-0.966362,  0.002944), id: 10 },
    Vtx2ID { pos: glam::Vec2::new(-0.343999,  0.002944), id: 10 },
    Vtx2ID { pos: glam::Vec2::new(-0.801757,  0.456748), id: 11 },
    Vtx2ID { pos: glam::Vec2::new(-0.418906,  0.456748), id: 11 },
    Vtx2ID { pos: glam::Vec2::new(-0.966933,  0.642849), id: 11 },
    Vtx2ID { pos: glam::Vec2::new( -0.22829,  0.642849), id: 11 },
    Vtx2ID { pos: glam::Vec2::new(-1.049423,  0.002886), id: 12 },
    Vtx2ID { pos: glam::Vec2::new(-0.860694,  0.160996), id: 12 },
    Vtx2ID { pos: glam::Vec2::new(-1.001431,  0.606703), id: 12 },
    Vtx2ID { pos: glam::Vec2::new(-0.833312,   0.42651), id: 12 },
    Vtx2ID { pos: glam::Vec2::new(-0.427677,  0.160552), id: 13 },
    Vtx2ID { pos: glam::Vec2::new( -0.25199, -0.009925), id: 13 },
    Vtx2ID { pos: glam::Vec2::new(-0.398528,  0.429421), id: 13 },
    Vtx2ID { pos: glam::Vec2::new(-0.194717,  0.639647), id: 13 },
    Vtx2ID { pos: glam::Vec2::new( 0.195464, -0.643742), id: 14 },
    Vtx2ID { pos: glam::Vec2::new( 0.932727, -0.643742), id: 14 },
    Vtx2ID { pos: glam::Vec2::new( 0.394308, -0.453839), id: 14 },
    Vtx2ID { pos: glam::Vec2::new( 0.767108, -0.453839), id: 14 },
    Vtx2ID { pos: glam::Vec2::new(  0.17246,  -0.59405), id: 15 },
    Vtx2ID { pos: glam::Vec2::new( 0.359071, -0.421071), id: 15 },
    Vtx2ID { pos: glam::Vec2::new( 0.215888,  0.031189), id: 15 },
    Vtx2ID { pos: glam::Vec2::new( 0.384834, -0.154733), id: 15 },
    Vtx2ID { pos: glam::Vec2::new( 0.793386, -0.414179), id: 16 },
    Vtx2ID { pos: glam::Vec2::new( 0.980636, -0.640586), id: 16 },
    Vtx2ID { pos: glam::Vec2::new( 0.816281, -0.145988), id: 16 },
    Vtx2ID { pos: glam::Vec2::new( 1.036602,  0.029795), id: 16 },
    Vtx2ID { pos: glam::Vec2::new( 0.392941, -0.086802), id: 17 },
    Vtx2ID { pos: glam::Vec2::new( 0.833556, -0.086802), id: 17 },
    Vtx2ID { pos: glam::Vec2::new( 0.405802,   0.09269), id: 17 },
    Vtx2ID { pos: glam::Vec2::new(  0.84793,   0.09269), id: 17 },
    Vtx2ID { pos: glam::Vec2::new( 0.306852,  0.002944), id: 17 },
    Vtx2ID { pos: glam::Vec2::new( 0.929214,  0.002944), id: 17 },
    Vtx2ID { pos: glam::Vec2::new( 0.471457,  0.456748), id: 18 },
    Vtx2ID { pos: glam::Vec2::new( 0.854307,  0.456748), id: 18 },
    Vtx2ID { pos: glam::Vec2::new(  0.30628,  0.642849), id: 18 },
    Vtx2ID { pos: glam::Vec2::new( 1.044923,  0.642849), id: 18 },
    Vtx2ID { pos: glam::Vec2::new(  0.22379,  0.002886), id: 19 },
    Vtx2ID { pos: glam::Vec2::new( 0.412519,  0.160996), id: 19 },
    Vtx2ID { pos: glam::Vec2::new( 0.271783,  0.606703), id: 19 },
    Vtx2ID { pos: glam::Vec2::new( 0.439902,   0.42651), id: 19 },
    Vtx2ID { pos: glam::Vec2::new( 0.845536,  0.160552), id: 20 },
    Vtx2ID { pos: glam::Vec2::new( 1.021223, -0.009925), id: 20 },
    Vtx2ID { pos: glam::Vec2::new( 0.874685,  0.429421), id: 20 },
    Vtx2ID { pos: glam::Vec2::new( 1.078496,  0.639647), id: 20 },
    Vtx2ID { pos: glam::Vec2::new( 1.213487, -0.643742), id: 21 },
    Vtx2ID { pos: glam::Vec2::new( 1.950749, -0.643742), id: 21 },
    Vtx2ID { pos: glam::Vec2::new( 1.412331, -0.453839), id: 21 },
    Vtx2ID { pos: glam::Vec2::new( 1.785131, -0.453839), id: 21 },
    Vtx2ID { pos: glam::Vec2::new( 1.190482,  -0.59405), id: 22 },
    Vtx2ID { pos: glam::Vec2::new( 1.377094, -0.421071), id: 22 },
    Vtx2ID { pos: glam::Vec2::new( 1.233911,  0.031189), id: 22 },
    Vtx2ID { pos: glam::Vec2::new( 1.402857, -0.154733), id: 22 },
    Vtx2ID { pos: glam::Vec2::new( 1.811408, -0.414179), id: 23 },
    Vtx2ID { pos: glam::Vec2::new( 1.998658, -0.640586), id: 23 },
    Vtx2ID { pos: glam::Vec2::new( 1.834303, -0.145988), id: 23 },
    Vtx2ID { pos: glam::Vec2::new( 2.054624,  0.029795), id: 23 },
    Vtx2ID { pos: glam::Vec2::new( 1.410964, -0.086802), id: 24 },
    Vtx2ID { pos: glam::Vec2::new( 1.851578, -0.086802), id: 24 },
    Vtx2ID { pos: glam::Vec2::new( 1.423825,   0.09269), id: 24 },
    Vtx2ID { pos: glam::Vec2::new( 1.865952,   0.09269), id: 24 },
    Vtx2ID { pos: glam::Vec2::new( 1.324874,  0.002944), id: 24 },
    Vtx2ID { pos: glam::Vec2::new( 1.947237,  0.002944), id: 24 },
    Vtx2ID { pos: glam::Vec2::new(  1.48948,  0.456748), id: 25 },
    Vtx2ID { pos: glam::Vec2::new(  1.87233,  0.456748), id: 25 },
    Vtx2ID { pos: glam::Vec2::new( 1.324303,  0.642849), id: 25 },
    Vtx2ID { pos: glam::Vec2::new( 2.062946,  0.642849), id: 25 },
    Vtx2ID { pos: glam::Vec2::new( 1.241813,  0.002886), id: 26 },
    Vtx2ID { pos: glam::Vec2::new( 1.430542,  0.160996), id: 26 },
    Vtx2ID { pos: glam::Vec2::new( 1.289805,  0.606703), id: 26 },
    Vtx2ID { pos: glam::Vec2::new( 1.457924,   0.42651), id: 26 },
    Vtx2ID { pos: glam::Vec2::new( 1.863559,  0.160552), id: 27 },
    Vtx2ID { pos: glam::Vec2::new( 2.039246, -0.009925), id: 27 },
    Vtx2ID { pos: glam::Vec2::new( 1.892708,  0.429421), id: 27 },
    Vtx2ID { pos: glam::Vec2::new( 2.096519,  0.639647), id: 27 },
    Vtx2ID { pos: glam::Vec2::new(-2.319725,  0.893913), id: 32 },
    Vtx2ID { pos: glam::Vec2::new(-1.767762,  0.893913), id: 32 },
    Vtx2ID { pos: glam::Vec2::new(-2.319725,  1.204886), id: 32 },
    Vtx2ID { pos: glam::Vec2::new(-1.767762,  1.204886), id: 32 },
    Vtx2ID { pos: glam::Vec2::new(-1.634569,  0.893913), id: 33 },
    Vtx2ID { pos: glam::Vec2::new(-1.082606,  0.893913), id: 33 },
    Vtx2ID { pos: glam::Vec2::new(-1.634569,  1.204886), id: 33 },
    Vtx2ID { pos: glam::Vec2::new(-1.082606,  1.204886), id: 33 },
    Vtx2ID { pos: glam::Vec2::new(-0.942275,  0.893913), id: 34 },
    Vtx2ID { pos: glam::Vec2::new(-0.390313,  0.893913), id: 34 },
    Vtx2ID { pos: glam::Vec2::new(-0.942275,  1.204886), id: 34 },
    Vtx2ID { pos: glam::Vec2::new(-0.390313,  1.204886), id: 34 },
    Vtx2ID { pos: glam::Vec2::new(-0.257119,  0.893913), id: 35 },
    Vtx2ID { pos: glam::Vec2::new( 0.294844,  0.893913), id: 35 },
    Vtx2ID { pos: glam::Vec2::new(-0.257119,  1.204886), id: 35 },
    Vtx2ID { pos: glam::Vec2::new( 0.294844,  1.204886), id: 35 },
    Vtx2ID { pos: glam::Vec2::new( 0.431606,  0.893913), id: 36 },
    Vtx2ID { pos: glam::Vec2::new( 0.983569,  0.893913), id: 36 },
    Vtx2ID { pos: glam::Vec2::new( 0.431606,  1.204886), id: 36 },
    Vtx2ID { pos: glam::Vec2::new( 0.983569,  1.204886), id: 36 },
    Vtx2ID { pos: glam::Vec2::new( 1.116762,  0.893913), id: 37 },
    Vtx2ID { pos: glam::Vec2::new( 1.668725,  0.893913), id: 37 },
    Vtx2ID { pos: glam::Vec2::new( 1.116762,  1.204886), id: 37 },
    Vtx2ID { pos: glam::Vec2::new( 1.668725,  1.204886), id: 37 },
    Vtx2ID { pos: glam::Vec2::new( 1.809056,  0.893913), id: 38 },
    Vtx2ID { pos: glam::Vec2::new( 2.361018,  0.893913), id: 38 },
    Vtx2ID { pos: glam::Vec2::new( 1.809056,  1.204886), id: 38 },
    Vtx2ID { pos: glam::Vec2::new( 2.361018,  1.204886), id: 38 },
    Vtx2ID { pos: glam::Vec2::new(-0.632139, -1.180878), id: 39 },
    Vtx2ID { pos: glam::Vec2::new(-0.245536, -1.180878), id: 39 },
    Vtx2ID { pos: glam::Vec2::new(-0.632139, -0.905955), id: 39 },
    Vtx2ID { pos: glam::Vec2::new(-0.245536, -0.905955), id: 39 },
    Vtx2ID { pos: glam::Vec2::new( 0.231189, -1.180878), id: 40 },
    Vtx2ID { pos: glam::Vec2::new( 0.617792, -1.180878), id: 40 },
    Vtx2ID { pos: glam::Vec2::new( 0.231189, -0.905955), id: 40 },
    Vtx2ID { pos: glam::Vec2::new( 0.617792, -0.905955), id: 40 },
    Vtx2ID { pos: glam::Vec2::new( -0.18772, -0.403361), id: 41 },
    Vtx2ID { pos: glam::Vec2::new( 0.091462, -0.431448), id: 41 },
    Vtx2ID { pos: glam::Vec2::new(-0.106829,  0.400715), id: 41 },
    Vtx2ID { pos: glam::Vec2::new( 0.172354,  0.372628), id: 41 }
];

// And the index buffer
const INDEX_COUNT: usize = 252;
const INDICES: [u16; INDEX_COUNT] = [
    1, 2, 0,
    1, 3, 2,
    5, 6, 4,
    5, 7, 6,
    8, 11, 10,
    8, 9, 11,
    17, 14, 16,
    13, 16, 12,
    17, 15, 14,
    13, 17, 16,
    19, 20, 18,
    19, 21, 20,
    23, 24, 22,
    23, 25, 24,
    27, 28, 26,
    27, 29, 28,
    31, 32, 30,
    31, 33, 32,
    35, 36, 34,
    35, 37, 36,
    38, 41, 40,
    38, 39, 41,
    47, 44, 46,
    43, 46, 42,
    47, 45, 44,
    43, 47, 46,
    49, 50, 48,
    49, 51, 50,
    53, 54, 52,
    53, 55, 54,
    57, 58, 56,
    57, 59, 58,
    61, 62, 60,
    61, 63, 62,
    65, 66, 64,
    65, 67, 66,
    68, 71, 70,
    68, 69, 71,
    77, 74, 76,
    73, 76, 72,
    77, 75, 74,
    73, 77, 76,
    79, 80, 78,
    79, 81, 80,
    83, 84, 82,
    83, 85, 84,
    87, 88, 86,
    87, 89, 88,
    91, 92, 90,
    91, 93, 92,
    95, 96, 94,
    95, 97, 96,
    98, 101, 100,
    98, 99, 101,
    107, 104, 106,
    103, 106, 102,
    107, 105, 104,
    103, 107, 106,
    109, 110, 108,
    109, 111, 110,
    113, 114, 112,
    113, 115, 114,
    117, 118, 116,
    117, 119, 118,
    121, 122, 120,
    121, 123, 122,
    125, 126, 124,
    125, 127, 126,
    129, 130, 128,
    129, 131, 130,
    133, 134, 132,
    133, 135, 134,
    137, 138, 136,
    137, 139, 138,
    141, 142, 140,
    141, 143, 142,
    145, 146, 144,
    145, 147, 146,
    149, 150, 148,
    149, 151, 150,
    153, 154, 152,
    153, 155, 154,
    157, 158, 156,
    157, 159, 158
];