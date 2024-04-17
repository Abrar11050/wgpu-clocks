#![cfg_attr(
    all(
        target_os = "windows",
        not(feature = "console"),
    ),
    windows_subsystem = "windows"
)]
#![allow(non_snake_case)]
use std::{borrow::Cow, fs::read_to_string};
use wgpu::RenderPipelineDescriptor;
use clockutils::{
    run, cast_struct_to_u8_slice, get_resource_folder_for, 
    ExecDraw, SingleUniformBuffer, DrawspaceScales, RenderTexture,
    ResourceTexture, BasicFilteringSampler, Vtx3UV, PlyGeoBuffers,
    SURFACE_FORMAT
};
use chrono::{Local, Timelike};
use winit::platform::modifier_supplement::KeyEventExtModifierSupplement;

#[repr(C, align(8))]
struct MatrixData {
    matrix: glam::Mat4
}

fn calc_matrix_and_facing(
    phi: f32, theta: f32, dist: f32, elevation: f32,
    resolution: glam::Vec2, extent: glam::Vec2
) -> (MatrixData, bool) {
    // Generic orbital camera setup, centered at (0.0, 0.0, elevation)
    let rotation = glam::Mat4::from_euler(
        glam::EulerRot::ZXY,
        phi.to_radians(),
        theta.to_radians(),
        0.0
    );
    let center = glam::Vec3::new(0.0, 0.0, elevation);

    let cam_pos = rotation * glam::Vec4::new(0.0, -dist, 0.0, 1.0);
    let cam_up  = rotation * glam::Vec4::new(0.0,   0.0, 1.0, 1.0);

    let cam_pos = glam::Vec3::new(cam_pos.x, cam_pos.y, cam_pos.z) + center;
    let cam_up  = glam::Vec3::new(cam_up.x,  cam_up.y,  cam_up.z);

    let scale = {
        let dscales = DrawspaceScales::new(resolution, extent);

        glam::Mat4::from_scale(glam::Vec3 {
            x: dscales.scale[0],
            y: dscales.scale[1],
            z: 1.0
        })
    };

    let proj = glam::Mat4::perspective_rh(0.5_f32.atan() * 2.0, 1.0, 1.0, 200.0);

    let view = glam::Mat4::look_at_rh(
        cam_pos,
        center,
        cam_up
    );

    let mat = MatrixData { matrix: scale * proj * view };
    let day = cam_pos.y < 0.0; // Do we need to render the day scene or the night scene? (true = day)

    (mat, day)
}

struct DynamicResources {
    rtexture_bindgroup: wgpu::BindGroup, // render texture as shader resource (for reading from shader)

    rtexture_color: wgpu::TextureView, // render-texture color target (for writing on as attachment)
    rtexture_depth: wgpu::TextureView, // render-texture depth target
    surface_depth:  wgpu::TextureView  // surface/swapchain depth target
}

const DAY_SKY_COLOR:   wgpu::Color = wgpu::Color { r: 1.0,      g: 0.463917, b: 0.125578, a: 1.0 };
const NIGHT_SKY_COLOR: wgpu::Color = wgpu::Color { r: 0.002352, g: 0.003925, b: 0.021981, a: 1.0 };

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Portals in video games are usually drawn by aligning the secondary camera according to the primary(screen) camera.
/// So that the relative distance and orientation between (primary cam and entry portal) and (secondary cam and leaving portal) are the same.
/// In this implementation, it is quite simpler cause we're using only one transformation matrix and the camera distance and orientation are already synced.
/// 
/// The scene where the leaving portal resides (the "other world") is rendered in a pass.
/// After that, the scene where the entry portal reside (the "current world") is rendered in a subsequent pass.
/// The entry portal's color is sampled from the render texture that was produced in the previous pass.
/// The UV coordinates of it is based on its screen space coordinates.
/// 
/// In this implementation however, no mechanism for teleportation is introduced.
struct Portal {
    textured_pipeline: wgpu::RenderPipeline,
    digits_pipeline:   wgpu::RenderPipeline,
    portal_pipeline:   wgpu::RenderPipeline,

    matrix_bindgroup:    wgpu::BindGroup,
    terrain_bindgroups:  Vec<wgpu::BindGroup>,
    platform_bindgroups: Vec<wgpu::BindGroup>,
    digits_bindgroup:    wgpu::BindGroup,

    dynamic_resources: DynamicResources,

    matrix_ubuffer: wgpu::Buffer,

    terrain_geometry:  PlyGeoBuffers,
    platform_geometry: PlyGeoBuffers,
    sun_geometry:      PlyGeoBuffers,
    moon_geometry:     PlyGeoBuffers,
    digits_geometry:   PlyGeoBuffers,
    portal_geometry:   PlyGeoBuffers,

    angle_phi:     f32,
    angle_theta:   f32,
    distance:      f32,
    elevation:     f32,
    auto_rotation: bool,
    window_size:   (u32, u32)
}

/// called when scene is resized
fn create_dynamic_resources(texsize: (u32, u32), device: &wgpu::Device) -> DynamicResources {
    let fsampler = BasicFilteringSampler::new(device);

    let rtexture_color = RenderTexture::new(
        texsize, SURFACE_FORMAT,
        true, device
    );

    let rtexture_depth = RenderTexture::new(
        texsize, DEPTH_FORMAT,
        false, device
    );

    let surface_depth = RenderTexture::new(
        texsize, DEPTH_FORMAT,
        false, device
    );

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            RenderTexture::default_layout_entry(0),
            BasicFilteringSampler::default_layout_entry(1)
        ]
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label:   None,
        layout:  &bind_group_layout,
        entries: &[
            rtexture_color.get_entry(0),
            fsampler.get_entry(1)
        ]
    });

    DynamicResources {
        rtexture_bindgroup: bind_group,

        rtexture_color: rtexture_color.view,
        rtexture_depth: rtexture_depth.view,
        surface_depth:  surface_depth.view
    }
}

impl ExecDraw for Portal {
    fn setup(
        config:   &wgpu::SurfaceConfiguration,
        _adapter: &wgpu::Adapter,
        device:   &wgpu::Device,
        queue:    &wgpu::Queue
    ) -> Self where Self: Sized {
        // In this implementation, bindgroups are fragmented (1 or 2 resources per bindgroup) to increase flexibility
        let resources = get_resource_folder_for("portal").unwrap();
        
        // load the 3D meshes
        let terrain_geometry  = PlyGeoBuffers::new(device, resources.join("meshes/terrain_geo.ply").as_path().to_str().unwrap());
        let platform_geometry = PlyGeoBuffers::new(device, resources.join("meshes/platform_geo.ply").as_path().to_str().unwrap());

        let sun_geometry  = PlyGeoBuffers::new(device, resources.join("meshes/sun_geo.ply").as_path().to_str().unwrap());
        let moon_geometry = PlyGeoBuffers::new(device, resources.join("meshes/moon_geo.ply").as_path().to_str().unwrap());

        let digits_geometry = PlyGeoBuffers::new(device, resources.join("meshes/digit_geo.ply").as_path().to_str().unwrap());

        let portal_geometry = PlyGeoBuffers::new(device, resources.join("meshes/portal_geo.ply").as_path().to_str().unwrap());

        let fsampler = BasicFilteringSampler::new(device);

        let common_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                ResourceTexture::default_layout_entry(0),
                BasicFilteringSampler::default_layout_entry(1)
            ]
        });

        // load a texture and form a single bindgroup from it
        let texture_to_bindgroup = |path: &str| {
            let texture = ResourceTexture::new(path, device, queue);
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label:   None,
                layout:  &common_bind_group_layout,
                entries: &[ texture.get_entry(0), fsampler.get_entry(1) ]
            })
        };

        // [day, night]
        let terrain_bindgroups: Vec<wgpu::BindGroup> = [
            resources.join("textures/terrain_lightmap_day.png").as_path().to_str().unwrap(),
            resources.join("textures/terrain_lightmap_night.png").as_path().to_str().unwrap()
        ].into_iter().map(texture_to_bindgroup).collect();

        // [day, night]
        let platform_bindgroups: Vec<wgpu::BindGroup> = [
            resources.join("textures/portal_lightmap_day.png").as_path().to_str().unwrap(),
            resources.join("textures/portal_lightmap_night.png").as_path().to_str().unwrap()
        ].into_iter().map(texture_to_bindgroup).collect();

        // digits sprite sheet
        let digits_bindgroup = texture_to_bindgroup(resources.join("textures/beurmon_digits.png").as_path().to_str().unwrap());

        // the transformation matrix
        let (matrix_ubuffer, matrix_bindgroup, matrix_bindgroup_layout) = {
            let umatrix = SingleUniformBuffer::new::<MatrixData>(device, wgpu::ShaderStages::VERTEX_FRAGMENT);

            let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[ SingleUniformBuffer::default_layout_entry(0, &umatrix) ]
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label:   None,
                layout:  &bind_group_layout,
                entries: &[ umatrix.get_entry(0) ]
            });

            (umatrix.buffer, bind_group, bind_group_layout)
        };

        let dynamic_resources = create_dynamic_resources((config.width, config.height), device);

        let depth_stencil_state = wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default()
        };

        // used for the terrain, platform, sun and moon
        let primitive_state_culling = wgpu::PrimitiveState {
            topology:     wgpu::PrimitiveTopology::TriangleList,
            cull_mode:    Some(wgpu::Face::Back),
            polygon_mode: wgpu::PolygonMode::Fill,
            ..Default::default()
        };

        // used for the portal
        let primitive_state_nocull = wgpu::PrimitiveState {
            topology:     wgpu::PrimitiveTopology::TriangleList,
            cull_mode:    None,
            polygon_mode: wgpu::PolygonMode::Fill,
            ..Default::default()
        };

        // { pos: vec3, uv: vec2 }
        let vertex_buffer_layouts = [
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<Vtx3UV>() as wgpu::BufferAddress,
                step_mode:    wgpu::VertexStepMode::Vertex,
                attributes:   &[
                    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, shader_location: 0, offset: 0 },
                    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, shader_location: 1, offset: 3 * std::mem::size_of::<f32>() as u64 }
                ]
            }
        ];

        // same layout for all
        // takes in one texture+sampler pair, and one transformation matrix uniform buffer as bindgroup.
        // Also room for max 8 bytes of push constants
        let primary_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts:   &[ &common_bind_group_layout, &matrix_bindgroup_layout ],
            push_constant_ranges: &[
                wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    range:  0..8
                }
            ]
        });

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

        // Simple pipeline for drawing basic textured meshes (terrain, platform)
        // Supports 180 deg rotation
        let textured_pipeline = {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label:  None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
                    read_to_string(resources.join("shaders/textured.wgsl")).unwrap().as_str()
                ))
            });

            device.create_render_pipeline(&RenderPipelineDescriptor {
                label:  None,
                layout: Some(&primary_pipeline_layout),
                depth_stencil: Some(depth_stencil_state.clone()),
                multisample: wgpu::MultisampleState::default(),
                multiview:   None,
                vertex: wgpu::VertexState {
                    module:      &shader,
                    entry_point: "vs_main",
                    buffers:     &vertex_buffer_layouts
                },
                fragment: Some(wgpu::FragmentState {
                    module:      &shader,
                    entry_point: "fs_main",
                    targets:     &[ Some(SURFACE_FORMAT.into()) ]
                }),
                primitive: primitive_state_culling
            })
        };

        // Draw digits on quad by addressing into the sprite sheet. Multi instance.
        // instance=0 gets drawn normally,
        // instance=1 gets flipped on x-axis
        // Supports UV flipping
        // Supports 180 deg rotation
        let digits_pipeline = {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label:  None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
                    read_to_string(resources.join("shaders/digits.wgsl")).unwrap().as_str()
                ))
            });

            device.create_render_pipeline(&RenderPipelineDescriptor {
                label:  None,
                layout: Some(&primary_pipeline_layout),
                depth_stencil: Some(depth_stencil_state.clone()),
                multisample: wgpu::MultisampleState::default(),
                multiview:   None,
                vertex: wgpu::VertexState {
                    module:      &shader,
                    entry_point: "vs_main",
                    buffers:     &vertex_buffer_layouts
                },
                fragment: Some(wgpu::FragmentState {
                    module:      &shader,
                    entry_point: "fs_main",
                    targets:     &[ Some(color_target_state) ]
                }),
                primitive: primitive_state_nocull
            })
        };

        // Main portal drawing pipeline
        // Supports obtaining UV coordinates from screen-space coordinates
        let portal_pipeline = {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label:  None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
                    read_to_string(resources.join("shaders/portal.wgsl")).unwrap().as_str()
                ))
            });

            device.create_render_pipeline(&RenderPipelineDescriptor {
                label:  None,
                layout: Some(&primary_pipeline_layout),
                depth_stencil: Some(depth_stencil_state.clone()),
                multisample: wgpu::MultisampleState::default(),
                multiview:   None,
                vertex: wgpu::VertexState {
                    module:      &shader,
                    entry_point: "vs_main",
                    buffers:     &vertex_buffer_layouts
                },
                fragment: Some(wgpu::FragmentState {
                    module:      &shader,
                    entry_point: "fs_main",
                    targets:     &[ Some(SURFACE_FORMAT.into()) ]
                }),
                primitive: primitive_state_nocull
            })
        };


        Self {
            textured_pipeline,
            digits_pipeline,
            portal_pipeline,

            matrix_bindgroup,
            terrain_bindgroups,
            platform_bindgroups,
            digits_bindgroup,
            
            dynamic_resources,
            
            matrix_ubuffer,
            
            terrain_geometry,
            platform_geometry,

            sun_geometry,
            moon_geometry,
            digits_geometry,
            portal_geometry,

            angle_phi:     0.0,
            angle_theta:   0.0,
            distance:      70.0,
            elevation:     10.0,
            auto_rotation: true,
            window_size:   (config.width, config.height)
        }
    }

    fn resize(self: &mut Self, width: u32, height: u32, device: &wgpu::Device, _queue: &wgpu::Queue) {
        let dynamic_resources = create_dynamic_resources((width, height), device);
        
        self.dynamic_resources = dynamic_resources;
        self.window_size = (width, height);
    }

    fn onkey(self: &mut Self, event: winit::event::KeyEvent, _device: &wgpu::Device, _queue: &wgpu::Queue) {
        let ELEVATION_SHIFT: f32 = 1.0;
        if event.state == winit::event::ElementState::Pressed {
            match event.key_without_modifiers().as_ref() {
                winit::keyboard::Key::Named(winit::keyboard::NamedKey::ArrowUp) => {
                    self.elevation -= ELEVATION_SHIFT;
                },
                winit::keyboard::Key::Named(winit::keyboard::NamedKey::ArrowDown) => {
                    self.elevation += ELEVATION_SHIFT;
                },
                _ => {}
            }
        }
    }

    fn draw(self: &mut Self, texview: &wgpu::TextureView, device: &wgpu::Device, queue: &wgpu::Queue) {

        fn draw_geometry<'a, 'b>(rpass: &mut wgpu::RenderPass<'a>, geo: &'b PlyGeoBuffers, instances: u32) where 'b: 'a {
            rpass.set_index_buffer(geo.ibuffer.slice(..), wgpu::IndexFormat::Uint16);
            rpass.set_vertex_buffer(0, geo.vbuffer.slice(..));

            rpass.draw_indexed(0..geo.icount as u32, 0, 0..instances);
        }

        // As the transformation matrix updates very frequently (e.g. every frame)
        // The updating of its uniform buffer is moved to the draw function
        let (matdata, facing_day) = calc_matrix_and_facing(
            self.angle_phi, self.angle_theta, self.distance, self.elevation,
            glam::Vec2::new(self.window_size.0 as f32, self.window_size.1 as f32),
            glam::Vec2::new(1.0, 1.0)
        );

        queue.write_buffer(&self.matrix_ubuffer, 0, cast_struct_to_u8_slice(&matdata));

        // Obtained the two digits of current time, packed into a single u32
        // day scene => hour digits
        // night scene => minute digits
        let digits: u32 = {
            let now = Local::now();

            let selected = if facing_day { now.hour() } else { now.minute() };

            let tens = selected / 10;
            let ones = selected % 10;

            tens << 16 | ones
        };

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // 1st Render pass, draw the terrain+sun/moon+digits, a.k.a. the "other world"
        // For the night scene, the terrain+moon+digits are rotated 180 degs so that we don't need to move the camera or used a 2nd camera
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label:                    None,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.dynamic_resources.rtexture_depth,
                    depth_ops: Some(wgpu::Operations {
                        load:  wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store
                    }),
                    stencil_ops: None
                }),
                timestamp_writes:    None,
                occlusion_query_set: None,
                color_attachments:   &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.dynamic_resources.rtexture_color,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load:  wgpu::LoadOp::Clear(if facing_day { DAY_SKY_COLOR } else { NIGHT_SKY_COLOR }),
                        store: wgpu::StoreOp::Store
                    }
                })]
            });

            // Draw terrain, sun/moon
            rpass.set_pipeline(&self.textured_pipeline);
            rpass.set_bind_group(0, &self.terrain_bindgroups[if facing_day { 0 } else { 1 }], &[]);
            rpass.set_bind_group(1, &self.matrix_bindgroup, &[]);
            rpass.set_push_constants(wgpu::ShaderStages::VERTEX_FRAGMENT, 0, cast_struct_to_u8_slice(&[!facing_day as u32, 0]));
            draw_geometry(&mut rpass, &self.terrain_geometry, 1);
            draw_geometry(&mut rpass, if facing_day { &self.sun_geometry } else { &self.moon_geometry }, 1);

            // Draw the digits
            rpass.set_pipeline(&self.digits_pipeline);
            rpass.set_bind_group(0, &self.digits_bindgroup, &[]);
            rpass.set_bind_group(1, &self.matrix_bindgroup, &[]);
            rpass.set_push_constants(wgpu::ShaderStages::VERTEX_FRAGMENT, 0, cast_struct_to_u8_slice(&[!facing_day as u32, digits]));
            draw_geometry(&mut rpass, &self.digits_geometry, 2);
        }

        // Draw the portal quad and the platform, a.k.a. the "current world"
        // The portal's UV coordinated are obtained from the quad's vertices' screen space coordinates
        // The portal texture is the rendered frame of the "other world" (the render texture of the previous pass)
        // The platform of drawn twice, once with the daytime side lightmap texture,
        // and another time rotated 180 deg with the nighttime side lightmap texture.
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label:                    None,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.dynamic_resources.surface_depth,
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

            // Draw the portal
            rpass.set_pipeline(&self.portal_pipeline);
            rpass.set_bind_group(0, &self.dynamic_resources.rtexture_bindgroup, &[]);
            rpass.set_bind_group(1, &self.matrix_bindgroup, &[]);
            rpass.set_push_constants(wgpu::ShaderStages::VERTEX_FRAGMENT, 0, cast_struct_to_u8_slice(&[!facing_day as u32, 0]));
            draw_geometry(&mut rpass, &self.portal_geometry, 1);

            rpass.set_pipeline(&self.textured_pipeline);
            rpass.set_bind_group(1, &self.matrix_bindgroup, &[]);

            // Draw the daytime side platform
            rpass.set_bind_group(0, &self.platform_bindgroups[0], &[]);
            rpass.set_push_constants(wgpu::ShaderStages::VERTEX_FRAGMENT, 0, cast_struct_to_u8_slice(&[0_u32, 0]));
            draw_geometry(&mut rpass, &self.platform_geometry, 1);

            // Draw the nighttime side platform
            rpass.set_bind_group(0, &self.platform_bindgroups[1], &[]);
            rpass.set_push_constants(wgpu::ShaderStages::VERTEX_FRAGMENT, 0, cast_struct_to_u8_slice(&[1_u32, 0]));
            draw_geometry(&mut rpass, &self.platform_geometry, 1);
        }

        queue.submit(std::iter::once(encoder.finish()));

        if self.auto_rotation {
            self.angle_phi = (self.angle_phi + 1.0) % 360.0;
        }
    }

    fn onmousemove(self: &mut Self, delta: (f64, f64), state: u32, _device: &wgpu::Device, _queue: &wgpu::Queue) {
        let THETA_SHIFT: f32 = 0.5;
        let PHI_SHIFT:   f32 = 0.5;

        let dx = -delta.0 as f32;
        let dy = -delta.1 as f32;

        if state & 1 << 2 != 0 {
            self.angle_theta   = (self.angle_theta + dy * THETA_SHIFT).clamp(-90.0, 90.0);
            self.angle_phi     = self.angle_phi    + dx * PHI_SHIFT;
            self.auto_rotation = false;
        }
    }

    fn onmousescroll(self: &mut Self, delta: (f64, f64), _state: u32, _device: &wgpu::Device, _queue: &wgpu::Queue) {
        let DIST_SHIFT: f32 = 3.0;

        let dy = -delta.1 as f32;

        self.distance = (self.distance + dy * DIST_SHIFT).clamp(0.0, 1000.0);
    }

    fn onmousebutton(self: &mut Self, state: u32, _device: &wgpu::Device, _queue: &wgpu::Queue) {
        if state & 1 != 0 {
            self.auto_rotation = !self.auto_rotation;
        }
    }
}

fn main() {
    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    #[allow(unused_mut)]
    let mut builder = winit::window::WindowBuilder::new();
    let window = builder
        .with_inner_size(winit::dpi::LogicalSize { width: 512.0, height: 512.0 })
        .with_title("Portal Clock")
        .build(&event_loop)
        .unwrap();

    pollster::block_on(run::<Portal>(
        event_loop, window,
        Some(wgpu::Features::PUSH_CONSTANTS))
    );
}