use wgpu::util::DeviceExt;
use image::{io::Reader as ImageReader, EncodableLayout};
use std::path::PathBuf;

pub const SURFACE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;
pub trait ExecDraw {
    fn setup(
        config:  &wgpu::SurfaceConfiguration,
        adapter: &wgpu::Adapter,
        device:  &wgpu::Device,
        queue:   &wgpu::Queue) -> Self where Self: Sized;

    fn resize(self: &mut Self, width: u32, height: u32, device: &wgpu::Device, queue: &wgpu::Queue);

    fn draw(self: &mut Self, texview: &wgpu::TextureView, device: &wgpu::Device, queue: &wgpu::Queue);

    fn onkey(self: &mut Self, _event: winit::event::KeyEvent, _device: &wgpu::Device, _queue: &wgpu::Queue) {}

    fn onmousemove(self: &mut Self, _delta: (f64, f64), _state: u32, _device: &wgpu::Device, _queue: &wgpu::Queue) {}

    fn onmousescroll(self: &mut Self, _delta: (f64, f64), _state: u32, _device: &wgpu::Device, _queue: &wgpu::Queue) {}

    fn onmousebutton(self: &mut Self, _state: u32, _device: &wgpu::Device, _queue: &wgpu::Queue) {}
}

/// App runner.
/// Modified version of WGPU sample boilerplate.
/// Takes in an `ExecDraw` derived struct and calls necessary functions
pub async fn run<T: ExecDraw>(
    event_loop: winit::event_loop::EventLoop<()>,
    window:     winit::window::Window,
    features:   Option<wgpu::Features>
) {
    let mut size = window.inner_size();
    size.width   = size.width.max(1);
    size.height  = size.height.max(1);

    let instance = wgpu::Instance::default();

    let surface = unsafe { instance.create_surface(&window).unwrap() };

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference:       wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface:     Some(&surface)
        })
        .await
        .expect("Failed to find an appropriate adapter");

    let swapchain_capabilities = surface.get_capabilities(&adapter);
    if !swapchain_capabilities.formats.into_iter().any(|format| { format == SURFACE_FORMAT }) {
        panic!("Seeking for support of surface format \"wgpu::TextureFormat::Bgra8UnormSrgb\", but not found");
    }
    
    let swapchain_format = SURFACE_FORMAT;

    let mut device_limits = wgpu::Limits::downlevel_webgl2_defaults().using_resolution(adapter.limits());
    device_limits.max_push_constant_size = 64;  // Needed for push constants
    device_limits.max_storage_buffers_per_shader_stage = 8; // Needed for storage buffers
    device_limits.max_storage_buffer_binding_size = 64 * 1024;  // Needed for storage buffers

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: match features {
                    None => wgpu::Features::empty(),
                    Some(f) => f
                },
                limits: device_limits
            },
            None,
        )
        .await
        .expect("Failed to create device");

    let mut config = wgpu::SurfaceConfiguration {
        usage:        wgpu::TextureUsages::RENDER_ATTACHMENT,
        format:       swapchain_format,
        width:        size.width,
        height:       size.height,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode:   wgpu::CompositeAlphaMode::Auto,
        view_formats: vec![swapchain_format]
    };

    surface.configure(&device, &config);

    let mut execdraw = T::setup(&config, &adapter, &device, &queue);

    let mut cursor_in_window = false;
    let mut mouse_button_state = 0_u32;

    let _ = event_loop.run(move |event, target| {
        // Have the closure take ownership of the resources.
        // `event_loop.run` never returns, therefore we must do this to ensure
        // the resources are properly cleaned up.
        let _ = (&instance, &adapter, &execdraw);

        if let winit::event::Event::WindowEvent { window_id: _, event, } = event {
            match event {
                winit::event::WindowEvent::Resized(new_size) => {
                    // Reconfigure the surface with the new size
                    config.width = new_size.width.max(1);
                    config.height = new_size.height.max(1);
                    surface.configure(&device, &config);
                    // On macos the window needs to be redrawn manually after resizing
                    execdraw.resize(new_size.width.max(1), new_size.height.max(1), &device, &queue);
                    window.request_redraw();
                },
                winit::event::WindowEvent::CloseRequested => target.exit(),
                winit::event::WindowEvent::RedrawRequested => {
                    let frame = surface.get_current_texture().expect("Failed to acquire next swap chain texture");
                    let view  = frame.texture.create_view(&wgpu::TextureViewDescriptor {
                        format: Some(swapchain_format),
                        ..wgpu::TextureViewDescriptor::default()
                    });

                    execdraw.draw(&view, &device, &queue);
                    frame.present();
                    window.request_redraw();
                },
                winit::event::WindowEvent::KeyboardInput { event, .. } => {
                    execdraw.onkey(event, &device, &queue);
                },
                winit::event::WindowEvent::CursorLeft { .. } => {
                    cursor_in_window = false;
                },
                winit::event::WindowEvent::CursorEntered { .. } => {
                    cursor_in_window = true;
                }
                _ => {}
            }
        } else if let winit::event::Event::DeviceEvent { device_id: _, event } = event {
            match event {
                winit::event::DeviceEvent::MouseMotion { delta } => {
                    if cursor_in_window {
                        execdraw.onmousemove(delta, mouse_button_state, &device, &queue);
                    }
                },
                winit::event::DeviceEvent::MouseWheel { delta } => {
                    if cursor_in_window {
                        if let winit::event::MouseScrollDelta::LineDelta(dx, dy) = delta {
                            let delta = (dx as f64, dy as f64);
                            execdraw.onmousescroll(delta, mouse_button_state, &device, &queue);
                        }
                    }
                },
                winit::event::DeviceEvent::Button { button, state } => {
                    match state {
                        winit::event::ElementState::Pressed => {
                            mouse_button_state |= 1 << button;
                        },
                        winit::event::ElementState::Released => {
                            mouse_button_state &= !(1 << button);
                        },
                    }
                    
                    if cursor_in_window {
                        execdraw.onmousebutton(mouse_button_state, &device, &queue);
                    }
                }
                _ => {}
            }
        }
    });
}

pub fn load_png_rgba8(path: &str) -> (u32, u32, Vec<u8>) {
    let dynimage = ImageReader::open(path).unwrap().decode().unwrap();
    let rgba8 = dynimage.to_rgba8();
    let raw = rgba8.as_raw();

    (rgba8.width(), rgba8.height(), raw.clone())
}

/// Basic read-only texture resource made from pixel data
pub struct ResourceTexture {
    pub texture: wgpu::Texture,
    pub view:    wgpu::TextureView,
    pub width:   u32,
    pub height:  u32
}

impl ResourceTexture {
    pub fn new(path: &str, device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let (width, height, data) = load_png_rgba8(path);

        let texture = device.create_texture_with_data(queue, &wgpu::TextureDescriptor {
            label:           None,
            size:            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count:    1,
            dimension:       wgpu::TextureDimension::D2,
            format:          wgpu::TextureFormat::Rgba8Unorm,
            usage:           wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats:    &[]
        }, data.as_bytes());

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        Self { texture, view, width, height }
    }

    pub fn get_entry(self: &Self, binding: u32) -> wgpu::BindGroupEntry {
        wgpu::BindGroupEntry {
            binding,
            resource: wgpu::BindingResource::TextureView(&self.view)
        }
    }

    pub fn default_layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                multisampled:   false,
                sample_type:    wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2
            },
            count: None
        }
    }
}

/// Basic Linear filtering sampler with edge clipping
pub struct BasicFilteringSampler {
    pub sampler: wgpu::Sampler
}

impl BasicFilteringSampler {
    pub fn new(device: &wgpu::Device) -> Self {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: None,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter:     wgpu::FilterMode::Linear,
            min_filter:     wgpu::FilterMode::Linear,
            mipmap_filter:  wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Self { sampler }
    }

    pub fn get_entry(self: &Self, binding: u32) -> wgpu::BindGroupEntry {
        wgpu::BindGroupEntry {
            binding,
            resource: wgpu::BindingResource::Sampler(&self.sampler)
        }
    }

    pub fn default_layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None
        }
    }
}

/// Rewritable uniform buffer for a single struct/variable
pub struct SingleUniformBuffer {
    pub buffer: wgpu::Buffer,
    pub stages: wgpu::ShaderStages
}

impl SingleUniformBuffer {
    pub fn new<T>(device: &wgpu::Device, stages: wgpu::ShaderStages) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size:  std::mem::size_of::<T>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });

        Self { buffer, stages }
    }

    pub fn get_entry(self: &Self, binding: u32) -> wgpu::BindGroupEntry {
        wgpu::BindGroupEntry {
            binding,
            resource: self.buffer.as_entire_binding()
        }
    }

    pub fn default_layout_entry(binding: u32, sub: &Self) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: sub.stages,
            ty: wgpu::BindingType::Buffer {
                ty:                 wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size:   None
            },
            count: None
        }
    }
}

/// Read only storage buffer for array data
pub struct ImmutableStorageBuffer {
    pub buffer: wgpu::Buffer,
    pub stages: wgpu::ShaderStages
}

impl ImmutableStorageBuffer {
    pub fn new(device: &wgpu::Device, stages: wgpu::ShaderStages, init: &[u8]) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    None,
            contents: init,
            usage:    wgpu::BufferUsages::STORAGE
        });

        Self { buffer, stages }
    }

    pub fn get_entry(self: &Self, binding: u32) -> wgpu::BindGroupEntry {
        wgpu::BindGroupEntry {
            binding,
            resource: self.buffer.as_entire_binding()
        }
    }

    pub fn default_layout_entry(binding: u32, sub: &Self) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: sub.stages,
            ty: wgpu::BindingType::Buffer {
                ty:                 wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size:   None
            },
            count: None
        }
    }
}

/// Texture that can be rendered on in a pass and sampled from in a subsequent pass
/// Usable for both color or depth targets
/// Single sample
pub struct RenderTexture {
    pub texture: wgpu::Texture,
    pub view:    wgpu::TextureView,
    pub format:  wgpu::TextureFormat,
    pub width:   u32,
    pub height:  u32
}

impl RenderTexture {
    pub fn new(
        size: (u32, u32), format: wgpu::TextureFormat,
        bindable: bool, device: &wgpu::Device
    ) -> Self {
        let (width, height) = size;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label:           None,
            size:            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count:    1,
            dimension:       wgpu::TextureDimension::D2,
            view_formats:    &[],
            usage:           match bindable {
                false => wgpu::TextureUsages::RENDER_ATTACHMENT, // usually depth only targets
                true  => wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING // usually color targets
            },
            format
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        Self { texture, view, format, width, height }
    }

    pub fn get_layout_entry(self: &Self, binding: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                multisampled:   false,
                sample_type:    wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2
            },
            count: None
        }
    }

    pub fn get_entry(self: &Self, binding: u32) -> wgpu::BindGroupEntry {
        wgpu::BindGroupEntry {
            binding,
            resource: wgpu::BindingResource::TextureView(&self.view)
        }
    }

    pub fn default_layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                multisampled:   false,
                sample_type:    wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2
            },
            count: None
        }
    }
}

/// Collection of data that can be used for adapting with various window size and aspect ratio
/// The WGPU shader coordinate system is [-1..1] in both axes, with origin (0, 0) in the middle.
/// The system stretches/compresses as window gets resized, but we need uniform scaling for both axes.
/// Also, the examples use units that can be out of the [-1..1] range.
/// 
/// As for such, introducing adaptive scaling system for x and y axis.
/// There is a 2D measurement called `extent`, which is a rectangle,
/// whose `width = 2 * extent.x` and `height = 2 * extent.y`.
/// Thus the range is `[-extent.x..extent.x]` (horizontally) and `[-extent.y..extent.y]` (vertically)
/// The center of the rectangle is the origin. Any point described within this extent is guaranteed to be on the screen/window.
/// Though being 2D, it can also be applied to 3D transformation matrices as well.
#[repr(C, align(8))]
pub struct DrawspaceScales {
    /// the scale that'll be applied to the vertices
    pub scale: glam::Vec2,
    /// the supplied extent
    pub extent: glam::Vec2,
    /// the supplied screen resolution
    pub resolution: glam::Vec2,
    /// amount of pixels per drawing unit, changes according to screen size
    pub density: f32
}

impl DrawspaceScales {
    pub fn new(resolution: glam::Vec2, extent: glam::Vec2) -> Self {
        let [width, height] = resolution.to_array();
        let [ext_x, ext_y]  = extent.to_array();

        let aspect_ratio_window = width / height;
        let aspect_ratio_extent = ext_x / ext_y;

        if aspect_ratio_window > aspect_ratio_extent {
            // the window's ceiling+floor touche the extent's ceiling+floor
            let scale = glam::Vec2::new(aspect_ratio_window * ext_y, ext_y).recip();
            let density = (height * 0.5) / ext_y;
            Self { scale, extent, resolution, density }
        } else {
            // the window's side walls touche the extent's side walls
            let scale = glam::Vec2::new(ext_x, (1.0 / aspect_ratio_window) * ext_x).recip();
            let density = (width * 0.5) / ext_x;
            Self { scale, extent, resolution, density }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Vtx2ID {
    pub pos: glam::Vec2,
    pub id:  u32
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Vtx3UV {
    pub pos: glam::Vec3,
    pub uv:  glam::Vec2
}

#[derive(Debug)]
pub struct PlyMesh {
    pub vertices: Vec<Vtx3UV>,
    pub indices:  Vec<u16>
}

impl PlyMesh {
    pub fn new(path: &str) -> Result<Self, &str> {
        use ply_rs::ply::Property::{Float, ListUInt};

        let mut file = std::fs::File::open(path).unwrap();
        let parser = ply_rs::parser::Parser::<ply_rs::ply::DefaultElement>::new();
        let ply = parser.read_ply(&mut file);

        assert!(ply.is_ok());
        let ply = ply.unwrap();

        let vertex_count = ply.header.elements["vertex"].count;
        let face_count   = ply.header.elements["face"].count;

        let mut vertices: Vec<Vtx3UV> = Vec::with_capacity(vertex_count);
        let mut indices:  Vec<u16> = Vec::with_capacity(face_count * 3);

        let vertex_payload = &ply.payload["vertex"];
        let face_payload   = &ply.payload["face"];


        for item in vertex_payload {
            let collect_f32 = |key| {
                match item[key] {
                    Float(val) => val,
                    _ => std::f32::NAN
                }
            };

            let pos = ["x", "y", "z"].map(collect_f32);
            let uv  = ["s", "t"].map(collect_f32);

            if pos.into_iter().any(|v| v.is_nan()) || uv.into_iter().any(|v| v.is_nan()) {
                return Err("Illegal data type in vertex, expected float");
            }

            vertices.push(Vtx3UV {
                pos: glam::Vec3::from_array(pos),
                uv:  glam::Vec2::from_array(uv)
            });
        }

        for item in face_payload {
            match &item["vertex_indices"] {
                ListUInt(facedata) => {
                    if facedata.len() != 3 {
                        return Err("Illegal index count in face, expected 3");
                    }

                    indices.push(facedata[0] as u16);
                    indices.push(facedata[1] as u16);
                    indices.push(facedata[2] as u16);
                },
                _ => {
                    return Err("Illegal data type in face, expected uint");
                }
            };
        }

        Ok(Self { vertices, indices })
    }
}

#[allow(dead_code)]
pub struct PlyGeoBuffers {
    pub vbuffer: wgpu::Buffer,
    pub ibuffer: wgpu::Buffer,
    pub vcount:  usize,
    pub icount:  usize
}

impl PlyGeoBuffers {
    pub fn new(device: &wgpu::Device, path: &str) -> Self {
        let mesh = PlyMesh::new(path).unwrap();

        let (vbuffer, ibuffer) = create_vertex_and_index_buffers(
            device,
            cast_slice_to_u8_slice(mesh.vertices.as_slice()),
            cast_slice_to_u8_slice(mesh.indices.as_slice())
        );

        Self {
            vbuffer, ibuffer,
            vcount: mesh.vertices.len(),
            icount: mesh.indices.len()
        }
    }
}

pub fn cast_struct_to_u8_slice<T>(data: &T) -> &[u8] {
    let len = std::mem::size_of::<T>();

    unsafe {
        let ptr = data as *const T as *const u8;
        std::slice::from_raw_parts(ptr, len)
    }
}

pub fn cast_slice_to_u8_slice<T>(data: &[T]) -> &[u8] {
    let len = std::mem::size_of::<T>() * data.len();

    unsafe {
        let ptr = data.as_ptr() as *const u8;
        std::slice::from_raw_parts(ptr, len)
    }
}

pub fn create_vertex_and_index_buffers(device: &wgpu::Device, vdata: &[u8], idata: &[u8]) -> (wgpu::Buffer, wgpu::Buffer) {
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    None,
        contents: vdata,
        usage:    wgpu::BufferUsages::VERTEX
    });

    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    None,
        contents: idata,
        usage:    wgpu::BufferUsages::INDEX
    });

    (vertex_buffer, index_buffer)
}

pub fn get_resource_folder_for(sub_folder: &str) -> std::io::Result<PathBuf> {
    use std::io::{Error, ErrorKind};
    #[allow(non_snake_case)]
    let RESOURCE_FOLDER = "resources";

    let mut folder = std::env::current_dir()?;
    folder.push(RESOURCE_FOLDER);
    folder.push(sub_folder);

    if folder.exists() && folder.is_dir() {
        return Ok(folder);
    }

    let mut folder = std::env::current_exe()?;
    folder.pop();
    folder.push(RESOURCE_FOLDER);
    folder.push(sub_folder);

    if folder.exists() && folder.is_dir() {
        return Ok(folder);
    }
    
    return Err(Error::new(
        ErrorKind::NotFound,
        format!("The specified resource folder {}/{} wasn't found", RESOURCE_FOLDER, sub_folder))
    );
}

pub const fn rgba32(r: u8, g: u8, b: u8, a: u8) -> u32 {
    let mut col = a as u32;
    col |= (b as u32) <<  8;
    col |= (g as u32) << 16;
    col |= (r as u32) << 24;
    return col;
}

pub fn lerp_u32_color(c0: u32, c1: u32, t: f64) -> u32 {
    let c0_r = ((c0 >> 24) & 0xFF) as f64;
    let c0_g = ((c0 >> 16) & 0xFF) as f64;
    let c0_b = ((c0 >>  8) & 0xFF) as f64;
    let c0_a = ((c0 >>  0) & 0xFF) as f64;

    let c1_r = ((c1 >> 24) & 0xFF) as f64;
    let c1_g = ((c1 >> 16) & 0xFF) as f64;
    let c1_b = ((c1 >>  8) & 0xFF) as f64;
    let c1_a = ((c1 >>  0) & 0xFF) as f64;

    let r = ((1.0 - t) * c0_r + t * c1_r) as u32;
    let g = ((1.0 - t) * c0_g + t * c1_g) as u32;
    let b = ((1.0 - t) * c0_b + t * c1_b) as u32;
    let a = ((1.0 - t) * c0_a + t * c1_a) as u32;

    return (r << 24) | (g << 16) | (b << 8) | a;
}

pub fn u32_col_to_wgpu_col(col: u32) -> wgpu::Color {
    let col_r = ((col >> 24) & 0xFF) as f64;
    let col_g = ((col >> 16) & 0xFF) as f64;
    let col_b = ((col >>  8) & 0xFF) as f64;
    let col_a = ((col >>  0) & 0xFF) as f64;

    let scale = 1.0 / 255.0;

    return wgpu::Color {
        r: col_r * scale,
        g: col_g * scale,
        b: col_b * scale,
        a: col_a * scale
    };
}