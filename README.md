# WGPU Clocks

These are my implementation of various clock faces, mostly made as a part of my w(eb)gpu grinding, and because I just like making clocks in general. Rather than relying on 3rd party drawing libraries/rendering frameworks, these clocks are constituted from bare wgpu triangles, shaders, and textures; orchestrating render passes manually. WGPU is chosen instead of lower level APIs like Vulkan for two reasons: portability, and sweeping away the pain of writing truckloads of boilerplate and memory and sync management, while being spiritually similar to Vulkan. Don't consider them as fully optimized clock widgets that you might want to run them on your smartwatch (albeit possible I presume), these more of serve as demos for wgpu.

*Note: These demos makes extensive use of the push constant feature ``(wgpu::Features::PUSH_CONSTANTS)``, which isn't currently available in WebGPU, or older backends like OpenGL*

The ``supplementary`` directory contains materials that might serve as aids for understanding the implementation mechanisms. Not all of the demos have supplementary materials. The Rust source files and WGSL source files are also commented.

The ``texture`` directories may have an ``all-res`` sub-directory containing various resolutions of each texture. These are however, not loaded by code. Rather they can be used to swap out the loadable texture (placed in the ``texture`` directory) for a different resolution version.

I may (or may not) extend this collection in the future.

## [2D] Seven-Segment Digital Clock

Generic 7-seg clock with switchable color/pattern platte. Use <kbd>Space</kbd> key top iterate through them. Press <kbd>T</kbd> key to switch between 24hr/12hr. Uses dual-pass gaussian blur filter for the glow effect.

https://github.com/Abrar11050/wgpu-clocks/assets/11440342/23dbb606-64e7-49e8-b6cf-d7211f2a6da1

## [2D] Polar Clock

Angle based time representation using rings/arcs and disks/circles. Smoothstep based anti-aliasing. Press <kbd>Space</kbd> key to go though the color palette.

https://github.com/Abrar11050/wgpu-clocks/assets/11440342/2ae97275-555b-45a9-b099-a85f9b9a62ee

## [3D] Mechanical Counter Clock

Digits placed on rotatable wheels. Makes use of instanced geometry. Font(s) used (bitmap sprite): **Haettenschweiler**.

https://github.com/Abrar11050/wgpu-clocks/assets/11440342/1e3df1c8-6c97-41e5-aae9-cdd26999b4a5

## [3D] Portal Clock

Multi-scene portal rendering (Two pass, screen-space UV mapping). 3D meshes loaded from disk. Day side shows hours and night side shows minutes. Use mouse to operate camera, controls are similar to that of [Blender](https://www.blender.org/) (hold down middle button and move to pan, scroll to zoom), left-click to toggle auto-rotation. Font(s) used (bitmap sprite): **Beurmon**, 3D meshes and textures made using [Blender](https://www.blender.org/).

https://github.com/Abrar11050/wgpu-clocks/assets/11440342/d19195db-2634-4103-92d2-9925358cba4d
