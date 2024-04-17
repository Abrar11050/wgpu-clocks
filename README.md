# WGPU Clocks

These are my implementation of various clock faces, mostly made as a part of my w(eb)gpu grinding, and because I just like making clocks in general. Rather than relying on 3rd party drawing libraries/rendering frameworks, these clocks are constituted from bare wgpu triangles, shaders, and textures; orchestrating render passes manually. WGPU is chosen instead of lower level APIs like Vulkan for two reasons: portability, and sweeping away the pain of writing truckloads of boilerplate and memory and sync management, while being spiritually similar to Vulkan. Don't consider them as fully optimized clock widgets that you might want to run them on your smartwatch (albeit possible I presume), these more of serve as demos for wgpu.

*Note: These demos makes extensive use of the push constant feature ``(wgpu::Features::PUSH_CONSTANTS)``, which isn't currently available in WebGPU, or older backends like OpenGL*

The ``supplementary`` directory contains materials that might serve as aids for understanding the implementation mechanisms. Not all of the demos have supplementary materials. The Rust source files and WGSL source files are also commented.

The ``texture`` directories may have an ``all-res`` sub-directory containing various resolutions of each texture. These are however, not loaded by code. Rather they can be used to swap out the loadable texture (placed in the ``texture`` directory) for a different resolution version.

I may (or may not) extend this collection in the future.

## [2D] Seven-Segment Digital Clock

Generic 7-seg clock with switchable color/pattern platte. Use <kbd>Space</kbd> key top iterate through them. Press <kbd>T</kbd> key to switch between 24hr/12hr. Uses dual-pass gaussian blur filter for the glow effect.

./videos/digital.mp4

## [2D] Polar Clock

Angle based time representation using rings/arcs and disks/circles. Smoothstep based anti-aliasing. Press <kbd>Space</kbd> key to go though the color palette.

./videos/polar.mp4

## [3D] Mechanical Counter Clock

Digits placed on rotatable wheels. Makes use of instanced geometry. Font(s) used (bitmap sprite): **Haettenschweiler**.

./videos/counter.mp4

## [3D] Portal Clock

Multi-scene portal rendering (Two pass, screen-space UV mapping). 3D meshes loaded from disk. Day side shows hours and night side shows minutes. Use mouse to operate camera, controls are similar to that of [Blender](https://www.blender.org/) (hold down middle button and move to pan, scroll to zoom), left-click to toggle auto-rotation. Font(s) used (bitmap sprite): **Beurmon**, 3D meshes and textures made using [Blender](https://www.blender.org/).

./videos/portal.mp4