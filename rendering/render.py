#!/usr/bin/env python
import sys 
import os
import glob

sys.path.append(".")

from rendering.renderer import GarmentRenderer

try:
    index = sys.argv.index("--path")
    path = sys.argv[index + 1]
except ValueError:
    print("Usage: blender --background rendering/scene.blend --python rendering/render.py --path <path_to_meshes>")


renderer = GarmentRenderer(
    cloth_paths=sorted(glob.glob(os.path.join(path, "*garment.obj"))),
    body_paths=sorted(glob.glob(os.path.join(path, "*body.obj"))),
    cloth_material="ClothMaterialYellow",
    body_material="MannequinMaterialDark",
    export_path=os.path.join(path, "render")
)

renderer.render(resolution_percentage=100, fov=50, start_frame=0, end_frame=None)
renderer.generate_video()

