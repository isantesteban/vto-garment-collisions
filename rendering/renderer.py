import bpy
import os

import numpy as np


class GarmentRenderer:

    def __init__(self, cloth_paths, body_paths, **config):
        self.cloth_paths = cloth_paths
        self.body_paths = body_paths

        # Blender scene
        self.camera = bpy.context.scene.camera
        self.camera_tracking = config.pop("camera_tracking", True)

        self.render_settings = bpy.context.scene.render
        self.render_body = config.pop("render_body", True)
        self.render_cloth = config.pop("render_cloth", True)
        self.follow_body = config.pop("follow_body", False)

        self.cloth_material = config.pop("cloth_material", "ClothMaterialPurple")
        self.body_material = config.pop("body_material", "BodyMaterial")
        self.rotation = config.pop("rotation", 0.0)
        self.translation = config.pop("translation", [0.0, 0.0, 0.0])

        # Export configuration
        self.export_path = config.pop("export_path", "tmp/render")
        self.suffix = config.pop("suffix", "")
        self.prefix = config.pop("prefix", "")
        self.digits = config.pop("digits", 5)
        self.extension = ".jpg"

        for key in config.keys():
            raise ValueError("Unexpected argument '" + key + "'")

        self.initialized = False
        self.initial_position = None
        

    def initialize(self):
        if self.initialized:
            return

        self.frames = max(len(self.cloth_paths), len(self.body_paths))

        # Create the export directory if does not exist
        if not os.path.exists(self.export_path):
            os.makedirs(self.export_path)

        self.initialized = True


    def disable_tracking(self):
        self.camera.constraints["Copy Location"].target = None
        self.camera.constraints["Track To"].target = None            
        self.camera_tracking = False


    def track_object(self, mesh):
        if not self.camera_tracking: 
            return

        self.camera.constraints["Copy Location"].target = mesh
        self.camera.constraints["Track To"].target = mesh         
  

    def get_body(self, frame):
        return self.load_obj(self.body_paths[frame], self.body_material)


    def get_cloth(self, frame):
        cloth = self.load_obj(self.cloth_paths[frame], self.cloth_material)
        # self.add_wireframe(cloth, thickness=0.0005)
        return cloth


    def add_wireframe(self, obj, thickness = 0.001):
        obj.data.materials.append(bpy.data.materials["WireframeMaterial"])
        obj.modifiers.new(type='WIREFRAME', name="Wireframe")
        obj.modifiers["Wireframe"].thickness = thickness
        obj.modifiers["Wireframe"].material_offset = 1
        obj.modifiers["Wireframe"].use_replace = False


    def load_obj(self, path, material):
        bpy.ops.import_scene.obj(filepath=path, split_mode="OFF")
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')
        bpy.ops.object.shade_smooth()

        mesh = bpy.context.selected_objects[0]
        mesh.active_material = bpy.data.materials[material]

        return mesh


    def delete_object(self, mesh):
        if mesh == None: 
            return

        bpy.ops.object.select_all(action='DESELECT')
        mesh.select_set(state=True)
        bpy.ops.object.delete()


    def render_path(self, frame):
        file_name = self.prefix + str(frame).zfill(self.digits) + self.suffix +  self.extension
        return os.path.join(self.export_path, file_name)


    def render(self, resolution_x = 1920, resolution_y = 1080, fov = 45, resolution_percentage = 100, start_frame = 0, end_frame = None):
        self.initialize()
     
        # Set render settings
        self.render_settings.resolution_x = resolution_x
        self.render_settings.resolution_y = resolution_y
        self.render_settings.resolution_percentage = resolution_percentage
        self.camera.data.sensor_width = max(fov, 25)
  
        body_mesh = None
        cloth_mesh = None
    
        # Start rendering
        for frame in range(self.frames):

            if frame < start_frame:
                continue

            if end_frame is not None and frame > end_frame:
                break

            print("\n[ INFO ] Rendering frame %d" % frame)

            # Get the objects of the current frame 
            if self.render_body and frame < len(self.body_paths):
                self.delete_object(body_mesh)
                body_mesh = self.get_body(frame)

            if frame < len(self.cloth_paths):
                self.delete_object(cloth_mesh)
                cloth_mesh = self.get_cloth(frame)

            # Make the camera look at the cloth (or the body if there is no cloth)
            if cloth_mesh != None and not self.follow_body:
                self.track_object(cloth_mesh)

            if body_mesh != None and self.follow_body:
                self.track_object(body_mesh)    
            
            # Rotate the cloth
            cloth_mesh.parent = body_mesh
            cloth_mesh.matrix_parent_inverse = body_mesh.matrix_world.inverted()
            body_mesh.rotation_euler[2] = self.rotation
       
            # Render the frame
            self.render_settings.filepath = self.render_path(frame)
            bpy.ops.render.render(write_still = True) 

        self.delete_object(body_mesh)
        self.delete_object(cloth_mesh)


    def generate_video(self, fps = 30, crf = 20, name="output"):
        file_pattern = self.prefix + "%0" + str(self.digits) + "d" + self.suffix + self.extension

        command = "ffmpeg"
        command += " -i " + os.path.join(self.export_path, file_pattern)
        command += " -c:v libx264 -profile:v high -pix_fmt yuv420p"
        command += " -framerate " + str(fps)
        command += " -crf " + str(crf)
        command += " -y " + os.path.join(self.export_path, name + ".mp4")

        os.system(command)

    # Convert jpg sequence to video
    # ffmpeg -framerate 30 -i %05d.jpg -c:v libx264 -profile:v high -crf 10 -pix_fmt yuv420p output.mp4

def read_obj(file_name, read_uvs=False):
    # Counts number of vertices and faces
    vertices = []
    faces = []
    uvs = []
    faces_uv = []

    with open(file_name, 'r') as fp:
        for line in fp:
            line_split = line.split()
            
            if not line_split:
                continue

            if read_uvs and line_split[0] == 'vt':
                uvs.append([line_split[1], line_split[2]])

            if line_split[0] == 'v':
                vertices.append([line_split[1], line_split[2], line_split[3]])

            if line_split[0] == 'f':
                vertex_indices = [s.split("/")[0] for s in line_split[1:]]
                faces.append(vertex_indices)

                if read_uvs:
                    uv_indices = [s.split("/")[1] for s in line_split[1:]]
                    faces_uv.append(uv_indices)

    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32) - 1

    if read_uvs:
        uvs = np.array(uvs, dtype=np.float32)
        faces_uv = np.array(faces_uv, dtype=np.int32) - 1
        return vertices, faces, uvs, faces_uv

    return vertices, faces