import os

import numpy as np
import tensorflow as tf
from scipy.spatial.transform import Rotation as R


def load_motion(path, separate_arms=True):
    motion_dict = dict(np.load(path))

    # The recurrent regressor is trained with 30fps sequences
    target_fps = 30
    drop_factor = int(motion_dict["mocap_framerate"] // target_fps)

    trans = motion_dict["trans"][::drop_factor]
    poses = motion_dict["poses"][::drop_factor, :72]
    shape = motion_dict["betas"][:10]

    # Separate arms
    if separate_arms:
        angle = 15
        left_arm = 17
        right_arm = 16

        poses = poses.reshape((-1, poses.shape[-1] // 3, 3))
        rot = R.from_euler('z', -angle, degrees=True)
        poses[:, left_arm] = (rot * R.from_rotvec(poses[:, left_arm])).as_rotvec()
        rot = R.from_euler('z', angle, degrees=True)
        poses[:, right_arm] = (rot * R.from_rotvec(poses[:, right_arm])).as_rotvec()

        poses = poses.reshape((poses.shape[0], -1))

    # Swap axes
    rotation = R.from_euler("zx", [-90, 270], degrees=True)
    root_rotation = R.from_rotvec(poses[:, :3])
    poses[:, :3] = (rotation * root_rotation).as_rotvec()
    trans = rotation.apply(trans)

    # Remove hand rotation
    poses[:, 66:] = 0

    # Center model in first frame
    trans = trans - trans[0]

    return {
        "pose": poses.astype(np.float32),
        "shape": shape.astype(np.float32),
        "translation": trans.astype(np.float32),   
    }


def load_obj(filename):
    vertices = []
    faces = []

    with open(filename, 'r') as fp:
        for line in fp:
            line_split = line.split()
            
            if not line_split:
                continue

            elif line_split[0] == 'v':
                vertices.append([line_split[1], line_split[2], line_split[3]])

            elif line_split[0] == 'f':
                vertex_indices = [s.split("/")[0] for s in line_split[1:]]
                faces.append(vertex_indices)

    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32) - 1

    return vertices, faces


def save_obj(filename, vertices, faces):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w') as fp:
        for v in vertices:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

        for f in (faces + 1):  # Faces are 1-based, not 0-based in obj files
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
    
    print("Saved:", filename)
