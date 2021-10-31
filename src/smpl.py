import pickle

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from . import math
from . import skinning


class SMPL(keras.layers.Layer):
    def __init__(self, model_path, name="smpl", **kwargs):
        super(SMPL, self).__init__(name, **kwargs)

        with open(model_path, 'rb') as f:
            dd = pickle.load(f, encoding='latin1')

        self.num_shapes = dd['shapedirs'].shape[-1]
        self.num_vertices = dd["v_template"].shape[-2]
        self.num_faces = dd["f"].shape[-2]
        self.num_joints = dd["J_regressor"].shape[0]

        self.skinning_weights = tf.convert_to_tensor(
            value=dd["weights"],
            dtype=self.dtype,
            name="skinning_weights"
        )

        self.template_vertices = tf.convert_to_tensor(
            value=dd["v_template"],
            dtype=self.dtype,
            name="template_vertices"
        )

        self.faces = tf.convert_to_tensor(
            value=dd["f"],
            dtype=tf.int32,
            name="faces"
        )

        self.shapedirs = tf.convert_to_tensor(
            value=dd["shapedirs"].reshape([-1, self.num_shapes]).T,
            dtype=self.dtype,
            name="shapedirs"
        )

        self.posedirs = tf.convert_to_tensor(
            value=dd["posedirs"].reshape([-1, dd['posedirs'].shape[-1]]).T,
            dtype=self.dtype,
            name="posedirs"
        )

        self.joint_regressor = tf.convert_to_tensor(
            value=dd["J_regressor"].T.todense(),
            dtype=self.dtype,
            name="joint_regressor"
        )

        self.kintree_table = dd['kintree_table'][0].astype(np.int32)


    def call(self, shape=None, pose=None, translation=None):
        # Add shape blenshape
        shape_blendshape = tf.reshape(
            tensor=tf.matmul(shape, self.shapedirs),
            shape=[-1, self.num_vertices, 3],
            name="shape_blendshape"
        )

        vs = self.template_vertices + shape_blendshape  

        if pose is None:
            return vs, tf.zeros((self.num_joints, 4, 4))

        # Compute local joint locations and rotations
        pose = tf.reshape(pose, [-1, self.num_joints, 3])
        joint_rotations_local = math.AxisAngleToMatrix()(pose)
        joint_locations_local = tf.stack(
            values=[
                tf.matmul(vs[:, :, 0], self.joint_regressor),
                tf.matmul(vs[:, :, 1], self.joint_regressor),
                tf.matmul(vs[:, :, 2], self.joint_regressor)
            ],
            axis=2,
            name="joint_locations_local"
        )

        # Add pose blenshape
        pose_feature = tf.reshape(
            tensor=joint_rotations_local[:, 1:, :, :] - tf.eye(3),
            shape=[-1, 9 * (self.num_joints - 1)]
        )

        pose_blendshape = tf.reshape(
            tensor=tf.matmul(pose_feature, self.posedirs),
            shape=[-1, self.num_vertices, 3],
            name="pose_blendshape"
        )

        vp = vs + pose_blendshape

        # Compute global joint transforms
        joint_transforms, joint_locations = skinning.PoseSkeleton()(
            joint_rotations_local,
            joint_locations_local,
            self.kintree_table
        )

        # Apply linear blend skinning
        v = skinning.LBS()(vp, joint_transforms, self.skinning_weights)

        # Apply translation
        if translation is not None:
            v += translation[:, tf.newaxis, :]

        tensor_dict = {
            "shape_blendshape": shape_blendshape,
            "pose_blendshape": pose_blendshape,
            "pose_feature": pose_feature,
            "joint_transforms": joint_transforms,
            "joint_locations": joint_locations,
            "joint_locations_local": joint_locations_local,
            "vertices_shaped": vs,
            "vertices_posed": vp
        }

        return v, tensor_dict