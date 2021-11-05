import tensorflow as tf
import tensorflow.keras as keras


class PoseSkeleton(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(PoseSkeleton, self).__init__(**kwargs)


    def call(self, joint_rotations, joint_positions, parents):
        """
        Computes absolute joint locations given pose.

        Args:
            joint_rotations: batch_size x K x 3 x 3 rotation vector of K joints
            joint_positions: batch_size x K x 3, joint locations before posing
            parents: vector of size K holding the parent id for each joint

        Returns
            joint_transforms: `Tensor`: batch_size x K x 4 x 4 relative joint transformations for LBS.
            joint_positions_posed: batch_size x K x 3, joint locations after posing
        """
        batch_size = tf.shape(joint_rotations)[0]
        num_joints = len(parents)

        def make_affine(rotation, translation, name=None):
            '''
            Args:
                rotation: batch_size x 3 x 3
                translation: batch_size x 3 x 1
            '''
            rotation_homo = tf.pad(rotation, [[0, 0], [0, 1], [0, 0]])
            translation_homo = tf.concat([translation, tf.ones([batch_size, 1, 1])], 1)
            affine_transform = tf.concat([rotation_homo, translation_homo], 2)
            return affine_transform

        joint_positions = tf.expand_dims(joint_positions, axis=-1)      
        root_rotation = joint_rotations[:, 0, :, :]
        root_transform = make_affine(root_rotation, joint_positions[:, 0])

        # Traverse joints to compute global transformations
        transforms = [root_transform]
        for joint, parent in enumerate(parents[1:], start=1):
            position = joint_positions[:, joint] - joint_positions[:, parent]
            transform_local = make_affine(joint_rotations[:, joint], position)
            transform_global = tf.matmul(transforms[parent], transform_local)
            transforms.append(transform_global)
        transforms = tf.stack(transforms, axis=1) 

        # Extract joint positions
        joint_positions_posed = transforms[:, :, :3, 3]

        # Compute affine transforms relative to initial state (i.e., t-pose)
        zeros = tf.zeros([batch_size, num_joints, 1, 1])
        joint_rest_positions = tf.concat([joint_positions, zeros], axis=2)
        init_bone = tf.matmul(transforms, joint_rest_positions)
        init_bone = tf.pad(init_bone, [[0, 0], [0, 0], [0, 0], [3, 0]])
        joint_transforms = transforms - init_bone

        return joint_transforms, joint_positions_posed


class LBS(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LBS, self).__init__(**kwargs)


    def call(self, vertices, joint_rotations, skinning_weights):
        skinning_weights = tf.convert_to_tensor(skinning_weights, self.dtype)

        batch_size = tf.shape(vertices)[0]
        num_joints = skinning_weights.shape[-1]
        num_vertices = vertices.shape[-2]
   
        W = skinning_weights
        if len(skinning_weights.shape.as_list()) < len(vertices.shape.as_list()):
            W = tf.tile(tf.convert_to_tensor(skinning_weights), [batch_size, 1])
            W = tf.reshape(W, [batch_size, skinning_weights.shape[-2], num_joints])    
 
        A = tf.reshape(joint_rotations, (-1, num_joints, 16))
        T = tf.matmul(W, A)
        T = tf.reshape(T, (-1, num_vertices, 4, 4))

        ones = tf.ones([batch_size, num_vertices, 1])
        vertices_homo = tf.concat([vertices, ones], axis=2)
        skinned_homo = tf.matmul(T, tf.expand_dims(vertices_homo, -1))
        skinned_vertices = skinned_homo[:, :, :3, 0]

        return skinned_vertices
