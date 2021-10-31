import numpy as np
import tensorflow as tf 
import tensorflow.keras as keras


class AxisAngleToMatrix(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AxisAngleToMatrix, self).__init__(**kwargs)


    def call(self, axis_angle):
        """Converts rotations in axis-angle representation to rotation matrices

        Args:
            axis_angle: tensor of shape batch_size x 3

        Returns:
            rotation_matrix: tensor of shape batch_size x 3 x 3
        """
        initial_shape = tf.shape(axis_angle)

        axis_angle = tf.reshape(axis_angle, [-1, 3])
        batch_size = tf.shape(axis_angle)[0]

        angle = tf.expand_dims(tf.norm(axis_angle + 1e-8, axis=1), -1)
        axis = tf.expand_dims(tf.math.divide(axis_angle, angle), -1)

        angle = tf.expand_dims(angle, -1)
        cos = tf.cos(angle)
        sin = tf.sin(angle)

        outer = tf.matmul(axis, axis, transpose_b=True, name="outer")

        eyes = tf.tile(tf.expand_dims(tf.eye(3), 0), [batch_size, 1, 1])
        R = cos * eyes + (1 - cos) * outer + sin * Skew()(axis)
        R = tf.reshape(R, tf.concat([initial_shape, [3]], axis=0))

        return R


class Skew(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Skew, self).__init__(**kwargs)


    def call(self, vec):
        """Returns the skew symetric version of each 3x3 matrix in a vector

        Args:
            vec: tensor of shape batch_size x 3

        Returns:
            rotation_matrix: tensor of shape batch_size x 3 x 3
        """
        batch_size = tf.shape(vec)[0]
        
        col_inds = tf.constant([1, 2, 3, 5, 6, 7])
        batch_inds = tf.reshape(tf.range(0, batch_size) * 9, [-1, 1])
        indices = tf.reshape(batch_inds + col_inds, [-1, 1])

        updates = tf.stack(
            values=[-vec[:, 2], vec[:, 1], vec[:, 2],
                    -vec[:, 0], -vec[:, 1], vec[:, 0]],
            axis=1
        )
        updates = tf.reshape(updates, [-1])
                
        out_shape = [batch_size * 9]
        res = tf.scatter_nd(indices, updates, out_shape)
        res = tf.reshape(res, [batch_size, 3, 3])

        return res


def finite_diff(x, h):
    v = np.zeros(x.shape)
    v[1:] = (x[1:] - x[0:-1]) / h
    return v
