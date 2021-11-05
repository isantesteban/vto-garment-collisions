import numpy as np
import tensorflow as tf
from scipy.spatial.transform import Rotation as R

from . import math
from . import skinning
from . import smpl


def load_model():
    return {
        "smpl": smpl.SMPL(
            "assets/SMPL/basicModel_f_lbs_10_207_0_v1.0.0.pkl"
        ),

        "body/pose_encoder": tf.keras.models.load_model(
            "trained_models/diffused_body/pose_encoder",
            compile=False
        ),

        "body/skinning_weights": tf.keras.models.load_model(
            "trained_models/diffused_body/skinning_weights",
            compile=False
        ),

        "body/pose_blendshape": tf.keras.models.load_model(
            "trained_models/diffused_body/pose_blendshape",
            compile=False
        ),

        "body/shape_blendshape": tf.keras.models.load_model(
            "trained_models/diffused_body/shape_blendshape",
            compile=False
        ),

        "tshirt/gru": tf.keras.models.load_model(
            "trained_models/tshirt/gru",
            compile=False
        ),

        "tshirt/decoder": tf.keras.models.load_model(
            "trained_models/tshirt/decoder",
            compile=False
        )
    }


def run_model(model_dict, motion):
    '''
    This function evaluates the runtime pipeline step by step.
    
    Note: to run our model at interactive framerates we wrap 
    this code into a custom Keras model and use TensorRT to optimize 
    the computational graph. We provide the unoptimized code because
    despite being slower it's much clearer and shows all the computations
    involved in our method (and doesn't require additional dependencies).
    '''

    num_frames = len(motion["pose"])

    # Run pose encoder
    motion["pose_encoded"] = model_dict["body/pose_encoder"].predict(
        motion["pose"][:, 3:]
    )

    # Compute velocities and accelerations for input vector
    h = 1.0 / 30.0 # 30 fps
    motion['translation_vel'] = math.finite_diff(motion['translation'], h)
    motion['translation_acc'] = math.finite_diff(motion['translation_vel'], h)

    rot =  R.from_rotvec(motion['pose'][:,:3])
    motion['euler_angles'] = rot.as_euler('zxy')
    motion['euler_angles_vel'] = math.finite_diff(motion['euler_angles'], h)
    motion['euler_angles_acc'] = math.finite_diff(motion['euler_angles_vel'], h)

    motion['pose_encoded_vel'] = math.finite_diff(motion['pose_encoded'], h)
    motion['pose_encoded_acc'] = math.finite_diff(motion['pose_encoded_vel'], h)

    shape = np.tile(motion["shape"], (num_frames, 1))

    # Run model
    print("[INFO] Run recurrent regressor...")
    v_encoded = model_dict["tshirt/gru"].predict({
        'shape': np.expand_dims(shape, axis=0),
        'pose_encoded': np.expand_dims(motion['pose_encoded'], axis=0),
        'pose_encoded_vel': np.expand_dims(motion['pose_encoded_vel'], axis=0),
        'pose_encoded_acc': np.expand_dims(motion['pose_encoded_acc'], axis=0),
        'translation_vel': np.expand_dims(motion['translation_vel'], axis=0),
        'translation_acc': np.expand_dims(motion['translation_acc'], axis=0),
        'euler_angles_vel': np.expand_dims(motion['euler_angles_vel'], axis=0),
        'euler_angles_acc': np.expand_dims(motion['euler_angles_acc'], axis=0),
    })[0]

    print("[INFO] Project from latent space to canonical space...")
    v_canonical = model_dict["tshirt/decoder"].predict(v_encoded)
    v_canonical_flat = tf.reshape(v_canonical, (-1, 3))
    num_vertices = v_canonical.shape[-2]
 
    print("[INFO] Project from canonical space to unpose...")
    v_body, smpl_dict = model_dict["smpl"](shape, motion["pose"])
    
    pose_repeat = tf.repeat(smpl_dict['pose_feature'], num_vertices, axis=0)
    pose_blendshape = model_dict["body/pose_blendshape"].predict([v_canonical_flat, pose_repeat])
    
    shape_repeat = tf.repeat(shape, num_vertices, axis=0)
    shape_blendshape = model_dict["body/shape_blendshape"].predict([v_canonical_flat, shape_repeat])

    skinning_weights = model_dict["body/skinning_weights"].predict(v_canonical_flat)
    skinning_weights = tf.reshape(skinning_weights, (num_frames, num_vertices, -1))

    v_unpose = v_canonical_flat + pose_blendshape + shape_blendshape
    v_unpose = tf.reshape(v_unpose, (num_frames, num_vertices, 3))

    print("[INFO] Compute linear blend skinning...")
    joint_transforms = smpl_dict['joint_transforms']
    v_garment = skinning.LBS()(v_unpose, joint_transforms, skinning_weights)

    # Add translation
    v_body = v_body + motion["translation"][:, None, :]
    v_garment = v_garment + motion["translation"][:, None, :]

    print("[INFO] Done!")

    return v_garment.numpy(), v_body.numpy()
