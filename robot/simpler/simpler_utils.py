import numpy as np
import simpler_env
import tensorflow as tf
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from transforms3d.euler import euler2axangle

from experiments.robot.robot_utils import normalize_gripper_action


def get_simpler_img(env, obs, resize_size):
    """
    Takes in environment and observation and returns resized image as numpy array.

    NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
                    the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
    """
    assert isinstance(resize_size, int)
    image = get_image_from_maniskill2_obs_dict(env, obs)

    # Preprocess the image the exact same way that the Berkeley Bridge folks did it
    # to minimize distribution shift.
    # NOTE (Moo Jin): Yes, we resize down to 256x256 first even though the image may end up being
    # resized up to a different resolution by some models. This is just so that we're in-distribution
    # w.r.t. the original preprocessing at train time.
    IMAGE_BASE_PREPROCESS_SIZE = 128
    # Resize to image size expected by model
    image = tf.image.encode_jpeg(image)  # Encode as JPEG, as done in RLDS dataset builder
    image = tf.io.decode_image(image, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
    image = tf.image.resize(
        image, (IMAGE_BASE_PREPROCESS_SIZE, IMAGE_BASE_PREPROCESS_SIZE), method="lanczos3", antialias=True
    )
    image = tf.image.resize(image, (resize_size, resize_size), method="lanczos3", antialias=True)
    image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)
    return image.numpy()


def get_simpler_env(task, model_family):
    """Initializes and returns the Simpler environment along with the task description."""
    env = simpler_env.make(task)
    return env


def get_simpler_dummy_action(model_family: str):
    if model_family == "octo":
        # TODO: don't hardcode the action horizon for Octo
        return np.tile(np.array([0, 0, 0, 0, 0, 0, -1])[None], (4, 1))
    else:
        return np.array([0, 0, 0, 0, 0, 0, -1])


def convert_maniskill(action):
    """
    Applies transforms to raw VLA action that Maniskill simpler_env env expects.
    Converts rotation to axis_angle.
    Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1] and binarizes.
    """
    assert action.shape[0] == 7

    # Change rotation to axis-angle
    action = action.copy()
    roll, pitch, yaw = action[3], action[4], action[5]
    action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
    action[3:6] = action_rotation_ax * action_rotation_angle

    # Binarize final gripper dimension & map to [-1...1]
    return normalize_gripper_action(action)
