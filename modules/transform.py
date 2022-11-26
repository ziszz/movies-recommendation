import os

import tensorflow as tf
import tensorflow_transform as tft
from absl import logging

FEATURES = [
    "userId",
    "movieId",
]


def transformed_name(key):
    return f"{key}_xf"


def preprocessing_fn(inputs):
    try:
        outputs = {}

        for key in FEATURES:
            outputs[key] = tf.cast(inputs[key], tf.int64)
    except BaseException as err:
        logging.error(f"ERROR IN _get_model:\n{err}")

    return outputs
