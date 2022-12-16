import tensorflow as tf
import tensorflow_transform as tft
from absl import logging

NUM_OF_BUCKETS = 1

FEATURE_KEYS = [
    "userId",
    "movieId",
]

LABEL_KEY = "rating"


def transformed_name(key):
    return f"{key}_xf"


def preprocessing_fn(inputs):
    try:
        outputs = {}

        for key in FEATURE_KEYS:
            outputs[transformed_name(key)] = tf.cast(inputs[key], tf.int64)

        outputs[transformed_name(LABEL_KEY)] = tf.cast(
            inputs[LABEL_KEY], tf.int64)

        return outputs
    except BaseException as err:
        logging.error(f"ERROR IN preprocessing_fn:\n{err}")
