import tensorflow as tf
import tensorflow_transform as tft
from absl import logging

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
            tft.compute_and_apply_vocabulary(
                inputs[key],
                num_oov_buckets=1,
                vocab_filename=f"{key}_vocab"
            )

        outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[key], tf.int64)
        return outputs
    except BaseException as err:
        logging.error(f"ERROR IN preprocessing_fn:\n{err}")
