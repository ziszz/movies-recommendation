import tensorflow as tf
import tensorflow_transform as tft
from absl import logging

from modules.utils import transformed_name

NUM_OF_BUCKETS = 1

NUMERICAL_FEATURES = ["userId", "movieId"]
CATEGORICAL_FEATURE = "genres"
LABEL_KEY = "rating"


def preprocessing_fn(inputs):
    try:
        outputs = {}

        for key in NUMERICAL_FEATURES:
            outputs[transformed_name(key)] = tf.cast(inputs[key], tf.int64)

        outputs[transformed_name(CATEGORICAL_FEATURE)] = tft.compute_and_apply_vocabulary(
            inputs[CATEGORICAL_FEATURE])

        outputs[transformed_name(LABEL_KEY)] = tf.cast(
            inputs[LABEL_KEY], tf.int64)

        return outputs
    except BaseException as err:
        logging.error(f"ERROR IN preprocessing_fn:\n{err}")
