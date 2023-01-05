import tensorflow as tf
import tensorflow_transform as tft
from absl import logging

from modules.utils import transformed_name

NUM_OF_BUCKETS = 1

NUMERICAL_FEATURES = "userId"
CATEGORICAL_FEATURE = "title"
LABEL_KEY = "rating"


def preprocessing_fn(inputs):
    try:
        outputs = {}

        outputs[transformed_name(NUMERICAL_FEATURES)] = tf.cast(
            inputs[NUMERICAL_FEATURES], tf.int64)

        outputs[transformed_name(CATEGORICAL_FEATURE)] = tft.compute_and_apply_vocabulary(
            inputs[CATEGORICAL_FEATURE])

        outputs[transformed_name(LABEL_KEY)] = tf.cast(
            inputs[LABEL_KEY], tf.int64)

        return outputs
    except BaseException as err:
        logging.error(f"ERROR IN preprocessing_fn:\n{err}")
