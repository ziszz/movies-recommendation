import tensorflow as tf
import tensorflow_transform as tft
from absl import logging

from modules.utils import transformed_name

NUM_OF_BUCKETS = 1

NUMERICAL_FEATURES = "userId"
CATEGORICAL_FEATURES = ["genres", "title"]
LABEL_KEY = "rating"


def preprocessing_fn(inputs):
    try:
        outputs = {}

        outputs[transformed_name(NUMERICAL_FEATURES)] = tf.cast(
            inputs[NUMERICAL_FEATURES], tf.int64)

        for key in CATEGORICAL_FEATURES:
            outputs[transformed_name(key)] = tft.compute_and_apply_vocabulary(
                inputs[key])

        outputs[transformed_name(LABEL_KEY)] = tft.scale_to_0_1(
            inputs[LABEL_KEY])

        return outputs
    except BaseException as err:
        logging.error(f"ERROR IN preprocessing_fn:\n{err}")
