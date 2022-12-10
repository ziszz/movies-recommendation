import os

import tensorflow as tf
import tensorflow_transform as tft
from absl import logging

NUMERIC_FEATURE = "userId"
CATEGORICAL_FEATURE = "title"


def transformed_name(key):
    return f"{key}_xf"


def preprocessing_fn(inputs):
    try:
        outputs = {}

        outputs[transformed_name(NUMERIC_FEATURE)] = tf.cast(
            inputs[NUMERIC_FEATURE], tf.int64)
        outputs[transformed_name(CATEGORICAL_FEATURE)] = tf.strings.lower(
            inputs[CATEGORICAL_FEATURE])

        tft.compute_and_apply_vocabulary(
            inputs[NUMERIC_FEATURE],
            num_oov_buckets=1,
            vocab_filename=f"{NUMERIC_FEATURE}_vocab"
        )
        tft.compute_and_apply_vocabulary(
            inputs[CATEGORICAL_FEATURE],
            num_oov_buckets=1,
            vocab_filename=f"{CATEGORICAL_FEATURE}_vocab"
        )
        return outputs
    except BaseException as err:
        logging.error(f"ERROR IN _get_model:\n{err}")
