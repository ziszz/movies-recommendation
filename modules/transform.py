import os

import tensorflow as tf
import tensorflow_transform as tft
from absl import logging

NUMERICAL_FEATURE = "userId"
CATEGORICAL_FEATURE = "title"


def transformed_name(key):
    return f"{key}_xf"


def preprocessing_fn(inputs):
    try:
        outputs = {}

        outputs[CATEGORICAL_FEATURE] = tft.compute_and_apply_vocabulary(
            tf.strings.strip(inputs[CATEGORICAL_FEATURE]),
            num_oov_buckets=1,
            vocab_filename=os.path.abspath(f"vocabulary/{CATEGORICAL_FEATURE}"),
        )

        outputs[NUMERICAL_FEATURE] = tf.cast(inputs[NUMERICAL_FEATURE], tf.int64)
    except BaseException as err:
        logging.error(f"ERROR IN _get_model:\n{err}")
        
    return outputs
