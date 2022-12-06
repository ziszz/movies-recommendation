import tensorflow as tf
import tensorflow_transform as tft
from absl import logging

NUMERICAL_FEATURE = "userId"
CATEGORICAL_FEATURE = "title"
LABEL_KEY = "rating"


def transformed_name(key):
    return f"{key}_xf"


def preprocessing_fn(inputs):
    try:
        outputs = {}

        outputs[transformed_name(NUMERICAL_FEATURE)] = tf.cast(
            inputs[NUMERICAL_FEATURE], tf.int64)
        outputs[transformed_name(CATEGORICAL_FEATURE)] = tf.strings.lower(
            inputs[CATEGORICAL_FEATURE])

        tft.compute_and_apply_vocabulary(
            inputs[NUMERICAL_FEATURE],
            num_oov_buckets=1,
            vocab_filename=f"{NUMERICAL_FEATURE}_vocab"
        )
        tft.compute_and_apply_vocabulary(
            inputs[CATEGORICAL_FEATURE],
            num_oov_buckets=1,
            vocab_filename=f"{CATEGORICAL_FEATURE}_vocab"
        )

        outputs[transformed_name(LABEL_KEY)] = tf.cast(
            inputs[LABEL_KEY], tf.int64)

        return outputs
    except BaseException as err:
        logging.error(f"ERROR IN preprocessing_fn:\n{err}")
