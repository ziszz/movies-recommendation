import tensorflow as tf
from absl import logging

MOVIE_CATEGORICAL_FEATURE = "title"

def transformed_name(key):
    return f"{key}_xf"


def preprocessing_fn(inputs):
    try:
        outputs = {}
        outputs[transformed_name(MOVIE_CATEGORICAL_FEATURE)] = tf.strings.lower(inputs[MOVIE_CATEGORICAL_FEATURE])

        return outputs
    except BaseException as err:
        logging.error(f"ERROR IN preprocessing_fn:\n{err}")
