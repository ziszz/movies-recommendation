import tensorflow as tf
import tensorflow_transform as tft
from absl import logging

from modules.utils import transformed_name

NUM_OF_BUCKETS = 1

NUMERICAL_FEATURES = ["userId", "movieId"]
CATEGORICAL_FEATURES = "genres"


def preprocessing_fn(inputs):
    try:
        outputs = {}

        for key in NUMERICAL_FEATURES:
            outputs[transformed_name(key)] = tf.cast(inputs[key], tf.int64)
            tft.compute_and_apply_vocabulary(
                inputs[key],
                num_oov_buckets=NUM_OF_BUCKETS,
                vocab_filename=f"{key}_vocab"
            )

        cat_features = tf.strings.lower(inputs[CATEGORICAL_FEATURES])
        outputs[transformed_name(CATEGORICAL_FEATURES)] = tft.compute_and_apply_vocabulary(
            cat_features,
            num_oov_buckets=NUM_OF_BUCKETS,
            vocab_filename=f"{CATEGORICAL_FEATURES}_vocab"
        )

        return outputs
    except BaseException as err:
        logging.error(f"ERROR IN preprocessing_fn:\n{err}")
