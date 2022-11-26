import os

import tensorflow as tf
import tensorflow_recommenders as tfrs
from absl import logging
from keras import layers
from tfx_bsl.public import tfxio
from modules.transform import CATEGORICAL_FEATURE, NUMERICAL_FEATURE


def _input_fn(file_pattern, data_accessor, tf_transform_output, batch_size=64):
    try:
        return data_accessor.tf_dataset_factory(
            file_pattern,
            tfxio.TensorFlowDatasetOptions(
                batch_size=batch_size,
            ),
            tf_transform_output.transformed_metadata.schema,
        )
    except BaseException as err:
        logging.error(f"ERROR IN input_fn:\n{err}")


def _get_serve_tf_examples_fn(model, tf_transform_output):
    try:
        model.tft_layer = tf_transform_output.transform_features_layers()

        @tf.function
        def serve_tf_examples_fn(serialized_tf_examples):
            try:
                feature_spec = tf_transform_output.raw_feature_spec()
                parsed_features = tf.io.parse_example(
                    serialized_tf_examples, feature_spec)
                transformed_features = model.tft_layer(parsed_features)
                result = model(transformed_features)
            except BaseException as err:
                logging.error(f"ERROR IN serve_tf_examples_fn:\n{err}")

            return result
    except BaseException as err:
        logging.error(f"ERROR IN _get_serve_tf_examples_fn:\n{err}")

    return serve_tf_examples_fn


def _get_model(tf_transform_output):
    try:
        model = tf.keras.Sequ
    except BaseException as err:
        logging.error(f"ERROR IN _get_model:\n{err}")
