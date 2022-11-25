import os

import tensorflow as tf
from absl import logging
from keras import layers
from tensorflow_similarity.layers import MetricEmbedding
from tensorflow_similarity.losses import MultiSimilarityLoss
from tensorflow_similarity.models import SimilarityModel
from tfx_bsl.public import tfxio

from movie_transform import FEATURE_KEYS, transformed_name


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


def _get_model(movie_vocab, vectorizer_layer):
    try:
        input_features = []

        for key in FEATURE_KEYS:
            input_features.append(
                layers.Input(shape=(1,), name=transformed_name(key))
            )

        concatenate = layers.concatenate(input_features)
        x = layers.Conv2D(64, 3, activation='relu')(concatenate)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu')(x)
        outputs = MetricEmbedding(64)(x)

        model = SimilarityModel(input_features, outputs)

    except BaseException as err:
        logging.error(f"ERROR IN _get_model:\n{err}")

    return model
