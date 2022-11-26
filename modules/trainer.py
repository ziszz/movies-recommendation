import os
from typing import Dict, List, Text

import tensorflow as tf
import tensorflow_recommenders as tfrs
from absl import logging
from keras import layers
from tfx_bsl.public import tfxio

from modules.transform import FEATURE_KEYS, transformed_name


class RankingModel(tf.keras.Model):
    def __init__(self, tf_transform_output):
        super().__init__()
        self.embedding_dims = 64

        self.unique_user_ids = tf_transform_output.vocabulary_by_name(
            f"{FEATURE_KEYS[0]}_vocab")
        self.users_vocab_str = [b.decode() for b in self.unique_user_ids]

        self.user_embeddings = tf.keras.Sequential([
            layers.Input(shape=(1,), name=transformed_name(
                FEATURE_KEYS[0]), dtype=tf.int64),
            layers.Lambda(lambda x: tf.as_string(x)),
            layers.StringLookup(
                vocabulary=self.unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(
                len(self.users_vocab_str) + 1, self.embedding_dims),
        ])

        self.unique_movie_ids = tf_transform_output.vocabulary_by_name(
            f"{FEATURE_KEYS[1]}_vocab")
        self.movies_vocab_str = [b.decode() for b in self.unique_movie_ids]

        self.movie_embeddings = tf.keras.Sequential([
            layers.Input(shape=(1,), name=transformed_name(
                FEATURE_KEYS[1]), dtype=tf.int64),
            layers.Lambda(lambda x: tf.as_string(x)),
            layers.StringLookup(
                vocabulary=self.unique_movie_ids, mask_token=None),
            tf.keras.layers.Embedding(
                len(self.movies_vocab_str) + 1, self.embedding_dims),
        ])

        self.ratings = tf.keras.Sequential([
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(64, activation=tf.nn.relu),
            layers.Dense(1),
        ])

        def call(self, inputs):
            try:
                user_id, movie_id = inputs

                user_embedding = self.user_embeddings(user_id)
                movie_embedding = self.movie_embeddings(movie_id)

                return self.ratings(tf.concat(user_embedding, movie_embedding), axis=2)
            except BaseException as err:
                logging.error(f"ERROR IN RankingModel::call:\n{err}")


class RecommenderModel(tfrs.Model):
    def __init__(self):
        super().__init__()
        self.ranking_model = RankingModel()
        self.task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )

        def call(self, features: Dict[str, tf.Tensor]):
            try:
                return self.ranking_model((features[FEATURE_KEYS[0]], features[FEATURE_KEYS[1]]))
            except BaseException as err:
                logging.error(f"ERROR IN RecommenderModel::call:\n{err}")

        def compute_loss(self, features: Dict[Text, tf.Tensor], training=False):
            try:
                labels = features[1]
                rating_predictions = self(features[0])

                return self.task(labels=labels, predictions=rating_predictions)
            except BaseException as err:
                logging.error(
                    f"ERROR IN RecommenderModel::compute_loss:\n{err}")


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

                return result
            except BaseException as err:
                logging.error(f"ERROR IN serve_tf_examples_fn:\n{err}")

        return serve_tf_examples_fn

    except BaseException as err:
        logging.error(f"ERROR IN _get_serve_tf_examples_fn:\n{err}")


def _get_model():
    try:
        return RecommenderModel()
    except BaseException as err:
        logging.error(f"ERROR IN _get_model:\n{err}")
