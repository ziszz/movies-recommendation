import glob
import os
from typing import Any, Dict, NamedTuple, Text

import keras
import keras_tuner as kt
import tensorflow as tf
import tensorflow_recommenders as tfrs
import tensorflow_transform as tft
from absl import logging
from keras import layers
from keras_tuner.engine import base_tuner
from tfx.types import artifact_utils
from tfx_bsl.coders import example_coder
from tfx_bsl.public import tfxio

from modules.transform import (CATEGORICAL_FEATURE, LABEL_KEY,
                               NUMERICAL_FEATURE, transformed_name)

TunerFnResult = NamedTuple("TunerFnResult", [
    ("tuner", base_tuner.BaseTuner),
    ("fit_kwargs", Dict[Text, Any])
])

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_root_mean_squared_error",
    mode="min",
    verbose=1,
    patience=10,
)


class RecommenderNet(tfrs.models.Model):
    def __init__(self, tf_transform_output, movies_uri) -> None:
        super().__init__()

        movies_artifact = movies_uri.get()[0]
        input_dir = artifact_utils.get_split_uri([movies_artifact], "train")
        movie_files = glob.glob(os.path.join(input_dir, '*'))
        movies = tf.data.TFRecordDataset(movie_files, compression_type="GZIP")
        movies_ds = extract_str_feature(movies, CATEGORICAL_FEATURE)

        # weights
        self.rating_weight = 1.0
        self.retrieval_weight = 1.0

        # hyperparameters
        self.embedding_dims = 128,
        self.l2_regularizers = 1e-3

        # helper model
        self.user_model = self._build_user_model(
            tf_transform_output,
            self.embedding_dims,
            self.l2_regularizers,
        )
        self.movie_model = self._build_movie_model(
            tf_transform_output,
            self.embedding_dims,
            self.l2_regularizers,
        )

        # tasks
        self.rating_task = tfrs.keras.Ranking(
            loss=keras.losses.MeanSquaredError(),
            metrics=[keras.metrics.RootMeanSquarredError()],
        )
        self.retrieal_task = tfrs.keras.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=movies_ds.batch(120).map(self.movie_model)
            )
        )

    def _build_user_model(self, tf_transform_output, embedding_dims, l2_regularizers):
        try:
            unique_user_ids = tf_transform_output.vocabulary_by_name(
                f"{NUMERICAL_FEATURE}_vocab")
            users_vocab_str = [i.decode() for i in unique_user_ids]

            return keras.Sequential([
                layers.StringLookup(
                    vocabulary=users_vocab_str,
                    mask_token=None,
                ),
                layers.Embedding(
                    len(users_vocab_str) + 1,
                    embedding_dims,
                    embeddings_initializer='he_normal',
                    embeddings_regularizer=keras.regularizers.l2(
                        l2_regularizers),
                )
            ])
        except BaseException as err:
            logging.error(
                f"ERROR IN RecommenderNet::_build_user_model:\n{err}")

    def _build_movie_model(self, tf_transform_output, embedding_dims, l2_regularizers):
        try:
            unique_movie_ids = tf_transform_output.vocabulary_by_name(
                f"{CATEGORICAL_FEATURE}_vocab")
            movies_vocab_str = [i.decode() for i in unique_movie_ids]

            return keras.Sequential([
                layers.StringLookup(
                    vocabulary=movies_vocab_str,
                    mask_token=None,
                ),
                layers.Embedding(
                    len(movies_vocab_str) + 1,
                    embedding_dims,
                    embeddings_initializer='he_normal',
                    embeddings_regularizer=keras.regularizers.l2(
                        l2_regularizers),
                )
            ])
        except BaseException as err:
            logging.error(
                f"ERROR IN RecommenderNet::_build_user_model:\n{err}")

    def _rating_model(self):
        try:
            return keras.Sequential([
                layers.Dense(128, activation=tf.nn.relu),
                layers.Dense(64, activation=tf.nn.relu),
                layers.Dense(1),
            ])
        except BaseException as err:
            logging.error(
                f"ERROR IN RecommenerNet::_build_rating_model:\n{err}")

    def call(self, features: Dict[Text, tf.Tensor]):
        try:
            user_embeddings = self.user_model(
                features[transformed_name(NUMERICAL_FEATURE)])
            movie_embeddings = self.movie_model(
                features[transformed_name(CATEGORICAL_FEATURE)])

            return (
                user_embeddings,
                movie_embeddings,
                self.rating_model(
                    tf.concat([user_embeddings, movie_embeddings], axis=1))
            )
        except BaseException as err:
            logging.error(f"ERROR IN RecommenderNet::call:\n{err}")

    def compute_loss(self, features: Dict(Text, tf.Tensor), training=False):
        try:
            ratings = features.pop(transformed_name(LABEL_KEY))

            user_embeddings, movie_embeddings, rating_predictions = self(
                features)

            rating_loss = self.rating_task(
                labels=ratings,
                predictions=rating_predictions,
            )
            retrieval_loss = self.retrieval_task(
                user_embeddings, movie_embeddings)

            return (self.rating_weight * rating_loss + self.retrieval_weight + retrieval_loss)
        except BaseException as err:
            logging.error(f"ERROR IN RecommenderNet::compute_loss:\n{err}")


def extract_str_feature(dataset, feature_name):
    try:
        np_dataset = []
        for example in dataset:
            np_example = example_coder.ExampleToNumpyDict(example.numpy())
            np_dataset.append(np_example[feature_name][0].decode())
        return tf.data.Dataset.from_tensor_slices(np_dataset)
    except BaseException as err:
        logging.error(f"ERROR IN extract_str_feature:\n{err}")


def input_fn(file_pattern, data_accessor, tf_transform_output, batch_size=64):
    try:
        return data_accessor.tf_dataset_factory(
            file_pattern,
            tfxio.TensorFlowDatasetOptions(
                batch_size=batch_size,
            ),
            schema=tf_transform_output.transformed_metadata.schema
        ).repeat()
    except BaseException as err:
        logging.error(f"ERROR IN input_fn:\n{err}")


def _get_model(tf_transform_output, movies_uri):
    try:
        model = RecommenderNet(tf_transform_output, movies_uri)
        model.compile(optimizer=keras.optimizers.Adagrad(learning_rate=0.1))

        return model
    except BaseException as err:
        logging.error(f"ERROR IN _get_model:\n{err}")


def tuner_fn(fn_args):
    try:
        tf_transform_output = tft.TFTransformOutput(
            fn_args.transform_graph_path)

        train_dataset = input_fn(
            fn_args.train_files[0],
            fn_args.data_accessor,
            tf_transform_output,
            batch_size=128,
        )
        eval_dataset = input_fn(
            fn_args.eval_files[0],
            fn_args.data_accessor,
            tf_transform_output,
            batch_size=128,
        )

        tuner = kt.Hyperband(
            hypermodel=lambda hp: _get_model(
                hyperparameters=hp,
                tf_transform_output=tf_transform_output,
                rating_weight=0.0,
                retrieval_weight=1.0,
                movies_uri=fn_args.examples,
            ),
            objective=kt.Objective(
                "val_root_mean_squared_error",
                direction="min",
            ),
            max_epochs=fn_args.custom_config["epochs"],
            factor=3,
            directory=fn_args.working_dir,
            project_name="kt_hyperband",
        )

        return TunerFnResult(
            tuner=tuner,
            fit_kwargs={
                "x": train_dataset,
                "validation_data": eval_dataset,
                "steps_per_epoch": fn_args.train_steps,
                "validation_steps": fn_args.eval_steps,
                "callbacks": [early_stop]
            },
        )
    except BaseException as err:
        logging.error(f"ERROR IN tuner_fn after tuning:\n{err}")
