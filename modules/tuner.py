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

from modules.transform import FEATURE_KEYS, LABEL_KEY, transformed_name

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


class RankingModel(tf.keras.Model):
    def __init__(self, hyperparameters, tf_transform_output, rating_weight, retrieval_weight, movies_uri):
        super().__init__()

        num_hidden_layers = hyperparameters.Choice(
            "num_hidden_layers",
            values=[1, 2, 3],
        )
        embedding_dims = hyperparameters.Int(
            "embedding_dims",
            min_value=16,
            max_value=512,
            step=32,
        )
        l2_regularizers = hyperparameters.Choice(
            "l2_regularizers",
            values=[1e-2, 1e-3, 1e-4]
        )
        dense_unit = hyperparameters.Int(
            "dense_unit",
            min_value=32,
            max_value=1024,
            step=32,
        )
        dropout_rate = hyperparameters.Float(
            "dropout_rate",
            min_value=0.1,
            max_value=0.9,
            step=0.1,
        )

        rating_layers = []

        for _ in range(num_hidden_layers):
            rating_layers.append(layers.Dense(dense_unit, activation=tf.nn.relu))
            rating_layers.append(layers.Dropout(dropout_rate))

        rating_layers.append(layers.Dense(1))

        movies_artifact = movies_uri.get()[0]
        input_dir = artifact_utils.get_split_uri([movies_artifact], "train")
        movie_files = glob.glob(os.path.join(input_dir, "*"))
        movies = tf.data.TFRecordDataset(movie_files, compression_type="GZIP")
        movies_dataset = self.extract_str_feature(movies, "title")

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
                len(self.users_vocab_str) + 1,
                embedding_dims,
                embeddings_initializer='he_normal',
                embeddings_regularizer=keras.regularizers.l2(l2_regularizers),
            ),
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
                len(self.movies_vocab_str) + 1,
                embedding_dims,
                embeddings_initializer='he_normal',
                embeddings_regularizer=keras.regularizers.l2(l2_regularizers),
            ),
        ])

        self.ratings = tf.keras.Sequential(rating_layers)

        self.rating_task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )
        self.retrieval_task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=movies_dataset.batch(128).map(self.movie_model)
            )
        )

        self.rating_weight = rating_weight
        self.retrieval_weight = retrieval_weight

    def extract_str_feature(self, dataset, feature_name):
        np_dataset = []
        for example in dataset:
            np_example = example_coder.ExampleToNumpyDict(example.numpy())
            np_dataset.append(np_example[feature_name][0].decode())
        return tf.data.Dataset.from_tensor_slices(np_dataset)

    def call(self, features: Dict[str, tf.Tensor]):
        try:
            user_embeddings = features[transformed_name(FEATURE_KEYS[0])]
            movie_embeddings = features[transformed_name(FEATURE_KEYS[1])]

            return (
                user_embeddings,
                movie_embeddings,
                self.rating_model(
                    tf.concat([user_embeddings, movie_embeddings], axis=1)
                ),
            )
        except BaseException as err:
            logging.error(f"ERROR IN RankingModel::call:\n{err}")

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        try:
            ratings = features.pop("user_rating")

            user_embeddings, movie_embeddings, rating_predictions = self(
                features)

            rating_loss = self.rating_task(
                labels=ratings,
                predictions=rating_predictions,
            )
            retrieval_loss = self.retrieval_task(
                user_embeddings, movie_embeddings)

            return (self.rating_weight * rating_loss + self.retrieval_weight * retrieval_loss)
        except BaseException as err:
            logging.error(f"ERROR IN RecommenderModel::compute_loss:\n{err}")


class RecommenderModel(tfrs.Model):
    def __init__(self, hyperparameters, tf_transform_output):
        super().__init__()
        self.ranking_model = RankingModel(hyperparameters, tf_transform_output)
        self.task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )

    def call(self, features: Dict[str, tf.Tensor]):
        try:
            return self.ranking_model((
                features[transformed_name(FEATURE_KEYS[0])],
                features[transformed_name(FEATURE_KEYS[1])]
            ))
        except BaseException as err:
            logging.error(f"ERROR IN RecommenderModel::call:\n{err}")

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False):
        try:
            labels = features.pop(transformed_name(LABEL_KEY))
            rating_predictions = self(features)

            return self.task(labels=labels, predictions=rating_predictions)
        except BaseException as err:
            logging.error(
                f"ERROR IN RecommenderModel::compute_loss:\n{err}")


def input_fn(file_pattern, data_accessor, tf_transform_output, batch_size=64):
    try:
        return data_accessor.tf_dataset_factory(
            file_pattern,
            tfxio.TensorFlowDatasetOptions(
                batch_size=batch_size,
            ),
            tf_transform_output.transformed_metadata.schema,
        )
    except BaseException as err:
        logging.error(f"ERROR IN _input_fn:\n{err}")


def _get_model(hyperparameters, tf_transform_output):
    try:
        learning_rate = hyperparameters.Choice(
            "learning_rate",
            values=[1e-1, 1e-2, 1e-3, 1e-4]
        )
        model = RecommenderModel(hyperparameters, tf_transform_output)
        model.compile(
            optimizer=tf.keras.optimizers.Adagrad(learning_rate=learning_rate))
        return model
    except BaseException as err:
        logging.error(f"ERROR IN get_model:\n{err}")


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
            hypermodel=lambda hp: _get_model(hp, tf_transform_output),
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
