from typing import Any, Dict, NamedTuple, Text, Tuple

import keras
import keras_tuner as kt
import tensorflow as tf
import tensorflow_recommenders as tfrs
import tensorflow_transform as tft
from absl import logging
from keras import layers
from keras_tuner.engine import base_tuner

from modules.transform import FEATURE_KEYS, transformed_name
from modules.utils import input_fn

TunerFnResult = NamedTuple("TunerFnResult", [
    ("tuner", base_tuner.BaseTuner),
    ("val_loss", Dict[Text, Any])
])

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_root_mean_squared_error",
    mode="min",
    verbose=1,
    patience=10,
)


class CFModel(tfrs.Model):
    def __init__(self, hyperparameters, tf_transform_output):
        super().__init__()

        # hyperparameters
        embedding_dims = hyperparameters.Int(
            "embedding_dims", min_value=16, max_value=1024, step=32)
        l2_regularizers = hyperparameters.Choice(
            "l2_regularizers", values=[1e-2, 1e-3, 1e-4])
        num_hidden_layers = hyperparameters.Choice(
            "num_hidden_layers", values=[1, 2, 3])
        dense_unit = hyperparameters.Int(
            "dense_unit", min_value=8, max_value=256, step=32)
        dropout_rate = hyperparameters.Float(
            "dropout_rate", min_value=0.1, max_value=0.7, step=0.1)

        self.user_model = self._build_user_model(
            tf_transform_output, embedding_dims, l2_regularizers)
        self.movie_model = self._build_movie_model(
            tf_transform_output, embedding_dims, l2_regularizers)
        self.rating_model = self._build_rating_model(
            num_hidden_layers, dense_unit, dropout_rate)

        self.task = tfrs.tasks.Ranking(
            loss=keras.losses.MeanSquaredError(),
            metrics=[keras.metrics.RootMeanSquaredError()],
        )

    def _build_user_model(self, tf_transform_output, embedding_dims, l2_regularizers):
        try:
            unique_user_ids = tf_transform_output.vocabulary_by_name(
                f"{FEATURE_KEYS[0]}_vocab")
            users_vocab_str = [b.decode() for b in unique_user_ids]

            model = keras.Sequential([
                layers.Input(shape=(1,), name=transformed_name(
                    FEATURE_KEYS[0]), dtype=tf.int64),
                layers.Lambda(lambda x: tf.as_string(x)),
                layers.StringLookup(
                    vocabulary=users_vocab_str, mask_token=None),
                layers.Embedding(
                    len(users_vocab_str) + 1,
                    embedding_dims,
                    embeddings_initializer='he_normal',
                    embeddings_regularizer=keras.regularizers.l2(
                        l2_regularizers),
                ),
            ])

            return model
        except BaseException as err:
            logging.error(f"ERROR IN CFModel::_build_user_model:\n{err}")

    def _build_movie_model(self, tf_transform_output, embedding_dims, l2_regularizers):
        try:
            unique_movie_ids = tf_transform_output.vocabulary_by_name(
                f"{FEATURE_KEYS[1]}_vocab")
            movies_vocab_str = [b.decode() for b in unique_movie_ids]

            model = keras.Sequential([
                layers.Input(shape=(1,), name=transformed_name(
                    FEATURE_KEYS[1]), dtype=tf.int64),
                layers.Lambda(lambda x: tf.as_string(x)),
                layers.StringLookup(
                    vocabulary=movies_vocab_str, mask_token=None),
                layers.Embedding(
                    len(movies_vocab_str) + 1,
                    embedding_dims,
                    embeddings_initializer='he_normal',
                    embeddings_regularizer=keras.regularizers.l2(
                        l2_regularizers),
                ),
            ])

            return model
        except BaseException as err:
            logging.error(f"ERROR IN CFModel::_build_movie_model:\n{err}")

    def _build_rating_model(self, num_hidden_layers, dense_unit, dropout_rate):
        try:
            flc_layers = []

            for _ in range(num_hidden_layers):
                flc_layers.append(layers.Dense(
                    dense_unit, activation=tf.nn.relu))
                flc_layers.append(layers.Dropout(dropout_rate))

            flc_layers.append(layers.Dense(1))

            model = keras.Sequential(flc_layers)

            return model
        except BaseException as err:
            logging.error(f"ERROR IN CFModel::_build_rating_model:\n{err}")

    def call(self, features: Dict[Text, tf.Tensor]):
        try:
            user_embedding = self.user_model(
                features[transformed_name(FEATURE_KEYS[0])])
            movie_embedding = self.movie_model(
                features[transformed_name(FEATURE_KEYS[1])])

            return self.rating_model(
                tf.concat([user_embedding, movie_embedding], axis=2))
        except BaseException as err:
            logging.error(f"ERROR IN CFModel::call:\n{err}")

    def compute_loss(self, features: Tuple, training=False):
        try:
            labels = features[1]
            rating_predictions = self(features[0])

            return self.task(labels=labels, predictions=rating_predictions)
        except BaseException as err:
            logging.error(
                f"ERROR IN CFModel::compute_loss:\n{err}")


def _get_model(hyperparameters, tf_transform_output):
    try:
        learning_rate = hyperparameters.Choice(
            "learning_rate", values=[1e-1, 1e-2, 1e-3, 1e-4])

        model = CFModel(hyperparameters, tf_transform_output)
        model.compile(optimizer=keras.optimizers.Adagrad(
            learning_rate=learning_rate))

        return model
    except BaseException as err:
        logging.error(f"ERROR IN _get_model:\n{err}")


def tuner_fn(fn_args):
    try:
        tf_transform_output = tft.TFTransformOutput(
            fn_args.transform_graph_path)

        train_dataset = input_fn(
            fn_args.train_files[0], tf_transform_output, batch_size=128)
        eval_dataset = input_fn(
            fn_args.eval_files[0], tf_transform_output, batch_size=128)

        tuner = kt.Hyperband(
            hypermodel=lambda hp: _get_model(
                hyperparameters=hp,
                tf_transform_output=tf_transform_output,
            ),
            objective=[
                kt.Objective(
                    "val_loss",
                    direction="min",
                ),
                kt.Objective(
                    "val_root_mean_squared_error",
                    direction="min",
                ),
            ],
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
