import os
from typing import Dict, Text

import keras
import tensorflow as tf
import tensorflow_recommenders as tfrs
import tensorflow_transform as tft
from absl import logging
from keras import layers

from modules.transform import FEATURE_KEYS
from modules.utils import input_fn, transformed_name


class CFModel(tfrs.Model):
    def __init__(self, tf_transform_output):
        super().__init__()
        self.embedding_dims = 128
        self.l2_regularizers = 1e-3

        self.user_model = self._build_user_model(
            tf_transform_output, self.embedding_dims, self.l2_regularizers)
        self.movie_model = self._build_movie_model(
            tf_transform_output, self.embedding_dims, self.l2_regularizers)
        self.rating_model = self._build_rating_model(2, 128, .5)

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
                    vocabulary=unique_user_ids, mask_token=None),
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
                    vocabulary=unique_movie_ids, mask_token=None),
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
            rating_layers = []

            for _ in range(num_hidden_layers):
                rating_layers.append(layers.Dense(
                    dense_unit, activation=tf.nn.relu))
                rating_layers.append(layers.Dropout(dropout_rate))

            rating_layers.append(layers.Dense(1))

            model = keras.Sequential(rating_layers)

            return model
        except BaseException as err:
            logging.error(f"ERROR IN CFModel::_build_rating_model:\n{err}")

    def call(self, features: Dict[str, tf.Tensor]):
        try:
            user_embedding = self.user_model(
                features[transformed_name(FEATURE_KEYS[0])])
            movie_embedding = self.movie_model(
                features[transformed_name(FEATURE_KEYS[1])])

            return self.rating_model(
                tf.concat([user_embedding, movie_embedding], axis=2))
        except BaseException as err:
            logging.error(f"ERROR IN CFModel::call:\n{err}")

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False):
        try:
            labels = features[1]
            rating_predictions = self(features[0])

            return self.task(labels=labels, predictions=rating_predictions)
        except BaseException as err:
            logging.error(
                f"ERROR IN CFModel::compute_loss:\n{err}")


def _get_serve_tf_examples_fn(model, tf_transform_output):
    try:
        model.tft_layer = tf_transform_output.transform_features_layer()

        @tf.function
        def serve_tf_examples_fn(serialized_tf_examples):
            try:
                feature_spec = tf_transform_output.raw_feature_spec()
                parsed_features = tf.io.parse_example(
                    serialized_tf_examples, feature_spec)
                transformed_features = model.tft_layer(parsed_features)

                return model(transformed_features)
            except BaseException as err:
                logging.error(f"ERROR IN serve_tf_examples_fn:\n{err}")

        return serve_tf_examples_fn

    except BaseException as err:
        logging.error(f"ERROR IN _get_serve_tf_examples_fn:\n{err}")


def _get_model(tf_transform_output):
    try:
        model = CFModel(tf_transform_output)
        model.compile(optimizer=keras.optimizers.Adam())

        return model
    except BaseException as err:
        logging.error(f"ERROR IN _get_model:\n{err}")


def run_fn(fn_args):
    try:
        # hyperparameters = fn_args.hyperparameters["values"]

        tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

        train_dataset = input_fn(
            fn_args.train_files, tf_transform_output)
        eval_dataset = input_fn(
            fn_args.eval_files, tf_transform_output)

        unique_user_ids = tf_transform_output.vocabulary_by_name(
            f"{FEATURE_KEYS[0]}_vocab")
        users_vocab_str = [i.decode() for i in unique_user_ids]

        unique_movie_ids = tf_transform_output.vocabulary_by_name(
            f"{FEATURE_KEYS[1]}_vocab")
        movies_vocab_str = [i.decode() for i in unique_movie_ids]

        model = _get_model(tf_transform_output)

        log_dir = os.path.join(os.path.dirname(
            fn_args.serving_model_dir), "logs")

        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

        early_stop_callbacks = keras.callbacks.EarlyStopping(
            monitor="val_root_mean_squared_error",
            mode="min",
            verbose=1,
            patience=10,
        )

        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            fn_args.serving_model_dir,
            monitor="val_root_mean_squared_error",
            mode="min",
            verbose=1,
            save_best_only=True,
        )

        callbacks = [
            tensorboard_callback,
            early_stop_callbacks,
            model_checkpoint_callback,
        ]
    except BaseException as err:
        logging.error(f"ERROR IN run_fn before fit:\n{err}")

    try:
        model.fit(
            train_dataset,
            epochs=5,
            steps_per_epoch=fn_args.train_steps,
            validation_data=eval_dataset,
            validation_steps=fn_args.eval_steps,
            callbacks=callbacks,
            verbose=1,
        )
    except BaseException as err:
        logging.error(f"ERROR IN run_fn during fit:\n{err}")

    try:
        signatures = {
            "serving_default": _get_serve_tf_examples_fn(
                model, tf_transform_output,
            ).get_concrete_function(
                tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
            )
        }

        model.save(
            fn_args.serving_model_dir,
            save_format="tf",
            signatures=signatures,
        )
    except BaseException as err:
        logging.error(f"ERROR IN run_fn during export:\n{err}")
