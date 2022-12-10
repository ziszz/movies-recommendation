import os
from typing import Dict, Text

import keras
import tensorflow as tf
import tensorflow_recommenders as tfrs
import tensorflow_transform as tft
from absl import logging

from modules.transform import (CATEGORICAL_FEATURE, NUMERIC_FEATURE,
                               transformed_name)
from modules.tuner import input_fn


class RecommenderNet(tfrs.models.Model):
    def __init__(self, tf_transform_output, movies_data):
        super().__init__()
        embedding_dimension = 32

        self.user_model: tf.keras.layers.Layer = self._build_user_model(
            tf_transform_output, embedding_dimension, 1e-3)
        self.movie_model: tf.keras.layers.Layer = self._build_movie_model(
            tf_transform_output, embedding_dimension, 1e-3)

        self.retrieval_task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=movies_data.batch(128).map(self.movie_model)
            )
        )

    def _build_user_model(self, tf_transform_output, embedding_dims, l2_regularizers):
        try:
            unique_user_ids = tf_transform_output.vocabulary_by_name(
                f"{NUMERIC_FEATURE}_vocab")
            users_vocab_str = [b.decode() for b in unique_user_ids]

            return tf.keras.Sequential([
                tf.keras.layers.StringLookup(
                    vocabulary=users_vocab_str, mask_token=None),
                tf.keras.layers.Embedding(
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
            movies_vocab_str = [b.decode() for b in unique_movie_ids]

            return tf.keras.Sequential([
                tf.keras.layers.StringLookup(
                    vocabulary=movies_vocab_str, mask_token=None),
                tf.keras.layers.Embedding(
                    len(movies_vocab_str) + 1,
                    embedding_dims,
                    embeddings_initializer='he_normal',
                    embeddings_regularizer=keras.regularizers.l2(
                        l2_regularizers),
                )
            ])
        except BaseException as err:
            logging.error(
                f"ERROR IN RecommenderNet::_build_movie_model:\n{err}")

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False):
        try:
            user_embeddings = tf.squeeze(self.user_model(
                features[transformed_name(NUMERIC_FEATURE)]), axis=1)
            positive_movie_embeddings = self.movie_model(
                features[transformed_name(CATEGORICAL_FEATURE)])

            return self.retrieval_task(user_embeddings, positive_movie_embeddings)
        except BaseException as err:
            logging.error(f"ERROR IN RecommenderNet::compute_loss:\n{err}")


def _get_serve_tf_examples_fn(model, tf_transform_output):
    try:
        model.tft_layer = tf_transform_output.transform_features_layer()

        @tf.function(input_signature=[
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples"),
        ])
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


def _get_model(tf_transform_output, movies_data):
    try:
        model = RecommenderNet(
            tf_transform_output=tf_transform_output,
            movies_data=movies_data,
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

        return model
    except BaseException as err:
        logging.error(f"ERROR IN _get_model:\n{err}")


def run_fn(fn_args):
    try:
        # hyperparameters = fn_args.hyperparameters["values"]

        log_dir = os.path.join(os.path.dirname(
            fn_args.serving_model_dir), "logs")

        tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

        train_dataset = input_fn(
            fn_args.train_files, fn_args.data_accessor, tf_transform_output, batch_size=1)
        eval_dataset = input_fn(
            fn_args.eval_files, fn_args.data_accessor, tf_transform_output, batch_size=1)

        movies_data = train_dataset.map(
            lambda f: f[transformed_name(CATEGORICAL_FEATURE)])

        model = _get_model(
            tf_transform_output=tf_transform_output,
            movies_data=movies_data,
        )

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

        # early_stop_callbacks = tf.keras.callbacks.EarlyStopping(
        #     monitor="val_root_mean_squared_error",
        #     mode="min",
        #     verbose=1,
        #     patience=10,
        # )

        # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        #     fn_args.serving_model_dir,
        #     monitor="val_root_mean_squared_error",
        #     mode="min",
        #     verbose=1,
        #     save_best_only=True,
        # )

        callbacks = [
            tensorboard_callback,
        ]
    except BaseException as err:
        logging.error(f"ERROR IN run_fn before fit:\n{err}")

    try:
        model.fit(
            train_dataset,
            epochs=fn_args.custom_config["epochs"],
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
            )
        }

        model.save(
            fn_args.serving_model_dir,
            save_format="tf",
            signatures=signatures,
        )
    except BaseException as err:
        logging.error(f"ERROR IN run_fn during export:\n{err}")
