import os
from typing import Dict, Text

import keras
import tensorflow as tf
import tensorflow_recommenders as tfrs
import tensorflow_transform as tft
from absl import logging
from keras import layers

from modules.transform import FEATURE_KEYS, LABEL_KEY, transformed_name
from modules.tuner import get_model, input_fn


class RankingModel(tf.keras.Model):
    def __init__(self, tf_transform_output):
        super().__init__()
        self.embedding_dims = 128

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
                self.embedding_dims,
                embeddings_initializer='he_normal',
                embeddings_regularizer=keras.regularizers.l2(1e-3),
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
                self.embedding_dims,
                embeddings_initializer='he_normal',
                embeddings_regularizer=keras.regularizers.l2(1e-3),
            ),
        ])

        self.ratings = tf.keras.Sequential([
            layers.Dense(1024, activation=tf.nn.relu),
            layers.Dense(512, activation=tf.nn.relu),
            layers.Dense(1),
        ])

    def call(self, inputs):
        try:
            user_id, movie_id = inputs

            user_embedding = self.user_embeddings(user_id)
            movie_embedding = self.movie_embeddings(movie_id)

            return self.ratings(tf.concat([user_embedding, movie_embedding], axis=2))
        except BaseException as err:
            logging.error(f"ERROR IN RankingModel::call:\n{err}")


class RecommenderModel(tfrs.Model):
    def __init__(self, tf_transform_output):
        super().__init__()
        self.ranking_model = RankingModel(tf_transform_output)
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


def run_fn(fn_args):
    try:
        log_dir = os.path.join(os.path.dirname(
            fn_args.serving_model_dir), "logs")

        tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

        train_dataset = input_fn(
            fn_args.train_files,
            fn_args.data_accessor,
            tf_transform_output,
            batch_size=128,
        )
        eval_dataset = input_fn(
            fn_args.eval_files,
            fn_args.data_accessor,
            tf_transform_output,
            batch_size=128,
        )

        model = get_model(tf_transform_output)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

        early_stop_callbacks = tf.keras.callbacks.EarlyStopping(
            monitor="val_root_mean_squared_error",
            mode="min",
            verbose=1,
            patience=10,
        )
        
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            fn_args.serving_model_dir,
            monitor="val_root_mean_squared_error",
            mode="min",
            verbose=1,
            save_best_only=True,
        )
        
        callbacks = [
            tensorboard_callback,
            early_stop_callbacks,
            model_checkpoint_callback
        ]

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))

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
