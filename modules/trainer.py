import glob
import os
from typing import Dict, Text

import keras
import tensorflow as tf
import tensorflow_recommenders as tfrs
import tensorflow_transform as tft
from absl import logging
from keras import layers
from tfx.types import artifact_utils
from tfx_bsl.coders import example_coder

from modules.transform import FEATURE_KEYS, transformed_name
from modules.tuner import input_fn


class RecommenderModel(tf.keras.Model):
    def __init__(self, hyperparameters, tf_transform_output, rating_weight, retrieval_weight, movies_uri):
        super().__init__()

        rating_layers = []

        for _ in range(hyperparameters["num_hidden_layers"]):
            rating_layers.append(layers.Dense(
                hyperparameters["dense_unit"], activation=tf.nn.relu))
            rating_layers.append(layers.Dropout(
                hyperparameters["dropout_rate"]))

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
                hyperparameters["embedding_dims"],
                embeddings_initializer='he_normal',
                embeddings_regularizer=keras.regularizers.l2(
                    hyperparameters["l2_regularizers"]),
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
                hyperparameters["embedding_dims"],
                embeddings_initializer='he_normal',
                embeddings_regularizer=keras.regularizers.l2(
                    hyperparameters["l2_regularizers"]),
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


def _get_model(hyperparameters, tf_transform_output):
    try:
        model = RecommenderModel(hyperparameters, tf_transform_output)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=hyperparameters["learning_rate"]))
        return model
    except BaseException as err:
        logging.error(f"ERROR IN get_model:\n{err}")


def run_fn(fn_args):
    try:
        hyperparameters = fn_args.hyperparameters["values"]

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

        model = _get_model(hyperparameters, tf_transform_output)

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

    except BaseException as err:
        logging.error(f"ERROR IN run_fn before fit:\n{err}")

    try:
        model.fit(
            train_dataset,
            epochs=hyperparameters["tuner/epochs"],
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
