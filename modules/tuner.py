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


class RecommenderModel(tf.keras.Model):
    def __init__(self, hyperparameters, tf_transform_output, movies_uri):
        super().__init__()

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

        movies_artifact = movies_uri.get()[0]
        input_dir = artifact_utils.get_split_uri([movies_artifact], "train")
        movie_files = glob.glob(os.path.join(input_dir, "*"))
        movies = tf.data.TFRecordDataset(movie_files, compression_type="GZIP")
        movies_dataset = self._extract_str_feature(movies, "title")

        self.user_model = self._build_user_model(
            tf_transform_output,
            embedding_dims,
            l2_regularizers
        )

        self.movie_model = self._build_movie_model(
            tf_transform_output,
            embedding_dims,
            l2_regularizers
        )

        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=movies_dataset.batch(128).map(self.movie_model)
            )
        )

    def _extract_str_feature(self, dataset, feature_name):
        try:
            np_dataset = []
            for example in dataset:
                np_example = example_coder.ExampleToNumpyDict(example.numpy())
                np_dataset.append(np_example[feature_name][0].decode())
            return tf.data.Dataset.from_tensor_slices(np_dataset)
        except BaseException as err:
            logging.error(
                f"ERROR IN RecommenderModel::_extract_str_feature:\n{err}")

    def _build_user_model(self, tf_transform_output, embedding_dims, l2_regularizers):
        try:
            unique_user_ids = tf_transform_output.vocabulary_by_name(
                f"{FEATURE_KEYS[0]}_vocab")
            users_vocab_str = [b.decode() for b in unique_user_ids]

            model = tf.keras.Sequential([
                layers.Input(shape=(1,), name=transformed_name(
                    FEATURE_KEYS[0]), dtype=tf.int64),
                layers.Lambda(lambda x: tf.as_string(x)),
                layers.StringLookup(
                    vocabulary=self.unique_user_ids, mask_token=None),
                tf.keras.layers.Embedding(
                    len(users_vocab_str) + 1,
                    embedding_dims,
                    embeddings_initializer='he_normal',
                    embeddings_regularizer=keras.regularizers.l2(
                        l2_regularizers),
                ),
            ])

            return model
        except BaseException as err:
            logging.error(
                f"ERROR IN RecommenderModel::_build_user_model:\n{err}")

    def _build_movie_model(self, tf_transform_output, embedding_dims, l2_regularizers):
        try:
            unique_movie_ids = tf_transform_output.vocabulary_by_name(
                f"{FEATURE_KEYS[1]}_vocab")
            movies_vocab_str = [b.decode() for b in unique_movie_ids]

            model = tf.keras.Sequential([
                layers.Input(shape=(1,), name=transformed_name(
                    FEATURE_KEYS[1]), dtype=tf.int64),
                layers.Lambda(lambda x: tf.as_string(x)),
                layers.StringLookup(
                    vocabulary=self.unique_movie_ids, mask_token=None),
                tf.keras.layers.Embedding(
                    len(movies_vocab_str) + 1,
                    embedding_dims,
                    embeddings_initializer='he_normal',
                    embeddings_regularizer=keras.regularizers.l2(
                        l2_regularizers),
                ),
            ])

            return model
        except BaseException as err:
            logging.error(
                f"ERROR IN RecommenderModel::_build_movie_model:\n{err}")

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        try:
            user_embeddings = tf.squeeze(self.user_model(
                features[transformed_name(FEATURE_KEYS[0])]))
            positive_movie_embeddings = tf.squeeze(
                self.movie_model(features[transformed_name(FEATURE_KEYS[1])]))

            return self.task(user_embeddings, positive_movie_embeddings)
        except BaseException as err:
            logging.error(f"ERROR IN RecommenderModel::compute_loss:\n{err}")


def _gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")


def input_fn(file_pattern, tf_transform_output, num_epochs, batch_size=64):
    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=_gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY),
    )

    return dataset


def _get_model(hyperparameters, tf_transform_output, rating_weight, retrieval_weight, movies_uri):
    try:
        learning_rate = hyperparameters.Choice(
            "learning_rate",
            values=[1e-2, 1e-3, 1e-4]
        )
        model = RecommenderModel(
            hyperparameters,
            tf_transform_output,
            movies_uri,
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adagrad(learning_rate=learning_rate))
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
