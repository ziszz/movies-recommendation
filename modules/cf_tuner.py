from typing import Any, Dict, NamedTuple, Text

import keras
import keras_tuner as kt
import tensorflow as tf
import tensorflow_transform as tft
from absl import logging
from keras import layers
from keras_tuner.engine import base_tuner

from modules.cf_transform import FEATURE_KEYS, LABEL_KEY
from modules.utils import input_fn, transformed_name

TunerFnResult = NamedTuple("TunerFnResult", [
    ("tuner", base_tuner.BaseTuner),
    ("fit_kwargs", Dict[Text, Any])
])

rmse_early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_root_mean_squared_error",
    mode="min",
    verbose=1,
    patience=10,
)

loss_early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    mode="min",
    verbose=1,
    patience=10,
)


def _get_model(hyperparameters, unique_user_ids, unique_movie_ids):
    try:
        # hyperparameters
        embedding_dims = hyperparameters.Int(
            "embedding_dims", min_value=16, max_value=1024, step=32)
        l2_regularizers = hyperparameters.Choice(
            "l2_regularizers", values=[1e-2, 1e-3, 1e-4, 1e-5])
        num_hidden_layers = hyperparameters.Choice(
            "num_hidden_layers", values=[1, 2, 3])
        dense_unit = hyperparameters.Int(
            "dense_unit", min_value=8, max_value=512, step=32)
        dropout_rate = hyperparameters.Float(
            "dropout_rate", min_value=0.1, max_value=0.8, step=0.1)
        learning_rate = hyperparameters.Choice(
            "learning_rate", values=[1e-2, 1e-3, 1e-4, 1e-5])

        # users embedding
        user_input = layers.Input(
            shape=(1,), name=transformed_name(FEATURE_KEYS[0]), dtype=tf.int64)
        users_embedding = layers.Embedding(
            len(unique_user_ids) + 1,
            embedding_dims,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(
                l2_regularizers),
        )(user_input)

        # movie embedding
        movie_input = layers.Input(
            shape=(1,), name=transformed_name(FEATURE_KEYS[1]), dtype=tf.int64)
        movies_embedding = layers.Embedding(
            len(unique_movie_ids) + 1,
            embedding_dims,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(
                l2_regularizers),
        )(movie_input)

        concatenate = layers.concatenate([users_embedding, movies_embedding])
        deep = layers.Dense(dense_unit, activation=tf.nn.relu)(concatenate)

        for _ in range(num_hidden_layers):
            deep = layers.Dense(dense_unit, activation=tf.nn.relu)(deep)
            deep = layers.Dropout(dropout_rate)(deep)

        outputs = layers.Dense(1)(deep)

        model = keras.Model(inputs=[user_input, movie_input], outputs=outputs)

        model.summary()

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=keras.losses.MeanSquaredError(),
            metrics=[keras.metrics.RootMeanSquaredError()],
        )

        return model
    except BaseException as err:
        logging.error(f"ERROR IN _get_model:\n{err}")


def tuner_fn(fn_args):
    try:
        tf_transform_output = tft.TFTransformOutput(
            fn_args.transform_graph_path)

        train_dataset = input_fn(
            fn_args.train_files[0], tf_transform_output, batch_size=128, label_key=transformed_name(LABEL_KEY))
        eval_dataset = input_fn(
            fn_args.eval_files[0], tf_transform_output, batch_size=128, label_key=transformed_name(LABEL_KEY))

        unique_user_ids = tf_transform_output.vocabulary_by_name(
            f"{FEATURE_KEYS[0]}_vocab")
        users_vocab_str = [b.decode() for b in unique_user_ids]

        unique_movie_ids = tf_transform_output.vocabulary_by_name(
            f"{FEATURE_KEYS[1]}_vocab")
        movies_vocab_str = [b.decode() for b in unique_movie_ids]

        tuner = kt.Hyperband(
            hypermodel=lambda hp: _get_model(
                hyperparameters=hp,
                unique_user_ids=users_vocab_str,
                unique_movie_ids=movies_vocab_str,
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
                "callbacks": [rmse_early_stop, loss_early_stop]
            },
        )
    except BaseException as err:
        logging.error(f"ERROR IN tuner_fn after tuning:\n{err}")
