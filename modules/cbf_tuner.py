from typing import Any, Dict, NamedTuple, Text

import keras
import keras_tuner as kt
import tensorflow as tf
import tensorflow_transform as tft
from absl import logging
from keras import layers
from keras_tuner.engine import base_tuner

from modules.cbf_transform import CATEGORICAL_FEATURES, NUMERICAL_FEATURES
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


def _get_model(hyperparameters):
    try:
        # hyperparameters
        dense_unit1 = hyperparameters.Int(
            "dense_unit1", min_value=8, max_value=512, step=32)
        dense_unit2 = hyperparameters.Int(
            "dense_unit2", min_value=8, max_value=512, step=32)
        dense_unit3 = hyperparameters.Int(
            "dense_unit3", min_value=8, max_value=512, step=32)
        learning_rate = hyperparameters.Choice(
            "learning_rate", values=[1e-2, 1e-3, 1e-4, 1e-5])

        user_NN = keras.Sequential([
            layers.Dense(dense_unit1, activation=tf.nn.relu),
            layers.Dense(dense_unit2, activation=tf.nn.relu),
            layers.Dense(dense_unit3),
        ])

        movie_NN = keras.Sequential([
            layers.Dense(dense_unit1, activation=tf.nn.relu),
            layers.Dense(dense_unit2, activation=tf.nn.relu),
            layers.Dense(dense_unit3),
        ])

        # user neural network
        user_input = layers.Input(
            shape=(1,), name=transformed_name(NUMERICAL_FEATURES[0]))
        user_deep = user_NN(user_input)
        user_deep = tf.linalg.l2_normalize(user_deep, axis=1)

        # item neural network
        movie_features = []

        for key in [NUMERICAL_FEATURES[1], CATEGORICAL_FEATURES]:
            movie_features.append(
                layers.Input(shape=(1,), name=transformed_name(key))
            )

        concatenate = layers.concatenate(movie_features)
        movie_deep = movie_NN(concatenate)
        movie_deep = tf.linalg.l2_normalize(movie_deep, axis=1)

        outputs = layers.Dot(axes=1)([user_deep, movie_deep])

        inputs = [user_input, *movie_features]

        model = keras.Model(inputs=inputs, outputs=outputs)

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
            fn_args.train_files, tf_transform_output, batch_size=128)
        eval_dataset = input_fn(
            fn_args.eval_files, tf_transform_output, batch_size=128)

        tuner = kt.Hyperband(
            hypermodel=_get_model,
            objective=[
                kt.Objective(
                    "val_lounique_user_idss",
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
