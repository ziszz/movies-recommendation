import os

import keras
import tensorflow as tf
import tensorflow_transform as tft
from absl import logging
from keras import layers

from modules.utils import input_fn


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


def _get_model():
    try:
        user_NN = keras.Sequential([
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(64, activation=tf.nn.relu),
            layers.Dense(32),
        ])

        movie_NN = keras.Sequential([
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(64, activation=tf.nn.relu),
            layers.Dense(32),
        ])

        # user neural network
        user_input = layers.Input(shape=(1,))
        user_deep = user_NN(user_input)
        user_deep = tf.linalg.l2_normalize(user_deep, axis=1)

        # item neural network
        movie_input = layers.Input(shape=(2,))
        movie_deep = movie_NN(movie_input)
        movie_deep = tf.linalg.l2_normalize(movie_deep, axis=1)

        outputs = layers.Dot(axes=1)([user_deep, movie_deep])

        model = keras.Model(inputs=[user_input, movie_input], outputs=outputs)

        model.summary()

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss=keras.losses.MeanSquaredError(),
            metrics=[keras.metrics.RootMeanSquaredError()],
        )

        return model
    except BaseException as err:
        logging.error(f"ERROR IN _get_model:\n{err}")


def run_fn(fn_args):
    try:
        hyperparameters = fn_args.hyperparameters["values"]

        tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

        train_dataset = input_fn(
            fn_args.train_files, tf_transform_output, batch_size=128)
        eval_dataset = input_fn(
            fn_args.eval_files, tf_transform_output, batch_size=128)

        model = _get_model(hyperparameters)

        log_dir = os.path.join(os.path.dirname(
            fn_args.serving_model_dir), "logs")

        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

        rmse_early_stop_callbacks = keras.callbacks.EarlyStopping(
            monitor="val_root_mean_squared_error",
            mode="min",
            verbose=1,
            patience=10,
        )

        loss_early_stop_callbacks = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            verbose=1,
            patience=10,
        )

        callbacks = [
            tensorboard_callback,
            rmse_early_stop_callbacks,
            loss_early_stop_callbacks,
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
