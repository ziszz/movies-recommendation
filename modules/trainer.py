import os

import keras
import tensorflow as tf
import tensorflow_transform as tft
from absl import logging
from keras import layers

from modules.transform import FEATURE_KEYS, transformed_name


def gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")


def input_fn(file_pattern, tf_transform_output, batch_size=64):
    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
    )

    return dataset


def _get_serve_tf_examples_fn(model, tf_transform_output):
    try:
        model.tft_layer = tf_transform_output.transform_features_layer()

        @tf.function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        )
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


def _get_cf_model(unique_user_ids, unique_movie_ids):
    try:
        # hyperparameters
        embedding_dims = 128
        l2_regularizers = 1e-3

        # users embedding
        users_input = layers.Input(
            shape=(1,), name=transformed_name(FEATURE_KEYS[0]), dtype=tf.int64)
        users_embedding = layers.Embedding(
            len(unique_user_ids) + 1,
            embedding_dims,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(
                l2_regularizers),
        )(users_input)
        users_vector = layers.Flatten()(users_embedding)

        # movie embedding
        movies_input = layers.Input(
            shape=(1,), name=transformed_name(FEATURE_KEYS[1]), dtype=tf.int64)
        movies_embedding = layers.Embedding(
            len(unique_movie_ids) + 1,
            embedding_dims,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(
                l2_regularizers),
        )(movies_input)
        movies_vector = layers.Flatten()(movies_embedding)

        concatenate = tf.layers.concatenate([users_vector, movies_vector])

        deep = layers.Dense(128, activation='relu')(concatenate)
        deep = layers.Dense(32, activation='relu')(deep)
        outputs = layers.Dense(1)(deep)

        model = keras.Model(
            inputs=[users_input, movies_input], outputs=outputs)

        model.summary()

        model.compile(
            optimizers=keras.optimizers.Adagrad(learning_rate=0.1),
            loss=keras.losses.MeanSquaredError(),
            metrics=[keras.metrics.RootMeanSquaredError()],
        )

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
            f"{FEATURE_KEYS[0]}_vocab")
        movies_vocab_str = [i.decode() for i in unique_movie_ids]

        model = _get_cf_model(users_vocab_str, movies_vocab_str)

        log_dir = os.path.join(os.path.dirname(
            fn_args.serving_model_dir), "logs")

        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

        early_stop_callbacks = keras.callbacks.EarlyStopping(
            monitor="val_mean_squared_error",
            mode="min",
            verbose=1,
            patience=10,
        )

        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            fn_args.serving_model_dir,
            monitor="val_mean_squared_error",
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
            )
        }

        model.save(
            fn_args.serving_model_dir,
            save_format="tf",
            signatures=signatures,
        )
    except BaseException as err:
        logging.error(f"ERROR IN run_fn during export:\n{err}")
