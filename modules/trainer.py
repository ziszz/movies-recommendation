import os

import keras
import tensorflow as tf
import tensorflow_transform as tft
from absl import logging
from keras import layers

from modules.transform import FEATURE_KEYS, transformed_name
from modules.tuner import input_fn

# class RecommenderNet(tf.keras.Model):
#     def __init__(self, unique_user_ids, unique_movie_ids, **kwargs) -> None:
#         super(RecommenderNet, self).__init__(**kwargs)

#         self.user_embeddings = self._build_user_model(
#             unique_user_ids,
#             128,
#             1e-3,
#         )
#         self.movie_embeddings = self._build_movie_model(
#             unique_movie_ids,
#             128,
#             1e-3,
#         )
#         self.user_bias = layers.Embedding(len(unique_user_ids), 1)
#         self.movie_bias = layers.Embedding(len(unique_movie_ids), 1)

#     def _build_user_model(self, unique_user_ids, embedding_dims, l2_regularizers):
#         try:
#             return keras.Sequential([
#                 layers.StringLookup(
#                     vocabulary=unique_user_ids,
#                     mask_token=None,
#                 ),
#                 layers.Embedding(
#                     len(unique_user_ids) + 1,
#                     embedding_dims,
#                     embeddings_initializer='he_normal',
#                     embeddings_regularizer=keras.regularizers.l2(
#                         l2_regularizers),
#                 )
#             ])
#         except BaseException as err:
#             logging.error(
#                 f"ERROR IN RecommenderNet::_build_user_model:\n{err}")

#     def _build_movie_model(self, unique_movie_ids, embedding_dims, l2_regularizers):
#         try:
#             return keras.Sequential([
#                 layers.StringLookup(
#                     vocabulary=unique_movie_ids,
#                     mask_token=None,
#                 ),
#                 layers.Embedding(
#                     len(unique_movie_ids) + 1,
#                     embedding_dims,
#                     embeddings_initializer='he_normal',
#                     embeddings_regularizer=keras.regularizers.l2(
#                         l2_regularizers),
#                 )
#             ])
#         except BaseException as err:
#             logging.error(
#                 f"ERROR IN RecommenderNet::_build_user_model:\n{err}")

#     def call(self, inputs):
#         try:
#             users_vector = self.user_embeddings(
#                 inputs[transformed_name(FEATURE_KEYS[0])])
#             users_bias = self.user_bias(
#                 inputs[transformed_name(FEATURE_KEYS[0])])
#             movies_vector = self.movie_embeddings(
#                 inputs[transformed_name(FEATURE_KEYS[1])])
#             movies_bias = self.movie_bias(
#                 inputs[transformed_name(FEATURE_KEYS[1])])

#             dot_user_movies = tf.tensordot(users_vector, movies_vector, 2)
#             x = dot_user_movies + users_bias + movies_bias

#             return tf.nn.sigmoid(x)
#         except BaseException as err:
#             logging.error(f"ERROR IN RecommenderNet::call:\n{err}")


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


def _get_model(unique_user_ids, unique_movie_ids):
    try:
        embedding_dims = 128
        l2_regularizers = 1e-3

        # user embedding
        user_input = layers.Input(shape=(1,), name=transformed_name(
            FEATURE_KEYS[0]), dtype=tf.int64)
        user_embedding = layers.Embedding(
            len(unique_user_ids) + 1,
            embedding_dims,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(
                l2_regularizers),
        )(user_input)
        user_vector = layers.Flatten()(user_embedding)

        # movie embedding
        movie_input = layers.Input(
            shape=(1,), name=transformed_name(FEATURE_KEYS[1]), dtype=tf.int64)
        movie_embedding = layers.Embedding(
            len(unique_movie_ids) + 1,
            embedding_dims,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(
                l2_regularizers),
        )(movie_input)
        movie_vector = layers.Flatten()(movie_embedding)

        concatenate = layers.concatenate([user_vector, movie_vector])

        deep = layers.Dense(128, activation=tf.nn.relu)(concatenate)
        deep = layers.Dense(64, activation=tf.nn.relu)(deep)
        output = layers.Dense(1)(deep)

        model = keras.Model([user_input, movie_input], output)

        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.BinaryCrossentropy(),
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
            fn_args.train_files, fn_args.data_accessor, tf_transform_output, batch_size=1)
        eval_dataset = input_fn(
            fn_args.eval_files, fn_args.data_accessor, tf_transform_output, batch_size=1)

        unique_user_ids = tf_transform_output.vocabulary_by_name(
            f"{FEATURE_KEYS[0]}_vocab")
        users_vocab_str = [i.decode() for i in unique_user_ids]

        unique_movie_ids = tf_transform_output.vocabulary_by_name(
            f"{FEATURE_KEYS[1]}_vocab")
        movies_vocab_str = [i.decode() for i in unique_movie_ids]

        model = _get_model(
            unique_user_ids=users_vocab_str,
            unique_movie_ids=movies_vocab_str,
        )

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
            )
        }

        model.save(
            fn_args.serving_model_dir,
            save_format="tf",
            signatures=signatures,
        )
    except BaseException as err:
        logging.error(f"ERROR IN run_fn during export:\n{err}")
