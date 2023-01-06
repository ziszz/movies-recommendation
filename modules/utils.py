import os

import pandas as pd
import tensorflow as tf
from absl import logging


def transformed_name(key):
    return f"{key.lower()}_xf"


def create_str_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[bytes(value, "utf-8")]
        ),
    )


def create_int_feature(value):
    return tf.train.Feature(
        int64_list=tf.train.Int64List(
            value=[value]
        ),
    )


def create_float_feature(value):
    return tf.train.Feature(
        float_list=tf.train.FloatList(
            value=[value]
        ),
    )


def _gzip_reader_fn(filenames):
    try:
        return tf.data.TFRecordDataset(filenames, compression_type="GZIP")
    except BaseException as err:
        logging.error(f"ERROR IN _gzip_reader_fn:\n{err}")


def merge_dataset(data1_path: str, data2_path: str):
    try:
        df1 = pd.read_csv(data1_path)
        df2 = pd.read_csv(data2_path)

        df_final = df1.merge(df2)

        if not os.path.exists("data/merge"):
            os.makedirs("data/merge")

        df_final.to_csv("data/merge/movie_rating.csv", index=False)

        return "Merge dataset success"
    except BaseException as err:
        logging.error(f"ERROR IN merge_dataset:\n{err}")


def input_fn(file_pattern, tf_transform_output, batch_size=64, label_key=None):
    try:
        transform_feature_spec = (
            tf_transform_output.transformed_feature_spec().copy()
        )

        if label_key != None:
            dataset = tf.data.experimental.make_batched_features_dataset(
                file_pattern=file_pattern,
                batch_size=batch_size,
                features=transform_feature_spec,
                reader=_gzip_reader_fn,
                label_key=label_key,
            )
        else:
            dataset = tf.data.experimental.make_batched_features_dataset(
                file_pattern=file_pattern,
                batch_size=batch_size,
                features=transform_feature_spec,
                reader=_gzip_reader_fn,
            )

        return dataset
    except BaseException as err:
        logging.error(f"ERROR IN input_fn:\n{err}")
