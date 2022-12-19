import tensorflow as tf
from absl import logging

from modules.transform import LABEL_KEY, transformed_name


def _gzip_reader_fn(filenames):
    try:
        return tf.data.TFRecordDataset(filenames, compression_type="GZIP")
    except BaseException as err:
        logging.error(f"ERROR IN _gzip_reader_fn:\n{err}")


def input_fn(file_pattern, tf_transform_output, batch_size=64):
    try:
        transform_feature_spec = (
            tf_transform_output.transformed_feature_spec().copy()
        )

        dataset = tf.data.experimental.make_batched_features_dataset(
            file_pattern=file_pattern,
            batch_size=batch_size,
            features=transform_feature_spec,
            reader=_gzip_reader_fn,
            label_key=transformed_name(LABEL_KEY),
        )

        return dataset
    except BaseException as err:
        logging.error(f"ERROR IN input_fn:\n{err}")
