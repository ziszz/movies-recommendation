import tensorflow as tf

FEATURE_KEY = "title"

def transformed_name(key):
    return f"{key}_xf"


def preprocessing_fn(inputs):
    outputs = {}

    for key in FEATURE_KEY:
        outputs[transformed_name(key)] = tf.strings.lower(
            inputs[key]
        )
    return outputs
