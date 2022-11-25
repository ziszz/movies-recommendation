import tensorflow as tf

FEATURE_KEYS = [
<<<<<<< HEAD
    "title",
    "genres",
=======
    "title"
    "genres"
>>>>>>> cd8e42d6f1151079d8ed0d4cea55f2ba5e6076b1
]

def transformed_name(key):
    return f"{key}_xf"

def preprocessing_fn(inputs):
    outputs = {}
    
    for key in FEATURE_KEYS:
        outputs[transformed_name(key)] = tf.strings.lower(
            inputs[key]
<<<<<<< HEAD
        )  
=======
        )
    
>>>>>>> cd8e42d6f1151079d8ed0d4cea55f2ba5e6076b1
    return outputs
