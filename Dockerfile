FROM tensorflow/serving:2.8.0

COPY ./outputs/serving_model/cf_model /models/movie-cf-model

ENV MODEL_NAME=movie-cf-model