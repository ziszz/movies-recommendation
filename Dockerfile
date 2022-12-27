FROM tensorflow/serving:2.8.0

COPY ./outputs/serving_model/cbf_model /models/movies-cbf-model

ENV MODEL_NAME=movies-cbf-model