FROM tensorflow/serving:2.8.0 as cf

COPY ../../outputs/serving_model/cf_model /
COPY ./outputs/serving_model/cf_model /models/movies-cf-model

ENV MODEL_NAME=movies-cf-model

FROM tensorflow/serving:2.8.0 as cbf

COPY ../../outputs/serving_model/cbf_model /
COPY ./outputs/serving_model/cbf_model /models/movies-cbf-model

ENV MODEL_NAME=movies-cbf-model