FROM tensorflow/serving:2.8.0

COPY ./outputs/serving_model /models/movie-cf-model

ENV MODEL_NAME=movie-cf-model
ENV PORT=8501

RUN echo '#!/bin/bash \n\n\
    env\n\
    tensorflow_model_server --port=8500 --rest_api_port=${PORT} \
    --model_name=${MODEL_NAME}} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} \
    "$@"' > /usr/bin/tf_serving_entrypoint.sh \
    && chmod +x /usr/bin/tf_serving_entrypoint.sh