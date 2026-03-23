#
# This Dockerfile is configured specifically for the MobileNetV4 backbone.
# It installs additional dependencies like tf-models-official and applies patches
# required to run the lightweight backbone smoothly.
#
# To use this Dockerfile, use the `docker build` command.
# See https://docs.docker.com/engine/reference/builder/
# for more information.
#
FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

# 1. Force TensorFlow to use Legacy Keras (Keras 2)
# tf-models-official relies heavily on tf_keras, which is required for TF 2.16+
ENV TF_USE_LEGACY_KERAS=1

# 2. Install pure Python/system dependencies
RUN python3 -m pip install --no-cache-dir \
    "opencv-python-headless==4.8.1.78" \
    "pycocotools" \
    "pillow" \
    "kaggle" \
    "psutil" \
    "pandas" \
    "gin-config" \
    "pyyaml" \
    "matplotlib" \
    "tensorflow-datasets"

# 3. Install tf-models-official and patch the TPU import
RUN python3 -m pip install --no-cache-dir --no-deps "tf-models-official" && \
    sed -i 's/from tensorflow.python.tpu import embedding_context_utils as ecu/ecu = None/g' /usr/local/lib/python3.12/dist-packages/orbit/utils/loop_fns.py
