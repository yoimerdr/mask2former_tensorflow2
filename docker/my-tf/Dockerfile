#
# This Dockerfile is configured for the default ResNet50 backbone.
# It installs essential dependencies needed for standard Mask2Former training.
#
# To use this Dockerfile, use the `docker build` command.
# See https://docs.docker.com/engine/reference/builder/
# for more information.
#
FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

# Python packages from PyPI
RUN python3 -m pip install --no-cache-dir \
    "opencv-python-headless==4.8.1.78" \
    "pycocotools" \
    "pillow"
