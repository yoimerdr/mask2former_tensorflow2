"""
Author: Pavel Timonin
Created: 2026-03-22
Description: This script contains backbone factory functions for Mask2Former.
"""

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model


# Supported backbone types
BACKBONE_RESNET50 = "resnet50"
BACKBONE_MOBILENETV4 = "mobilenetv4"

SUPPORTED_BACKBONES = [BACKBONE_RESNET50, BACKBONE_MOBILENETV4]


def get_resnet50_backbone(
    input_shape: tuple = (480, 480, 3),
) -> tf.keras.Model:
    """
    Build a ResNet50 backbone and return feature maps C2, C3, C4, C5.

    Args:
        input_shape (tuple): Input image shape as
            (height, width, channels). Defaults to (480, 480, 3).

    Returns:
        tf.keras.Model: Model with inputs=image tensor and
            outputs=[c2, c3, c4, c5] feature maps.
    """
    backbone = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )
    backbone.trainable = True

    c2_output = backbone.get_layer("conv2_block3_out").output
    c3_output = backbone.get_layer("conv3_block4_out").output
    c4_output = backbone.get_layer("conv4_block6_out").output
    c5_output = backbone.get_layer("conv5_block3_out").output

    return Model(
        inputs=backbone.input,
        outputs=[c2_output, c3_output, c4_output, c5_output],
        name="resnet50_backbone",
    )


def get_mobilenetv4_backbone(
    input_shape: tuple = (480, 480, 3),
) -> tf.keras.Model:
    """
    Build a MobileNetV4ConvMedium backbone returning 4 feature maps.

    Uses ``tf-models-official`` MobileNet with the
    MobileNetV4ConvMedium architecture. Extracts outputs at strides
    4, 8, 16, and 32 (analogous to C2–C5 of ResNet50) with channel
    counts 48, 80, 160, and 256 respectively.

    Args:
        input_shape (tuple): Input image shape as
            (height, width, channels). Defaults to (480, 480, 3).

    Returns:
        tf.keras.Model: Model with inputs=image tensor and
            outputs=[c2, c3, c4, c5] feature maps at strides
            4/8/16/32.
    """
    from official.vision.modeling.backbones.mobilenet import MobileNet

    input_tensor = tf.keras.Input(shape=input_shape)

    mobilenet = MobileNet(
        model_id="MobileNetV4ConvMedium",
        filter_size_scale=1.0,
        input_specs=tf.keras.layers.InputSpec(
            shape=(None,) + input_shape
        ),
    )

    # MobileNet returns a dict keyed by level string ("2","3","4","5").
    endpoints = mobilenet(input_tensor, training=True)

    c2 = endpoints["2"]  # stride 4, 48 channels
    c3 = endpoints["3"]  # stride 8, 80 channels
    c4 = endpoints["4"]  # stride 16, 160 channels
    c5 = endpoints["5"]  # stride 32, 256 channels

    return Model(
        inputs=input_tensor,
        outputs=[c2, c3, c4, c5],
        name="mobilenetv4_backbone",
    )


def get_backbone(
    backbone_type: str = BACKBONE_RESNET50,
    input_shape: tuple = (480, 480, 3),
) -> tf.keras.Model:
    """
    Factory function to create the requested backbone.

    Args:
        backbone_type (str): Backbone identifier. One of
            ``"resnet50"`` or ``"mobilenetv4"``.
            Defaults to ``"resnet50"``.
        input_shape (tuple): Input image shape as
            (height, width, channels). Defaults to (480, 480, 3).

    Returns:
        tf.keras.Model: Backbone model with outputs
            [c2, c3, c4, c5].

    Raises:
        ValueError: If ``backbone_type`` is not supported.
    """
    if backbone_type == BACKBONE_RESNET50:
        return get_resnet50_backbone(input_shape=input_shape)
    elif backbone_type == BACKBONE_MOBILENETV4:
        return get_mobilenetv4_backbone(input_shape=input_shape)
    else:
        raise ValueError(
            f"Unsupported backbone_type='{backbone_type}'. "
            f"Choose from {SUPPORTED_BACKBONES}."
        )


def get_preprocess_fn(backbone_type: str = BACKBONE_RESNET50):
    """
    Return the appropriate preprocessing function for the backbone.

    ResNet50 uses ``tf.keras.applications.resnet50.preprocess_input``
    (BGR, zero-centered with ImageNet means). MobileNetV4 expects
    simple ``[0, 1]`` scaled RGB.

    Args:
        backbone_type (str): Backbone identifier. One of
            ``"resnet50"`` or ``"mobilenetv4"``.
            Defaults to ``"resnet50"``.

    Returns:
        callable: A function ``f(image) -> preprocessed_image``
            operating on float32 tensors in ``[0, 255]``.

    Raises:
        ValueError: If ``backbone_type`` is not supported.
    """
    if backbone_type == BACKBONE_RESNET50:
        return tf.keras.applications.resnet50.preprocess_input
    elif backbone_type == BACKBONE_MOBILENETV4:
        return _mobilenetv4_preprocess
    else:
        raise ValueError(
            f"Unsupported backbone_type='{backbone_type}'. "
            f"Choose from {SUPPORTED_BACKBONES}."
        )


def _mobilenetv4_preprocess(image: tf.Tensor) -> tf.Tensor:
    """
    Preprocess image for MobileNetV4 by scaling to [0, 1].

    Args:
        image (tf.Tensor): Input image tensor in [0, 255] range,
            shape [H, W, 3] or [B, H, W, 3].

    Returns:
        tf.Tensor: Scaled image tensor in [0, 1] range.
    """
    return tf.cast(image, tf.float32) / 255.0
