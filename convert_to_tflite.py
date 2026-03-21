"""
Author: Pavel Timonin
Created: 2026-03-09
Description: This script converts the trained Mask2Former model to a TFLite format.
"""

import logging
import os

import tensorflow as tf

from coco_dataset_optimized import COCOAnalysis
from config import Mask2FormerConfig
from model_functions import Mask2FormerModel


FORMAT = "%(asctime)s-%(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def convert_mask2former_to_tflite(output_path: str = "mask2former.tflite") -> None:
    """
    Convert a trained Mask2Former model to TFLite format.

    Loads the configuration, instantiates the model, restores the latest
    checkpoint, traces the forward pass with a fixed input signature,
    and converts it to a .tflite file using SELECT_TF_OPS.

    Args:
        output_path (str): The file path where the .tflite model will be saved.

    Returns:
        None

    Raises:
        FileNotFoundError: If no checkpoint is found in the path specified by config.
    """
    cfg = Mask2FormerConfig()

    logger.info("Initializing configuration and COCO analysis...")
    if cfg.use_panoptic_dataset:
        coco_info = COCOAnalysis(cfg.panoptic_train_annotation_path)
    else:
        coco_info = COCOAnalysis(cfg.train_annotation_path)

    num_classes = coco_info.get_num_classes()
    img_height, img_width = cfg.img_height, cfg.img_width

    logger.info(
        f"Building model with input shape (1, {img_height}, {img_width}, 3)..."
    )
    model = Mask2FormerModel(
        input_shape=(img_height, img_width, 3),
        transformer_input_channels=256,
        num_classes=num_classes,
        num_queries=100,
        num_decoder_layers=6,
        num_heads=8,
        dim_feedforward=1024,
    )
    model.build((1, img_height, img_width, 3))

    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint_path = cfg.test_model_path
    if os.path.isdir(checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)

    if checkpoint_path:
        checkpoint.restore(checkpoint_path).expect_partial()
        logger.info(f"Successfully restored weights from {checkpoint_path}")
    else:
        logger.error(f"No checkpoint found in {cfg.test_model_path}")
        raise FileNotFoundError(f"No checkpoint found in {cfg.test_model_path}")

    logger.info("Tracing the forward pass for TFLite conversion...")

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[1, img_height, img_width, 3], dtype=tf.float32)
        ]
    )
    def model_forward(images: tf.Tensor) -> tuple:
        """
        Trace forward pass returning only the main predictions.

        Args:
            images (tf.Tensor): Input tensor of shape [B, H, W, C].

        Returns:
            tuple: A tuple containing:
                - pred_logits (tf.Tensor): Class scores.
                - pred_masks (tf.Tensor): Mask predictions.
        """
        pred_logits, pred_masks, _ = model(images, training=False)
        return pred_logits, pred_masks

    concrete_func = model_forward.get_concrete_function()

    logger.info("Setting up TFLiteConverter...")
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    logger.info("Converting the model. This may take a while...")
    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    logger.info(f"TFLite model successfully saved to {output_path}")


if __name__ == "__main__":
    convert_mask2former_to_tflite()
