"""
Author: Yoimer Davila, Claude Code.
Created: 2024-04-17
Description: CLI for starting the Mask2Former training process with configurable parameters.
"""

import argparse
import logging
import tensorflow as tf
from config import Mask2FormerConfig
from train import train

# Set up logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Mask2Former Training CLI")

    # Path configurations
    parser.add_argument("--coco_root", type=str, help="Root directory for COCO dataset")
    parser.add_argument("--train_ann", type=str, help="Path to training annotations JSON")
    parser.add_argument("--tfrecord_dir", type=str, help="Directory containing training TFRecords")
    parser.add_argument("--model_path", type=str, default="./checkpoints", help="Path to model checkpoints")

    # Model architecture
    parser.add_argument("--backbone", type=str, choices=["resnet50", "mobilenetv4"], default="resnet50", help="Backbone type")
    parser.add_argument("--img_size", type=int, default=480, help="Image height and width (square)")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of transformer decoder layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--dim_ff", type=int, default=1024, help="Feed-forward network dimension")
    parser.add_argument("--input_channels", type=int, default=256, help="Transformer input channels")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0001, help="Initial learning rate")
    parser.add_argument("--warmup_steps", type=int, default=10000, help="Number of warmup steps")
    parser.add_argument("--save_iter", type=int, default=1, help="Save checkpoint every N epochs")

    # Advanced options
    parser.add_argument("--load_prev", action="store_true", help="Load previous model weights")
    parser.add_argument("--panoptic", action="store_true", help="Use panoptic dataset")
    parser.add_argument("--accum_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--use_accum", action="store_true", help="Enable gradient accumulation mode")
    parser.add_argument("--show_summary", action="store_true", help="Print model summary before training")
    parser.add_argument("--num_images", type=int, default=None, help="Limit number of images for dataset")

    args = parser.parse_args()

    # Initialize and override config
    cfg = Mask2FormerConfig()

    if args.coco_root: cfg.coco_root_path = args.coco_root
    if args.train_ann: cfg.train_annotation_path = args.train_ann
    if args.tfrecord_dir: cfg.tfrecord_dataset_directory_path = args.tfrecord_dir
    if args.model_path: cfg.model_path = args.model_path

    cfg.backbone_type = args.backbone
    cfg.img_height = args.img_size
    cfg.img_width = args.img_size
    cfg.num_decoder_layers = args.num_layers
    cfg.num_heads = args.num_heads
    cfg.dim_feedforward = args.dim_ff
    cfg.transformer_input_channels = args.input_channels

    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.warmup_steps = args.warmup_steps
    cfg.save_iter = args.save_iter

    cfg.load_previous_model = args.load_prev
    cfg.use_panoptic_dataset = args.panoptic
    cfg.accumulation_steps = args.accum_steps
    cfg.use_gradient_accumulation_steps = args.use_accum
    cfg.show_model_summary = args.show_summary
    cfg.number_images = args.num_images

    train(cfg)

if __name__ == '__main__':
    main()
