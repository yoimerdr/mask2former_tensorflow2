"""
Author: Pavel Timonin
Created: 2025-03-08
Description: PointRend for Mask2Former model.
"""

import tensorflow as tf

"""
Shape shorthand in this module:

    B: batch size (number of images)
    H: height
    W: width
    C: number of channels
    P: number of points
"""


@tf.function
def point_sample(input_tensor: tf.Tensor, point_coords: tf.Tensor, align_corners: bool = False) -> tf.Tensor:
    """
    Sample features at continuous coordinates via bilinear interpolation.

    A native TensorFlow wrapper to support point sampling on a feature map.
    Assumes `point_coords` lie inside [0, 1] x [0, 1] square.

    Args:
        input_tensor (tf.Tensor): A tensor of shape [B, H, W, C] containing features.
        point_coords (tf.Tensor): A tensor of shape [B, P, 2] containing
            [0, 1] x [0, 1] normalized point coordinates (w, h).
        align_corners (bool): If True, aligns the centers of the 4 corner pixels
            of the input and output tensors, preserving the values at the corner pixels.
            Defaults to False.

    Returns:
        tf.Tensor: A tensor of shape [B, P, C] containing features for points in
            `point_coords`. Obtained via bilinear interpolation.
    """
    shape = tf.shape(input_tensor)
    H = shape[1]
    W = shape[2]

    x = point_coords[:, :, 0]
    y = point_coords[:, :, 1]

    H_f = tf.cast(H, tf.float32)
    W_f = tf.cast(W, tf.float32)

    if align_corners:
        x = x * (W_f - 1.0)
        y = y * (H_f - 1.0)
    else:
        x = x * W_f - 0.5
        y = y * H_f - 0.5

    x = tf.clip_by_value(x, 0.0, W_f - 1.0)
    y = tf.clip_by_value(y, 0.0, H_f - 1.0)

    x0_pos = tf.math.floor(x)
    x1_pos = x0_pos + 1.0
    y0_pos = tf.math.floor(y)
    y1_pos = y0_pos + 1.0

    x0_c = tf.clip_by_value(x0_pos, 0.0, W_f - 1.0)
    x1_c = tf.clip_by_value(x1_pos, 0.0, W_f - 1.0)
    y0_c = tf.clip_by_value(y0_pos, 0.0, H_f - 1.0)
    y1_c = tf.clip_by_value(y1_pos, 0.0, H_f - 1.0)

    x0_i = tf.cast(x0_c, tf.int32)
    x1_i = tf.cast(x1_c, tf.int32)
    y0_i = tf.cast(y0_c, tf.int32)
    y1_i = tf.cast(y1_c, tf.int32)

    idx_00 = tf.stack([y0_i, x0_i], axis=-1)
    idx_01 = tf.stack([y0_i, x1_i], axis=-1)
    idx_10 = tf.stack([y1_i, x0_i], axis=-1)
    idx_11 = tf.stack([y1_i, x1_i], axis=-1)

    val_00 = tf.gather_nd(input_tensor, idx_00, batch_dims=1)
    val_01 = tf.gather_nd(input_tensor, idx_01, batch_dims=1)
    val_10 = tf.gather_nd(input_tensor, idx_10, batch_dims=1)
    val_11 = tf.gather_nd(input_tensor, idx_11, batch_dims=1)

    w_00 = tf.expand_dims((x1_pos - x) * (y1_pos - y), axis=-1)
    w_01 = tf.expand_dims((x - x0_pos) * (y1_pos - y), axis=-1)
    w_10 = tf.expand_dims((x1_pos - x) * (y - y0_pos), axis=-1)
    w_11 = tf.expand_dims((x - x0_pos) * (y - y0_pos), axis=-1)

    return val_00 * w_00 + val_01 * w_01 + val_10 * w_10 + val_11 * w_11


@tf.function
def generate_regular_grid_point_coords(num_regions: int, side_size: int) -> tf.Tensor:
    """
    Generate regular square grid of points in [0, 1] x [0, 1] coordinate space.

    Args:
        num_regions (int): The number of grids to sample, one for each region.
        side_size (int): The side size of the regular grid.

    Returns:
        tf.Tensor: A tensor of shape [num_regions, side_size^2, 2] that contains coordinates
            for the regular grids.
    """
    step = 1.0 / tf.cast(side_size, tf.float32)
    half_step = step / 2.0

    x = tf.linspace(half_step, 1.0 - half_step, side_size)
    y = tf.linspace(half_step, 1.0 - half_step, side_size)

    X, Y = tf.meshgrid(x, y)

    grid = tf.stack([X, Y], axis=-1)
    grid = tf.reshape(grid, [-1, 2])
    grid = tf.expand_dims(grid, axis=0)

    return tf.tile(grid, [num_regions, 1, 1])


@tf.function
def get_uncertain_point_coords_with_randomness(
        coarse_logits: tf.Tensor,
        num_points: int,
        oversample_ratio: float,
        importance_sample_ratio: float
) -> tf.Tensor:
    """
    Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty.

    The uncertainties are calculated for each point using the logit prediction.
    Mask2Former typically uses -abs(logits) for binary mask uncertainty.

    Args:
        coarse_logits (tf.Tensor): A tensor of shape [B, Hmask, Wmask, C] or
            [B, Hmask, Wmask, 1] for class-specific or class-agnostic prediction.
        num_points (int): The number of points P to sample.
        oversample_ratio (float): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled via importance sampling.

    Returns:
        tf.Tensor: A tensor of shape [B, P, 2] that contains the coordinates of P
            sampled points.
    """
    shape = tf.shape(coarse_logits)
    num_boxes = shape[0]
    num_sampled = tf.cast(tf.cast(num_points, tf.float32) * oversample_ratio, tf.int32)

    point_coords = tf.random.uniform(shape=[num_boxes, num_sampled, 2])
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)

    point_uncertainties = -tf.abs(point_logits)

    num_uncertain_points = tf.cast(importance_sample_ratio * tf.cast(num_points, tf.float32), tf.int32)
    num_random_points = num_points - num_uncertain_points

    point_uncertainties_squeeze = tf.squeeze(point_uncertainties, axis=-1)
    _, idx = tf.math.top_k(point_uncertainties_squeeze, k=num_uncertain_points)

    uncertain_points = tf.gather(point_coords, idx, batch_dims=1)

    if num_random_points > 0:
        random_points = tf.random.uniform(shape=[num_boxes, num_random_points, 2])
        point_coords = tf.concat([uncertain_points, random_points], axis=1)
    else:
        point_coords = uncertain_points

    return point_coords


@tf.function
def get_uncertain_point_coords_on_grid(uncertainty_map: tf.Tensor, num_points: int) -> tuple:
    """
    Find `num_points` most uncertain points from `uncertainty_map` grid.

    Args:
        uncertainty_map (tf.Tensor): A tensor of shape [B, H, W, 1] that contains uncertainty
            values for a set of points on a regular H x W grid.
        num_points (int): The number of points P to select.

    Returns:
        tuple:
            - point_indices (tf.Tensor): A tensor of shape [B, P] that contains indices from
              [0, H x W) of the most uncertain points.
            - point_coords (tf.Tensor): A tensor of shape [B, P, 2] that contains [0, 1] x [0, 1] normalized
              coordinates of the most uncertain points from the H x W grid.
    """
    shape = tf.shape(uncertainty_map)
    B = shape[0]
    H = shape[1]
    W = shape[2]

    H_f = tf.cast(H, tf.float32)
    W_f = tf.cast(W, tf.float32)
    h_step = 1.0 / H_f
    w_step = 1.0 / W_f

    num_points = tf.minimum(H * W, num_points)
    uncertainty_map = tf.reshape(uncertainty_map, [B, H * W])

    _, point_indices = tf.math.top_k(uncertainty_map, k=num_points)

    point_indices_f = tf.cast(point_indices, tf.float32)

    point_x = w_step / 2.0 + tf.math.floormod(point_indices_f, W_f) * w_step
    point_y = h_step / 2.0 + tf.math.floordiv(point_indices_f, W_f) * h_step

    point_coords = tf.stack([point_x, point_y], axis=-1)
    return point_indices, point_coords
