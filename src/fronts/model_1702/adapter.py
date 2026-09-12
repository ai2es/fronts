"""Adapters exposing the 2.0 model calling convention for legacy and 6-class models.

``fronts.evaluate.compute_stats`` calls ``model(x, training=False)`` on raw
(batch, lat, lon, level, variable) volume batches and compares predictions against 6-class
one-hot targets. :class:`FrontFinder1702Adapter` bridges model_1702's legacy conventions
(pre-normalized inputs, longitude-major orientation with latitude descending, 6 output
classes) to that contract; :class:`ClassPaddingAdapter` bridges other 6-class 2.0 checkpoints.

model_1702 is evaluated with a single full-domain forward pass rather than the legacy 288x128
tile stitching: the network is fully convolutional, a single pass gives it strictly more
spatial context than tiling (which, if anything, favors it), and the CONUS domain is exactly
its 288x128 training image size anyway.
"""

import keras
import numpy as np
import tensorflow as tf

from fronts.model_1702 import normalization

NUM_EVAL_CLASSES = 6
NUM_LEGACY_CLASSES = 6


class FrontFinder1702Adapter:
    """Wraps model_1702 to accept raw side-store batches and emit 6-class predictions.

    Input contract: (batch, lat, lon, level, variable) float32 in legacy units, with the level
    and variable axes ordered per ``normalization.LEVELS`` / ``normalization.VARIABLES``.
    Output contract: (batch, lat, lon, 6) — the sup1 softmax head, unpadded.
    """

    def __init__(self, model: keras.Model, lat_ascending: bool):
        self.model = model
        self.lat_ascending = lat_ascending
        maxs, mins = normalization.norm_min_max_tables()
        self.mins = tf.constant(mins)
        self.scales = tf.constant(maxs - mins)

    def __call__(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Runs model_1702 on a raw volume batch.

        Args:
            x: Batch shaped (batch, lat, lon, level, variable) in legacy units.
            training: Accepted for signature compatibility; inference only.

        Returns:
            Tensor shaped (batch, lat, lon, 6).
        """
        x = tf.cast(x, tf.float32)
        x = (x - self.mins) / self.scales
        x = tf.where(tf.math.is_finite(x), x, tf.zeros_like(x))
        if self.lat_ascending:
            x = tf.reverse(x, axis=[1])
        x = tf.transpose(x, [0, 2, 1, 3, 4])
        preds = self.model(x, training=False)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        preds = tf.transpose(preds, [0, 2, 1, 3])
        if self.lat_ascending:
            preds = tf.reverse(preds, axis=[1])
        return tf.pad(preds, [[0, 0], [0, 0], [0, 0], [0, NUM_EVAL_CLASSES - NUM_LEGACY_CLASSES]])


class ClassPaddingAdapter:
    """Pads a 2.0 model's class axis up to ``NUM_EVAL_CLASSES``, a no-op for 6-class checkpoints."""

    def __init__(self, model: keras.Model):
        self.model = model

    def __call__(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Runs the wrapped model and pads its class axis to ``NUM_EVAL_CLASSES``.

        Args:
            x: Batch in the wrapped model's native input format.
            training: Accepted for signature compatibility; inference only.

        Returns:
            Tensor with the last axis padded to ``NUM_EVAL_CLASSES`` classes.
        """
        preds = self.model(x, training=False)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        missing = NUM_EVAL_CLASSES - int(preds.shape[-1])
        if missing < 0:
            raise ValueError(f"Model emits {int(preds.shape[-1])} classes, more than the {NUM_EVAL_CLASSES} expected")
        if missing == 0:
            return preds
        return tf.pad(preds, [[0, 0], [0, 0], [0, 0], [0, missing]])


def normalize_volume(x: np.ndarray) -> np.ndarray:
    """Applies legacy min-max normalization to a raw volume array (numpy convenience mirror).

    Args:
        x: Array shaped (..., level, variable) in legacy units.

    Returns:
        Normalized array with non-finite values replaced by zero, as in the legacy pipeline.
    """
    maxs, mins = normalization.norm_min_max_tables()
    return np.nan_to_num((x - mins) / (maxs - mins), nan=0.0, posinf=0.0, neginf=0.0)
