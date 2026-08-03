"""Keras 3 loader for the legacy Keras 2.10 model_1702 HDF5 checkpoint.

Keras 3 cannot load the checkpoint directly: its four ``TFOpLambda`` layers
(``tf.compat.v1.squeeze`` on the level axis after each deep-supervision collapse conv) do not
exist in Keras 3, and their serialized ``inbound_nodes`` use a flat TF-op node format that the
legacy functional deserializer rejects. This module patches the stored ``model_config`` JSON —
swapping each ``TFOpLambda`` for the :class:`LevelSqueeze` layer and normalizing its inbound
nodes to the standard legacy format — then rebuilds the model and loads weights from the HDF5
``model_weights`` group, verifying that every stored weight was consumed.
"""

import copy
import json

import h5py
import keras
import numpy as np
from keras.src.legacy.saving import legacy_h5_format, saving_utils

TF_OP_LAMBDA_CLASS_NAME = "TFOpLambda"
EXPECTED_TF_OP_LAMBDA_COUNT = 4
EXPECTED_OUTPUT_CLASSES = 6
EXPECTED_OUTPUT_COUNT = 4


@keras.saving.register_keras_serializable(package="model_1702")
class LevelSqueeze(keras.layers.Layer):
    """Drop-in replacement for the legacy ``tf.compat.v1.squeeze`` TFOpLambda layers."""

    def __init__(self, axis: int, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        """Removes the configured axis from the input tensor."""
        return keras.ops.squeeze(inputs, axis=self.axis)

    def get_config(self) -> dict:
        """Returns the layer configuration."""
        config = super().get_config()
        config["axis"] = self.axis
        return config


def _extract_squeeze_axis(inbound_node: list) -> int:
    call_kwargs = inbound_node[3]
    axis = call_kwargs["axis"]
    if isinstance(axis, list):
        if len(axis) != 1:
            raise ValueError(f"Expected a single squeeze axis, found {axis}")
        axis = axis[0]
    return int(axis)


def patch_model_config(model_config: dict, expected_patch_count: int = EXPECTED_TF_OP_LAMBDA_COUNT) -> dict:
    """Rewrites TFOpLambda squeeze layers into LevelSqueeze layers with standard node format.

    Args:
        model_config: Decoded ``model_config`` JSON from a legacy HDF5 checkpoint.
        expected_patch_count: Number of TFOpLambda layers the config must contain (model_1702
            has four, one per deep-supervision head).

    Returns:
        A deep copy of the config with every TFOpLambda layer replaced. Safe to call on an
        already-patched config (no-op).

    Raises:
        ValueError: If the number of patched layers is neither 0 nor ``expected_patch_count``,
            or a TFOpLambda node has an unexpected structure.
    """
    patched = copy.deepcopy(model_config)
    patch_count = 0
    for layer in patched["config"]["layers"]:
        if layer["class_name"] != TF_OP_LAMBDA_CLASS_NAME:
            continue
        inbound_nodes = layer["inbound_nodes"]
        if len(inbound_nodes) != 1 or not isinstance(inbound_nodes[0][0], str):
            raise ValueError(f"Unexpected TFOpLambda inbound_nodes structure: {inbound_nodes}")
        flat_node = inbound_nodes[0]
        axis = _extract_squeeze_axis(flat_node)
        layer["class_name"] = "LevelSqueeze"
        layer["config"] = {"name": layer["name"], "axis": axis, "dtype": "float32", "trainable": True}
        layer["inbound_nodes"] = [[[flat_node[0], flat_node[1], flat_node[2], {}]]]
        patch_count += 1
    if patch_count not in (0, expected_patch_count):
        raise ValueError(f"Expected {expected_patch_count} TFOpLambda layers, patched {patch_count}")
    return patched


def _weight_dataset_names(weights_group: h5py.Group) -> list[str]:
    names = []
    for raw_layer_name in weights_group.attrs["layer_names"]:
        layer_name = raw_layer_name.decode() if isinstance(raw_layer_name, bytes) else raw_layer_name
        layer_group = weights_group[layer_name]
        for raw_weight_name in layer_group.attrs["weight_names"]:
            weight_name = raw_weight_name.decode() if isinstance(raw_weight_name, bytes) else raw_weight_name
            names.append(f"{layer_name}/{weight_name}")
    return names


def verify_weight_consumption(model: keras.Model, weights_group: h5py.Group) -> None:
    """Verifies every stored weight was loaded into the model.

    Args:
        model: Model rebuilt from the patched config with weights loaded.
        weights_group: The HDF5 ``model_weights`` group the weights were loaded from.

    Raises:
        ValueError: If the stored element count does not match the model's parameter count, or
            any stored weight tensor does not exactly match the corresponding model weight.
    """
    stored_names = _weight_dataset_names(weights_group)
    stored_elements = 0
    stored_values = {}
    for name in stored_names:
        layer_name = name.split("/")[0]
        dataset = weights_group[layer_name]["/".join(name.split("/")[1:])]
        stored_elements += int(np.prod(dataset.shape))
        stored_values[name] = np.asarray(dataset)

    model_params = model.count_params()
    if stored_elements != model_params:
        raise ValueError(f"HDF5 stores {stored_elements} weight elements but model has {model_params} parameters")

    layers_by_name = {layer.name: layer for layer in model.layers}
    for name, stored in stored_values.items():
        layer = layers_by_name[name.split("/")[0]]
        suffix = name.split("/")[-1].removesuffix(":0")
        matches = [w for w in layer.weights if w.name.split("/")[-1].removesuffix(":0") == suffix]
        if len(matches) != 1:
            raise ValueError(f"Could not uniquely match stored weight {name} in layer {layer.name}")
        loaded = np.asarray(matches[0])
        if loaded.shape != stored.shape or not np.array_equal(loaded, stored):
            raise ValueError(f"Loaded weight {name} does not match stored values")


def load_legacy_h5(h5_path: str, expected_squeeze_count: int = EXPECTED_TF_OP_LAMBDA_COUNT) -> keras.Model:
    """Rebuilds a legacy TFOpLambda-bearing HDF5 model under Keras 3 and loads its weights.

    Args:
        h5_path: Path to a legacy Keras 2.x HDF5 checkpoint.
        expected_squeeze_count: Number of TFOpLambda squeeze layers the config must contain.

    Returns:
        The rebuilt functional model with all stored weights loaded and verified.

    Raises:
        ValueError: If config patching, weight loading, or weight verification fails.
    """
    with h5py.File(h5_path, "r") as h5_file:
        raw_config = h5_file.attrs["model_config"]
        model_config = json.loads(raw_config if isinstance(raw_config, str) else raw_config.decode())
        patched = patch_model_config(model_config, expected_patch_count=expected_squeeze_count)
        with keras.saving.custom_object_scope({"LevelSqueeze": LevelSqueeze}):
            model = saving_utils.model_from_config(patched)
            legacy_h5_format.load_weights_from_hdf5_group(h5_file["model_weights"], model)
        verify_weight_consumption(model, h5_file["model_weights"])
    return model


def load_model_1702(h5_path: str) -> keras.Model:
    """Loads the legacy model_1702 HDF5 checkpoint under Keras 3.

    Args:
        h5_path: Path to ``model_1702.h5``.

    Returns:
        The rebuilt functional model with all legacy weights loaded. Outputs are the four
        deep-supervision softmax heads; index 0 (``sup1_softmax``) is the head used for
        prediction, shaped (batch, dim0, dim1, 6).

    Raises:
        ValueError: If config patching, weight loading, or output-structure verification fails.
    """
    model = load_legacy_h5(h5_path)

    if len(model.outputs) != EXPECTED_OUTPUT_COUNT:
        raise ValueError(f"Expected {EXPECTED_OUTPUT_COUNT} deep-supervision outputs, found {len(model.outputs)}")
    for output in model.outputs:
        if output.shape[-1] != EXPECTED_OUTPUT_CLASSES:
            raise ValueError(f"Expected {EXPECTED_OUTPUT_CLASSES} output classes, found {output.shape[-1]}")
    return model
