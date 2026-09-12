"""Tests for the legacy model_1702 HDF5 loader."""

import copy
import json
import os

import h5py
import keras
import numpy as np
import pytest
from keras.src.legacy.saving import legacy_h5_format, saving_utils

from fronts.model_1702 import loader

MODEL_1702_PATH = os.path.expanduser("~/data/fronts/model_1702.h5")

TF_OP_LAMBDA_LAYER = {
    "class_name": "TFOpLambda",
    "config": {
        "name": "tf.compat.v1.squeeze_3",
        "trainable": True,
        "dtype": "float32",
        "function": "compat.v1.squeeze",
    },
    "name": "tf.compat.v1.squeeze_3",
    "inbound_nodes": [["sup1_Conv3D_collapse", 0, 0, {"axis": [3], "name": None}]],
}


def _wrap_layers(layers: list[dict], input_name: str, output_name: str) -> dict:
    return {
        "class_name": "Functional",
        "config": {
            "name": "test_model",
            "layers": layers,
            "input_layers": [[input_name, 0, 0]],
            "output_layers": [[output_name, 0, 0]],
        },
    }


def _synthetic_legacy_config() -> dict:
    layers = [
        {
            "class_name": "InputLayer",
            "config": {
                "batch_input_shape": [None, 8, 8, 3, 2],
                "dtype": "float32",
                "sparse": False,
                "name": "Input",
            },
            "name": "Input",
            "inbound_nodes": [],
        },
        {
            "class_name": "Conv3D",
            "config": {
                "name": "sup1_Conv3D_collapse",
                "trainable": True,
                "dtype": "float32",
                "filters": 6,
                "kernel_size": [1, 1, 3],
                "strides": [1, 1, 1],
                "padding": "valid",
                "data_format": "channels_last",
                "dilation_rate": [1, 1, 1],
                "groups": 1,
                "activation": "linear",
                "use_bias": True,
                "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": 7}},
                "bias_initializer": {"class_name": "Zeros", "config": {}},
                "kernel_regularizer": None,
                "bias_regularizer": None,
                "activity_regularizer": None,
                "kernel_constraint": None,
                "bias_constraint": None,
            },
            "name": "sup1_Conv3D_collapse",
            "inbound_nodes": [[["Input", 0, 0, {}]]],
        },
        copy.deepcopy(TF_OP_LAMBDA_LAYER),
        {
            "class_name": "Activation",
            "config": {"name": "sup1_softmax", "trainable": True, "dtype": "float32", "activation": "softmax"},
            "name": "sup1_softmax",
            "inbound_nodes": [[["tf.compat.v1.squeeze_3", 0, 0, {}]]],
        },
    ]
    return _wrap_layers(layers, "Input", "sup1_softmax")


@pytest.fixture
def synthetic_legacy_h5(tmp_path):
    config = _synthetic_legacy_config()
    patched = loader.patch_model_config(config, expected_patch_count=1)
    with keras.saving.custom_object_scope({"LevelSqueeze": loader.LevelSqueeze}):
        reference_model = saving_utils.model_from_config(patched)
    h5_path = tmp_path / "legacy_model.h5"
    with h5py.File(h5_path, "w") as h5_file:
        h5_file.attrs["model_config"] = json.dumps(config)
        weights_group = h5_file.create_group("model_weights")
        legacy_h5_format.save_weights_to_hdf5_group(weights_group, reference_model)
    return str(h5_path), reference_model


class TestPatchModelConfig:
    def test_patches_verbatim_tf_op_lambda_entry(self):
        config = _wrap_layers([copy.deepcopy(TF_OP_LAMBDA_LAYER)], "Input", "tf.compat.v1.squeeze_3")
        patched = loader.patch_model_config(config, expected_patch_count=1)
        layer = patched["config"]["layers"][0]
        assert layer["class_name"] == "LevelSqueeze"
        assert layer["config"]["axis"] == 3
        assert layer["config"]["name"] == "tf.compat.v1.squeeze_3"
        assert layer["inbound_nodes"] == [[["sup1_Conv3D_collapse", 0, 0, {}]]]

    def test_original_config_not_mutated(self):
        config = _wrap_layers([copy.deepcopy(TF_OP_LAMBDA_LAYER)], "Input", "tf.compat.v1.squeeze_3")
        loader.patch_model_config(config, expected_patch_count=1)
        assert config["config"]["layers"][0]["class_name"] == "TFOpLambda"

    def test_idempotent_on_already_patched_config(self):
        config = _wrap_layers([copy.deepcopy(TF_OP_LAMBDA_LAYER)], "Input", "tf.compat.v1.squeeze_3")
        patched_once = loader.patch_model_config(config, expected_patch_count=1)
        patched_twice = loader.patch_model_config(patched_once, expected_patch_count=1)
        assert patched_twice == patched_once

    def test_wrong_patch_count_raises(self):
        config = _wrap_layers([copy.deepcopy(TF_OP_LAMBDA_LAYER)], "Input", "tf.compat.v1.squeeze_3")
        with pytest.raises(ValueError, match="Expected 4 TFOpLambda layers"):
            loader.patch_model_config(config)

    def test_multi_axis_squeeze_raises(self):
        bad_layer = copy.deepcopy(TF_OP_LAMBDA_LAYER)
        bad_layer["inbound_nodes"] = [["sup1_Conv3D_collapse", 0, 0, {"axis": [2, 3], "name": None}]]
        config = _wrap_layers([bad_layer], "Input", "tf.compat.v1.squeeze_3")
        with pytest.raises(ValueError, match="single squeeze axis"):
            loader.patch_model_config(config, expected_patch_count=1)


class TestLevelSqueeze:
    def test_squeezes_configured_axis(self):
        layer = loader.LevelSqueeze(axis=3)
        result = layer(np.zeros((2, 4, 4, 1, 6), dtype=np.float32))
        assert tuple(result.shape) == (2, 4, 4, 6)

    def test_config_round_trip(self):
        layer = loader.LevelSqueeze(axis=3, name="squeeze_test")
        rebuilt = loader.LevelSqueeze.from_config(layer.get_config())
        assert rebuilt.axis == 3
        assert rebuilt.name == "squeeze_test"


class TestSyntheticRoundTrip:
    def test_load_matches_reference_predictions(self, synthetic_legacy_h5):
        h5_path, reference_model = synthetic_legacy_h5
        loaded = loader.load_legacy_h5(h5_path, expected_squeeze_count=1)
        x = np.random.default_rng(11).random((2, 8, 8, 3, 2)).astype(np.float32)
        np.testing.assert_array_equal(
            np.asarray(loaded(x)),
            np.asarray(reference_model(x)),
        )

    def test_weight_verification_detects_tampering(self, synthetic_legacy_h5):
        h5_path, _ = synthetic_legacy_h5
        loaded = loader.load_legacy_h5(h5_path, expected_squeeze_count=1)
        conv = next(layer for layer in loaded.layers if layer.name == "sup1_Conv3D_collapse")
        conv.kernel.assign(np.asarray(conv.kernel) + 1.0)
        with h5py.File(h5_path, "r") as h5_file, pytest.raises(ValueError, match="does not match stored values"):
            loader.verify_weight_consumption(loaded, h5_file["model_weights"])

    def test_softmax_output(self, synthetic_legacy_h5):
        h5_path, _ = synthetic_legacy_h5
        loaded = loader.load_legacy_h5(h5_path, expected_squeeze_count=1)
        x = np.random.default_rng(3).random((1, 8, 8, 3, 2)).astype(np.float32)
        probs = np.asarray(loaded(x))
        np.testing.assert_allclose(probs.sum(axis=-1), 1.0, rtol=1e-5)


@pytest.mark.skipif(not os.path.exists(MODEL_1702_PATH), reason="model_1702.h5 not available")
class TestRealModel1702:
    @pytest.fixture(scope="class")
    def model_1702(self):
        return loader.load_model_1702(MODEL_1702_PATH)

    def test_structure(self, model_1702):
        assert len(model_1702.layers) == 112
        assert model_1702.count_params() == 8745968
        assert len(model_1702.outputs) == 4
        assert all(output.shape[-1] == 6 for output in model_1702.outputs)

    def test_prediction_is_valid_softmax(self, model_1702):
        x = np.random.default_rng(0).random((1, 64, 64, 5, 10)).astype(np.float32)
        preds = model_1702.predict(x, verbose=0)
        assert preds[0].shape == (1, 64, 64, 6)
        np.testing.assert_allclose(preds[0].sum(axis=-1), 1.0, rtol=1e-5)
