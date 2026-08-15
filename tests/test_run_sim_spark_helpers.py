#!/usr/bin/env python3
"""Unit tests for lightweight GPT-OSS Spark helper logic."""

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

try:
    from strategic_anatomy import runtime as run_sim_spark
except ModuleNotFoundError:
    run_sim_spark = None


class FakeModel:
    def __init__(self, hf_device_map=None, device=None):
        self.hf_device_map = hf_device_map
        self.device = device
        self._param = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        yield self._param


@unittest.skipIf(torch is None or run_sim_spark is None, "torch/transformers stack not installed")
class ParseGeneratedMoveTests(unittest.TestCase):
    def test_parses_decision_prefix(self):
        move = run_sim_spark.parse_generated_move('Decision: "A"', ["A", "B"])
        self.assertEqual(move, "A")

    def test_parses_choice_verb(self):
        move = run_sim_spark.parse_generated_move("I choose b because it is safer.", ["A", "B"])
        self.assertEqual(move, "B")

    def test_parses_bare_label(self):
        move = run_sim_spark.parse_generated_move("  *a*  ", ["A", "B"])
        self.assertEqual(move, "A")

    def test_returns_none_for_reasoning_only(self):
        move = run_sim_spark.parse_generated_move(
            "We need to parse the problem carefully before deciding.",
            ["A", "B"],
        )
        self.assertIsNone(move)


@unittest.skipIf(torch is None or run_sim_spark is None, "torch/transformers stack not installed")
class ResolveModelInputDeviceTests(unittest.TestCase):
    def test_prefers_first_cuda_device_in_hf_device_map(self):
        model = FakeModel(hf_device_map={"model.embed_tokens": "cpu", "model.layers.0": "cuda:1"})
        self.assertEqual(run_sim_spark.resolve_model_input_device(model), torch.device("cuda:1"))

    def test_handles_integer_hf_device_map_targets(self):
        model = FakeModel(hf_device_map={"": 0}, device=torch.device("cpu"))
        self.assertEqual(run_sim_spark.resolve_model_input_device(model), torch.device("cuda:0"))

    def test_falls_back_to_model_device_when_no_cuda_target_exists(self):
        model = FakeModel(hf_device_map={"model.layers.0": "cpu"}, device=torch.device("cuda:0"))
        self.assertEqual(run_sim_spark.resolve_model_input_device(model), torch.device("cuda:0"))


@unittest.skipIf(torch is None or run_sim_spark is None, "torch/transformers stack not installed")
class ResolveLoadDeviceMapTests(unittest.TestCase):
    def test_cuda0_maps_to_single_gpu_dict(self):
        self.assertEqual(run_sim_spark._resolve_load_device_map("cuda0"), {"": 0})

    def test_auto_is_preserved(self):
        self.assertEqual(run_sim_spark._resolve_load_device_map("auto"), "auto")

    def test_invalid_map_raises(self):
        with self.assertRaises(ValueError):
            run_sim_spark._resolve_load_device_map("balanced")


@unittest.skipIf(torch is None or run_sim_spark is None, "torch/transformers stack not installed")
class BnbInt8ParamsCompatPatchTests(unittest.TestCase):
    def test_int8params_constructor_ignores_hf_private_attrs(self):
        try:
            import bitsandbytes as bnb
        except ModuleNotFoundError:
            self.skipTest("bitsandbytes not installed")

        run_sim_spark._install_bnb_int8params_accelerate_compat_patch()
        param = bnb.nn.Int8Params(
            torch.zeros(2, 2),
            requires_grad=False,
            _is_hf_initialized=True,
        )
        self.assertEqual(type(param).__name__, "Int8Params")


if __name__ == "__main__":
    unittest.main()
