from jev_prob_bench.cli import _tiny_yaml


def test_tiny_yaml_handles_project_config_shape():
    config = _tiny_yaml(
        """
benchmark:
  version: "0.1.0"
  seed: 42
dataset:
  preset: "tiny"
  families:
    - explicit_probability
    - bayes
models:
  - oracle
"""
    )
    assert config["benchmark"]["seed"] == 42
    assert config["dataset"]["families"] == ["explicit_probability", "bayes"]

