import json

from jev_prob_bench.adapters.semif import SemIfAdapter, _parse_semif_distribution


def test_parse_semif_distribution_from_lists():
    raw = {"option_ids": ["not_A", "A"], "probabilities": [0.8, 0.2]}
    assert _parse_semif_distribution(raw, ["A", "not_A"]) == {"A": 0.2, "not_A": 0.8}


def test_semif_adapter_invokes_cli_and_maps_probabilities(tmp_path):
    fake = tmp_path / "semif-score"
    fake.write_text(
        """
#!/usr/bin/env python3
import argparse, json
parser = argparse.ArgumentParser()
parser.add_argument('--mode')
parser.add_argument('--model')
parser.add_argument('--revision')
parser.add_argument('--input')
parser.add_argument('--output')
args = parser.parse_args()
row = json.loads(open(args.input).readline())
assert row['options'][0]['id'] == 'A'
result = {
    'id': row['id'],
    'option_ids': ['not_A', 'A'],
    'probabilities': [0.75, 0.25],
    'model': {'source': args.model, 'revision': args.revision},
}
open(args.output, 'w').write(json.dumps(result) + '\\n')
""".lstrip(),
        encoding="utf-8",
    )
    fake.chmod(0o755)
    adapter = SemIfAdapter(name="semif_test", command=str(fake), model="test/model", revision="abc")
    pred = adapter.predict_distribution(
        {"p_A": 0.25},
        "What is P(A)?",
        ["A", "not_A"],
        item={"instance_id": "item-1"},
    )
    assert pred == {"A": 0.25, "not_A": 0.75}
    assert adapter.last_metadata["model_version"] == "test/model@abc"
