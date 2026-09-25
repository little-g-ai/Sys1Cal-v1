import json

from jev_prob_bench.cli import generate_dataset
from jev_prob_bench.evaluation.runner import run_benchmark


def test_oracle_pipeline(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    config = {
        "benchmark": {"version": "0.1.0", "seed": 3},
        "dataset": {"families": ["explicit_probability"], "preset": "tiny"},
    }
    generate_dataset(config, dataset, count_per_family=2)
    result_path = run_benchmark(dataset, [{"name": "oracle", "adapter": "mock", "strategy": "oracle"}], tmp_path / "run")
    rows = [json.loads(line) for line in result_path.read_text().splitlines()]
    assert rows
    assert all(row["metrics"]["tv"] == 0 for row in rows)



def test_runner_saves_each_repeat_as_a_distinct_result(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    config = {
        "benchmark": {"version": "0.1.0", "seed": 3},
        "dataset": {"families": ["explicit_probability"], "preset": "tiny"},
    }
    generate_dataset(config, dataset, count_per_family=1)
    item_count = sum(1 for line in dataset.read_text().splitlines() if line.strip())
    result_path = run_benchmark(
        dataset,
        [{"name": "oracle", "adapter": "mock", "strategy": "oracle"}],
        tmp_path / "run_repeats",
        repeats=3,
    )
    rows = [json.loads(line) for line in result_path.read_text().splitlines()]
    assert len(rows) == item_count * 3
    assert sorted({row["repeat_index"] for row in rows}) == [0, 1, 2]
    assert all(row["metrics"]["tv"] == 0 for row in rows)

    run_benchmark(
        dataset,
        [{"name": "oracle", "adapter": "mock", "strategy": "oracle"}],
        tmp_path / "run_repeats",
        repeats=3,
    )
    resumed_rows = [json.loads(line) for line in result_path.read_text().splitlines()]
    assert len(resumed_rows) == len(rows)
