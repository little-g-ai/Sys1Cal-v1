from jev_prob_bench.cli import main


if __name__ == "__main__":
    raise SystemExit(
        main(
            [
                "run",
                "--dataset",
                "datasets/generated/v0.1.0.jsonl",
                "--models",
                "config/benchmark.yaml",
                "--output",
                "results/run_001",
            ]
        )
    )

