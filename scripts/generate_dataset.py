from jev_prob_bench.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["generate", "--config", "config/benchmark.yaml", "--output", "datasets/generated/v0.1.0.jsonl"]))

