from __future__ import annotations

import argparse

from jev_prob_bench.evaluation.jev_parallel_primitives import run_jev_parallel_primitives


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--parallelism", type=int, default=16)
    parser.add_argument("--endpoint")
    parser.add_argument("--api-key-env", default="JEV_API_KEY")
    parser.add_argument("--model", default="jev-latest")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--requests-per-second", type=float, default=20.0)
    args = parser.parse_args()
    run_jev_parallel_primitives(
        dataset_path=args.dataset,
        output_dir=args.output,
        repeats=args.repeats,
        parallelism=args.parallelism,
        endpoint=args.endpoint,
        api_key_env=args.api_key_env,
        model=args.model,
        timeout=args.timeout,
        retries=args.retries,
        requests_per_second=args.requests_per_second,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
