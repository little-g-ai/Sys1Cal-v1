from __future__ import annotations

import argparse

from jev_prob_bench.parallel_primitives import make_parallel_primitive_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    make_parallel_primitive_dataset(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
