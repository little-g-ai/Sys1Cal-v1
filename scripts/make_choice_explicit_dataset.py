from __future__ import annotations

import argparse

from jev_prob_bench.dataset_variants import make_choice_explicit_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    make_choice_explicit_dataset(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

