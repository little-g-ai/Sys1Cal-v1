from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from jev_prob_bench import __version__
from jev_prob_bench.evaluation.runner import run_benchmark
from jev_prob_bench.generators import GENERATOR_REGISTRY
from jev_prob_bench.generators.base import PRESET_COUNTS
from jev_prob_bench.reporting.report import build_report
from jev_prob_bench.schemas import validate_distribution


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jev-prob-bench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--config", default="config/benchmark.yaml")
    generate_parser.add_argument("--output", required=True)
    generate_parser.add_argument("--preset", choices=PRESET_COUNTS)
    generate_parser.add_argument("--count-per-family", type=int)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--dataset")
    run_parser.add_argument("--models", default="config/models.example.yaml")
    run_parser.add_argument("--output")
    run_parser.add_argument("--resume")
    run_parser.add_argument("--repeats", type=int)

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--results", required=True)
    report_parser.add_argument("--output", required=True)

    args = parser.parse_args(argv)
    if args.command == "generate":
        config = load_config(args.config)
        preset = args.preset or config.get("dataset", {}).get("preset", "dev")
        count = args.count_per_family or PRESET_COUNTS[preset]
        generate_dataset(config, args.output, count)
        return 0
    if args.command == "run":
        if args.resume:
            output_dir = Path(args.resume)
            dataset = output_dir / "dataset.path"
            if not dataset.exists():
                raise SystemExit("--resume requires dataset.path in the run directory")
            dataset_path = dataset.read_text(encoding="utf-8").strip()
        else:
            if not args.dataset or not args.output:
                raise SystemExit("run requires --dataset and --output, or --resume")
            dataset_path = args.dataset
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "dataset.path").write_text(str(Path(dataset_path).resolve()), encoding="utf-8")
        config = load_config(args.models)
        model_configs = config.get("models", [])
        repeats = args.repeats or config.get("evaluation", {}).get("repeats", 1)
        run_benchmark(dataset_path, model_configs, output_dir, repeats=repeats, resume=True)
        return 0
    if args.command == "report":
        build_report(args.results, args.output)
        return 0
    return 1


def generate_dataset(config: dict[str, Any], output_path: str | Path, count_per_family: int) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    version = config.get("benchmark", {}).get("version", __version__)
    seed = int(config.get("benchmark", {}).get("seed", 42))
    families = config.get("dataset", {}).get("families") or list(GENERATOR_REGISTRY)
    with output.open("w", encoding="utf-8") as handle:
        for family in families:
            generator_cls = GENERATOR_REGISTRY[family]
            generator = generator_cls(version=version, seed=seed)
            for item in generator.generate(count_per_family):
                validate_distribution(item.gold_distribution, item.outcomes)
                handle.write(json.dumps(item.__dict__, sort_keys=True) + "\n")
    validate_dataset(output)
    return output


def validate_dataset(path: str | Path) -> None:
    by_latent: dict[str, dict[str, float]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            validate_distribution(item["gold_distribution"], item["outcomes"])
            latent = item["latent_instance_id"]
            if latent in by_latent and by_latent[latent] != item["gold_distribution"]:
                raise ValueError(f"representation mismatch for {latent}")
            by_latent[latent] = item["gold_distribution"]


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return json.loads(text)
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text) or {}
    except ModuleNotFoundError:
        return _tiny_yaml(text)


def _tiny_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any, str | None]] = [(-1, root, None)]
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip())
        line = raw_line.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if line.startswith("- "):
            value = _parse_scalar(line[2:])
            if not isinstance(parent, list):
                list_key = stack[-1][2]
                grandparent = stack[-2][1] if len(stack) > 1 else None
                if list_key is None or not isinstance(grandparent, dict):
                    raise ValueError("list item without key")
                new_list: list[Any] = []
                grandparent[list_key] = new_list
                stack[-1] = (stack[-1][0], new_list, list_key)
                parent = new_list
            parent.append(value)
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        if value.strip():
            parent[key] = _parse_scalar(value.strip())
        else:
            parent[key] = {}
            stack.append((indent, parent[key], key))
    return root


def _parse_scalar(value: str) -> Any:
    value = value.strip().strip('"')
    if value in {"true", "false"}:
        return value == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


if __name__ == "__main__":
    raise SystemExit(main())

