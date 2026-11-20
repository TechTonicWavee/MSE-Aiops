import argparse
import json
from pathlib import Path

from mlops_monitoring.workflow import run_workflow


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the synthetic model monitoring and retraining workflow."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for generated datasets, models, and reports (default: current directory).",
    )
    args = parser.parse_args()
    result = run_workflow(args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
