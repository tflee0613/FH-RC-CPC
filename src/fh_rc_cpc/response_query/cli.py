"""Verify or query published response requirements from an explicit data directory."""

import argparse
import json
import sys
from pathlib import Path

from . import analysis


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="extracted Public_Result_Data directory",
    )
    parser.add_argument(
        "--theta",
        type=float,
        help="nonnegative response regret tolerance; omit for full frontier",
    )
    args = parser.parse_args(argv)
    try:
        result = (
            analysis.verify(args.data_dir)
            if args.theta is None
            else analysis.query(args.data_dir, args.theta)
        )
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        print(f"response-retention: {error}", file=sys.stderr)
        return 2
    return 0
