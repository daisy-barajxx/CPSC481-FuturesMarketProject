#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path


def parse_payload(raw_text: str) -> list[dict]:
    payload = raw_text.strip()
    if not payload:
        raise ValueError("Input file is empty.")

    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Input is not valid JSON. Expected a JSON array or a quoted JSON string."
        ) from exc

    if isinstance(decoded, str):
        try:
            decoded = json.loads(decoded)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Outer JSON string was decoded, but inner content is not valid JSON."
            ) from exc

    if not isinstance(decoded, list):
        raise ValueError("Decoded payload is not a JSON array.")

    return decoded


def collect_headers(records: list[dict]) -> list[str]:
    headers: list[str] = []
    seen: set[str] = set()
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("Expected each record to be a JSON object.")
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                headers.append(key)
    return headers


def normalize_value(value):
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    return value


def write_csv(records: list[dict], output_path: Path) -> list[str]:
    headers = collect_headers(records)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in records:
            writer.writerow({k: normalize_value(row.get(k)) for k in headers})
    return headers


def write_jsonl(records: list[dict], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for row in records:
            if not isinstance(row, dict):
                raise ValueError("Expected each record to be a JSON object.")
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def default_output_path(input_path: Path, output_format: str) -> Path:
    suffix = "csv" if output_format == "csv" else "jsonl"
    return input_path.with_name(f"{input_path.stem}_parsed.{suffix}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Parse loaded_lob files stored as a quoted JSON blob and export as rows."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="loaded_lob_20250414__20250414_0921.csv",
        help="Input file path (default: loaded_lob_20250414__20250414_0921.csv).",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output file path. Defaults to <input_stem>_parsed.csv (or .jsonl).",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "jsonl"],
        default="csv",
        help="Output format (default: csv).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    output_path = (
        Path(args.output).expanduser()
        if args.output
        else default_output_path(input_path, args.format)
    )

    try:
        raw_text = input_path.read_text(encoding="utf-8")
        records = parse_payload(raw_text)
    except OSError as exc:
        print(f"Failed to read input file: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Parse error: {exc}", file=sys.stderr)
        return 1

    if not records:
        print("No records found in input payload.", file=sys.stderr)
        return 1

    try:
        if args.format == "csv":
            headers = write_csv(records, output_path)
            print(
                f"Parsed {len(records)} records to {output_path} with {len(headers)} columns."
            )
        else:
            write_jsonl(records, output_path)
            print(f"Parsed {len(records)} records to {output_path} as JSONL.")
    except OSError as exc:
        print(f"Failed to write output file: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Write error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
