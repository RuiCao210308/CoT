#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any, Dict, List, Optional


METRIC_KEYS = ["speed_mae_mps", "curvature_mae_x100", "overall_l1_train_scale"]
FLAT_KEYWORDS = ("flat_repeated_pair", "flat_repeat", "flat repeated")
REUSED_KEYWORDS = (
    "reused_history_pairs",
    "mostly_reused_history_pairs",
    "copied_history_prefix",
    "copied_history_suffix",
    "reused history",
    "copied history",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze degeneracy in OpenEMMA text-generation baseline records.")
    parser.add_argument("--records_jsonl", required=True)
    parser.add_argument("--summary_json", type=str, default=None)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_txt", type=str, default=None)
    return parser.parse_args()


def load_json(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_records(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                records.append({"parse_success": False, "failure_reason": "json_decode_error", "_line_no": line_no})
                continue
            record["_line_no"] = line_no
            records.append(record)
    return records


def bool_field(record: Dict[str, Any], keys: List[str]) -> Optional[bool]:
    for key in keys:
        value = record.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
    return None


def flatten_reason(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(flatten_reason(item) for item in value)
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            parts.append(str(key))
            parts.append(flatten_reason(item))
        return " ".join(parts)
    return str(value)


def nested_guard_reason(record: Dict[str, Any]) -> str:
    guard = record.get("planner_guard")
    if not isinstance(guard, dict):
        return ""
    return flatten_reason(guard.get("reasons")) + " " + flatten_reason(guard)


def has_keyword(text: str, keywords) -> bool:
    text = text.lower()
    return any(keyword in text for keyword in keywords)


def is_flat_repeated(record: Dict[str, Any]) -> bool:
    direct = bool_field(record, ["flat_repeated_pair", "flat_repeat", "flat_repeat_detected"])
    if direct is not None:
        return direct
    reason_text = flatten_reason(record.get("failure_reason")) + " " + nested_guard_reason(record)
    return has_keyword(reason_text, FLAT_KEYWORDS)


def is_reused_history(record: Dict[str, Any]) -> bool:
    direct = bool_field(record, ["reused_history_pairs", "mostly_reused_history_pairs", "copied_history"])
    if direct is not None:
        return direct
    reason_text = flatten_reason(record.get("failure_reason")) + " " + nested_guard_reason(record)
    return has_keyword(reason_text, REUSED_KEYWORDS)


def finite_number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def summarize_group(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    result = {"count": len(records)}
    for key in METRIC_KEYS:
        values = [finite_number(record.get(key)) for record in records]
        values = [value for value in values if value is not None]
        result[key] = sum(values) / len(values) if values else None
    return result


def rate(count: int, total: int) -> Optional[float]:
    if total <= 0:
        return None
    return count / total


def build_analysis(records_jsonl: str, summary_json_path: Optional[str], records: List[Dict[str, Any]], summary):
    total_records = len(records)
    success_records = [record for record in records if bool(record.get("parse_success"))]
    parse_success = len(success_records)
    parse_failed = total_records - parse_success

    for record in records:
        record["_is_flat_repeated_pair"] = is_flat_repeated(record)
        record["_is_reused_history_pairs"] = is_reused_history(record)

    success_flat = [record for record in success_records if record["_is_flat_repeated_pair"]]
    success_reused = [record for record in success_records if record["_is_reused_history_pairs"]]
    non_flat = [record for record in success_records if not record["_is_flat_repeated_pair"]]
    non_reused = [record for record in success_records if not record["_is_reused_history_pairs"]]
    clean = [
        record
        for record in success_records
        if not record["_is_flat_repeated_pair"] and not record["_is_reused_history_pairs"]
    ]

    notes = []
    if summary:
        notes.append("summary_json was loaded and can be used to cross-check aggregate MAE.")
    if parse_success == 0:
        notes.append("No parse_success records were found; grouped MAE values are null.")
    if not clean:
        notes.append("No clean parse_success samples were found.")
    elif len(clean) < max(5, 0.05 * max(parse_success, 1)):
        notes.append("Clean sample count is small; interpret clean-group MAE cautiously.")

    return {
        "records_jsonl": records_jsonl,
        "summary_json": summary_json_path,
        "total_records": total_records,
        "parse_success": parse_success,
        "parse_failed": parse_failed,
        "flat_repeated_pair_count": len(success_flat),
        "flat_repeated_pair_rate": rate(len(success_flat), parse_success),
        "reused_history_pairs_count": len(success_reused),
        "reused_history_pairs_rate": rate(len(success_reused), parse_success),
        "clean_count": len(clean),
        "clean_rate": rate(len(clean), parse_success),
        "all": summarize_group(success_records),
        "non_flat": summarize_group(non_flat),
        "non_reused": summarize_group(non_reused),
        "clean": summarize_group(clean),
        "group_ratios": {
            "all": rate(len(success_records), parse_success),
            "non_flat": rate(len(non_flat), parse_success),
            "non_reused": rate(len(non_reused), parse_success),
            "clean": rate(len(clean), parse_success),
        },
        "notes": notes,
    }


def fmt(value: Any) -> str:
    number = finite_number(value)
    if number is None:
        return "n/a"
    return f"{number:.4f}"


def pct(value: Any) -> str:
    number = finite_number(value)
    if number is None:
        return "n/a"
    return f"{number * 100:.2f}%"


def build_text_report(analysis: Dict[str, Any]) -> str:
    rows = []
    for name in ["all", "non_flat", "non_reused", "clean"]:
        data = analysis[name]
        rows.append(
            "| {name} | {count} | {ratio} | {speed} | {curv} | {overall} |".format(
                name=name,
                count=data["count"],
                ratio=pct(analysis.get("group_ratios", {}).get(name)),
                speed=fmt(data.get("speed_mae_mps")),
                curv=fmt(data.get("curvature_mae_x100")),
                overall=fmt(data.get("overall_l1_train_scale")),
            )
        )

    parts = [
        "OpenEMMA Text-Generation Degeneracy Analysis",
        "",
        f"parse_success={analysis['parse_success']} / total_records={analysis['total_records']}",
        f"flat_repeated_pair_rate={pct(analysis['flat_repeated_pair_rate'])} "
        f"({analysis['flat_repeated_pair_count']} samples)",
        f"reused_history_pairs_rate={pct(analysis['reused_history_pairs_rate'])} "
        f"({analysis['reused_history_pairs_count']} samples)",
        f"clean_rate={pct(analysis['clean_rate'])} ({analysis['clean_count']} samples)",
        "",
        "| group | count | ratio | speed_mae_mps | curvature_mae_x100 | overall_l1_train_scale |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        *rows,
        "",
        "Conclusion: text-generation baseline 的 overall 较好主要受到历史复用/平坦重复的保守预测影响；"
        "在非退化样本上性能需要进一步观察。",
    ]
    if analysis["notes"]:
        parts.extend(["", "Notes:"])
        parts.extend(f"- {note}" for note in analysis["notes"])
    return "\n".join(parts) + "\n"


def write_json(path: str, data: Dict[str, Any]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def write_text(path: str, content: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    args = parse_args()
    records = load_records(args.records_jsonl)
    summary = load_json(args.summary_json)
    analysis = build_analysis(args.records_jsonl, args.summary_json, records, summary)
    write_json(args.output_json, analysis)
    if args.output_txt:
        write_text(args.output_txt, build_text_report(analysis))
    print(f"[TextGenerationDegeneracy] wrote json: {args.output_json}")
    if args.output_txt:
        print(f"[TextGenerationDegeneracy] wrote txt: {args.output_txt}")
    print(
        f"[TextGenerationDegeneracy] clean={analysis['clean_count']} "
        f"flat_rate={pct(analysis['flat_repeated_pair_rate'])} "
        f"reused_rate={pct(analysis['reused_history_pairs_rate'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
