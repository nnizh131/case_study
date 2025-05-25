import json
import argparse
import logging
from pathlib import Path

logger = logging.getLogger("doc_prettifier")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def fix_encoding_issues(text: str | None) -> str | None:
    if not text:
        return text
    return text.replace("â‚¬", "EUR").replace("Ã‰", "É").strip()


def extract_by_type(entities: list[dict], key_type: str) -> str | None:
    for e in entities:
        if e.get("type") == key_type:
            raw = e.get("mentionText") or e.get("normalizedValue", {}).get("text")
            return fix_encoding_issues(raw)
    return None


def extract_taxes(entities: list[dict]) -> list[dict[str, str]]:
    taxes = []
    for e in entities:
        if e.get("type") == "tax_item":
            props = e.get("properties", [])
            taxes.append(
                {
                    "type": extract_by_type(props, "tax_type") or "",
                    "this_period": extract_by_type(props, "tax_this_period") or "",
                    "ytd": extract_by_type(props, "tax_ytd") or "",
                }
            )
    return taxes


def extract_deductions(entities: list[dict]) -> list[dict[str, str]]:
    deductions = []
    for e in entities:
        if e.get("type") == "deduction_item":
            props = e.get("properties", [])
            deductions.append(
                {
                    "type": extract_by_type(props, "deduction_type") or "",
                    "this_period": extract_by_type(props, "deduction_this_period")
                    or "",
                    "ytd": extract_by_type(props, "deduction_ytd") or "",
                }
            )
    return deductions


def extract_line_items(entities: list[dict]) -> list[dict[str, str]]:
    items = []
    for e in entities:
        if e.get("type") == "line_item":
            props = e.get("properties", [])
            desc = extract_by_type(props, "line_item/description")
            amount = extract_by_type(props, "line_item/amount")
            if desc or amount:
                items.append({"description": desc or "", "amount": amount or ""})
    return items


def is_receipt(entities: list[dict]) -> bool:
    return any(
        e["type"].startswith("receipt_") or e["type"] == "line_item" for e in entities
    )


def prettify_payslip(entities: list[dict]) -> dict:
    return {
        "document_type": "payslip",
        "employee_id": extract_by_type(entities, "employee_id"),
        "employee_name": extract_by_type(entities, "employee_name"),
        "employee_address": extract_by_type(entities, "employee_address"),
        "pay_date": extract_by_type(entities, "pay_date"),
        "start_date": extract_by_type(entities, "start_date"),
        "end_date": extract_by_type(entities, "end_date"),
        "gross_earnings": extract_by_type(entities, "gross_earnings"),
        "gross_earnings_ytd": extract_by_type(entities, "gross_earnings_ytd"),
        "net_pay": extract_by_type(entities, "net_pay"),
        "net_pay_ytd": extract_by_type(entities, "net_pay_ytd"),
        "deductions": extract_deductions(entities),
        "taxes": extract_taxes(entities),
    }


def prettify_receipt(entities: list[dict]) -> dict:
    return {
        "document_type": "receipt",
        "supplier_name": extract_by_type(entities, "supplier_name"),
        "supplier_address": extract_by_type(entities, "supplier_address"),
        "supplier_phone": extract_by_type(entities, "supplier_phone"),
        "receipt_date": extract_by_type(entities, "receipt_date"),
        "net_amount": extract_by_type(entities, "net_amount"),
        "total_tax_amount": extract_by_type(entities, "total_tax_amount"),
        "total_amount": extract_by_type(entities, "total_amount"),
        "currency": extract_by_type(entities, "currency"),
        "line_items": extract_line_items(entities),
    }


def process_file(file_path: Path) -> dict:
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    entities = data.get("entities", [])
    result = (
        prettify_receipt(entities)
        if is_receipt(entities)
        else prettify_payslip(entities)
    )
    return {k: v for k, v in result.items() if v}


def save_result(result: dict, original_file: Path, output_dir: Path) -> None:
    output_file = output_dir / f"{original_file.stem}_pretty.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved: {output_file.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prettify payslip/receipt JSON files.")
    parser.add_argument(
        "-i", "--input-dir", required=True, help="Directory with input JSON files"
    )
    parser.add_argument(
        "-o", "--output-dir", required=True, help="Directory to save prettified output"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )

    args = parser.parse_args()
    logger.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(input_dir.glob("*.json"))
    if not files:
        logger.warning("No JSON files found in input directory.")
        return

    for file_path in files:
        if file_path.name.endswith("_pretty.json"):
            logger.debug(f"Skipping already-prettified file: {file_path.name}")
            continue

        try:
            logger.info(f"Processing: {file_path.name}")
            result = process_file(file_path)
            save_result(result, file_path, output_dir)
        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {e}", exc_info=True)


if __name__ == "__main__":
    main()
