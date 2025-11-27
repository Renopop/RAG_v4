# csv_processing.py
import csv
from typing import List

def extract_text_from_csv(path: str, delimiter: str = ";") -> str:
    """
    Extracts all text from a CSV file and concatenates it into a single string.
    Each row is joined by spaces, and each line ends with a newline.
    """
    rows: List[str] = []

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            if not row:
                continue
            cleaned = " ".join(cell.strip() for cell in row if cell.strip())
            if cleaned:
                rows.append(cleaned)

    return "\n".join(rows)
