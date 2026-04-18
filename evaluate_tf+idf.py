import os
import re
import string
from collections import Counter

BASE_DIR = "/data/home/sai/Desktop/RAG-QA-System-main/RAG-QA-System-main"

REFERENCE_FILE = os.path.join(BASE_DIR, "reference_answers.txt")
#SYSTEM_OUTPUT_FILE = os.path.join(BASE_DIR, "system_output_tfidf.txt")
SYSTEM_OUTPUT_FILE = os.path.join(BASE_DIR, "system_output_closedbook.txt")

# You can change this later to:
# SYSTEM_OUTPUT_FILE = os.path.join(BASE_DIR, "system_output_vector.txt")


def normalize_answer(text: str) -> str:
    """
    Lower text, remove punctuation, articles, and extra whitespace.
    This follows the usual SQuAD style normalization.
    """
    text = text.lower().strip()

    text = "".join(ch for ch in text if ch not in string.punctuation)

    text = re.sub(r"\b(a|an|the)\b", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def exact_match_score(prediction: str, ground_truth: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def recall_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 and len(truth_tokens) == 0:
        return 1.0
    if len(truth_tokens) == 0:
        return 0.0
    if len(pred_tokens) == 0:
        return 0.0

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    return num_same / len(truth_tokens)


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 and len(truth_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return 0.0

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)

    return 2 * precision * recall / (precision + recall)


def metric_max_over_references(metric_fn, prediction: str, references: list[str]) -> float:
    return max(metric_fn(prediction, ref) for ref in references)


def read_lines(file_path: str) -> list[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def parse_reference_line(line: str) -> list[str]:
    """
    Supports multiple valid reference answers separated by semicolon.
    Example:
    Robert Downey Jr.; Downey Jr.
    """
    answers = [part.strip() for part in line.split(";") if part.strip()]
    return answers if answers else [""]


def main():
    references_raw = read_lines(REFERENCE_FILE)
    predictions = read_lines(SYSTEM_OUTPUT_FILE)

    if len(references_raw) != len(predictions):
        print("Mismatch in number of lines.")
        print(f"Reference lines: {len(references_raw)}")
        print(f"Prediction lines: {len(predictions)}")
        return

    total = len(predictions)
    exact_match_total = 0.0
    recall_total = 0.0
    f1_total = 0.0

    print(f"Evaluating {total} predictions...\n")

    for i, (pred, ref_line) in enumerate(zip(predictions, references_raw), start=1):
        references = parse_reference_line(ref_line)

        em = metric_max_over_references(exact_match_score, pred, references)
        recall = metric_max_over_references(recall_score, pred, references)
        f1 = metric_max_over_references(f1_score, pred, references)

        exact_match_total += em
        recall_total += recall
        f1_total += f1

        print(f"Q{i}")
        print(f"Prediction : {pred}")
        print(f"Reference(s): {references}")
        print(f"Exact Match: {em}")
        print(f"Recall     : {recall:.4f}")
        print(f"F1         : {f1:.4f}")
        print("-" * 60)

    exact_match_percent = 100.0 * exact_match_total / total
    recall_percent = 100.0 * recall_total / total
    f1_percent = 100.0 * f1_total / total

    print("\nFinal Results")
    print("=" * 60)
    print(f"Total Questions : {total}")
    print(f"Exact Match     : {exact_match_percent:.2f}%")
    print(f"Average Recall  : {recall_percent:.2f}%")
    print(f"Average F1      : {f1_percent:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()