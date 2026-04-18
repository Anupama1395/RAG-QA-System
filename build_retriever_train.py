import os
import re
import json
import random
import string
from typing import Dict, List

BASE_DIR = "/data/home/sai/Desktop/RAG-QA-System-main/RAG-QA-System-main"
QUESTIONS_FILE = os.path.join(BASE_DIR, "questions.txt")
REFERENCE_FILE = os.path.join(BASE_DIR, "reference_answers.txt")
DOC_DIR = os.path.join(BASE_DIR, "documents")
OUT_FILE = os.path.join(BASE_DIR, "retriever_train.jsonl")

DOCUMENT_FILES = [
    os.path.join(DOC_DIR, "avengers_endgame.txt"),
    os.path.join(DOC_DIR, "real_steel.txt"),
    os.path.join(DOC_DIR, "high_school_musical.txt"),
    os.path.join(DOC_DIR, "war_2019.txt"),
    os.path.join(DOC_DIR, "alice_in_wonderland.txt"),
    os.path.join(DOC_DIR, "bugonia.txt"),
]

MAX_SENTENCES_PER_CHUNK = 4
OVERLAP_SENTENCES = 1
RANDOM_SEED = 42
NEGATIVES_PER_EXAMPLE = 3

random.seed(RANDOM_SEED)


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("–", "-").replace("—", "-")
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def parse_reference_line(line: str) -> List[str]:
    refs = [x.strip() for x in line.split(";") if x.strip()]
    return refs if refs else [""]


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\[[0-9]+\]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_into_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sents = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sents if len(s.strip().split()) >= 3]


def chunk_text(text: str) -> List[str]:
    sents = split_into_sentences(text)
    chunks = []
    step = max(1, MAX_SENTENCES_PER_CHUNK - OVERLAP_SENTENCES)
    for start in range(0, len(sents), step):
        chunk = " ".join(sents[start:start + MAX_SENTENCES_PER_CHUNK]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def get_doc_title(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0].replace("_", " ").strip()


def load_chunk_records() -> List[Dict[str, str]]:
    records = []
    for path in DOCUMENT_FILES:
        if not os.path.exists(path):
            continue
        title = get_doc_title(path)
        with open(path, "r", encoding="utf-8") as f:
            text = clean_text(f.read())
        for chunk in chunk_text(text):
            records.append({
                "title": title,
                "source": path,
                "text": f"{title}. {chunk}",
                "raw_text": chunk,
            })
    return records


def detect_target_document(question: str) -> str | None:
    qn = normalize(question)
    for path in DOCUMENT_FILES:
        title = normalize(get_doc_title(path))
        title_no_year = re.sub(r"\b(19|20)\d{2}\b", "", title).strip()
        for candidate in {title, title_no_year}:
            if candidate and candidate in qn:
                return path
    return None


def answer_in_chunk(chunk: str, references: List[str]) -> bool:
    chunk_n = normalize(chunk)
    for ref in references:
        ref_n = normalize(ref)
        if ref_n and (ref_n in chunk_n or chunk_n in ref_n):
            return True
    return False


def main():
    questions = read_lines(QUESTIONS_FILE)
    references_raw = read_lines(REFERENCE_FILE)

    if len(questions) != len(references_raw):
        raise ValueError(f"questions={len(questions)} references={len(references_raw)} mismatch")

    chunks = load_chunk_records()
    by_source = {}
    for rec in chunks:
        by_source.setdefault(rec["source"], []).append(rec)

    num_written = 0

    with open(OUT_FILE, "w", encoding="utf-8") as out:
        for q, ref_line in zip(questions, references_raw):
            refs = parse_reference_line(ref_line)
            target_doc = detect_target_document(q)

            if target_doc is not None and target_doc in by_source:
                candidates = by_source[target_doc]
            else:
                candidates = chunks

            positives = [c for c in candidates if answer_in_chunk(c["raw_text"], refs)]
            if not positives:
                continue

            positive = positives[0]

            same_doc_negs = [c for c in candidates if c is not positive and not answer_in_chunk(c["raw_text"], refs)]
            other_negs = [c for c in chunks if c["source"] != positive["source"]]

            neg_pool = same_doc_negs + other_negs
            if not neg_pool:
                continue

            sampled_negs = random.sample(neg_pool, k=min(NEGATIVES_PER_EXAMPLE, len(neg_pool)))

            for neg in sampled_negs:
                row = {
                    "query": q,
                    "positive": positive["text"],
                    "negative": neg["text"]
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                num_written += 1

    print(f"Wrote {num_written} training triples to {OUT_FILE}")


if __name__ == "__main__":
    main()