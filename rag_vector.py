import os
import re
import string
from typing import List, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

BASE_DIR = "/home/ehr/Desktop/anupama/NLP_ASSIGNMENT-Building-RAG"
QUESTIONS_FILE = os.path.join(BASE_DIR, "questions.txt")
OUTPUT_FILE = os.path.join(BASE_DIR, "system_output_vector.txt")

DOCUMENT_FILES = [
    os.path.join(BASE_DIR, "avengers_endgame.txt"),
    os.path.join(BASE_DIR, "real_steel.txt"),
    os.path.join(BASE_DIR, "high_school_musical.txt"),
    os.path.join(BASE_DIR, "war_2019.txt"),
]

CHUNK_SIZE = 120
CHUNK_OVERLAP = 30
TOP_K = 6
MAX_INPUT_TOKENS = 384
MAX_NEW_TOKENS = 20
USE_FLAN_FALLBACK = False


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\[[0-9]+\]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_chunk(chunk: str) -> str:
    chunk = re.sub(r"\|.*?\|", " ", chunk)
    chunk = re.sub(r"\bv\s*t\s*e\b", " ", chunk, flags=re.IGNORECASE)
    chunk = re.sub(r"Contents\s+", " ", chunk, flags=re.IGNORECASE)
    chunk = re.sub(r"References\s+", " ", chunk, flags=re.IGNORECASE)
    chunk = re.sub(r"External links\s+", " ", chunk, flags=re.IGNORECASE)
    chunk = re.sub(r"\s+", " ", chunk)
    return chunk.strip()


def read_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return clean_text(f.read())


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    words = text.split()
    chunks = []

    start = 0
    step = max(1, chunk_size - overlap)

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


def split_into_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    cleaned = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence.split()) >= 3:
            cleaned.append(sentence)
    return cleaned


def load_documents() -> List[str]:
    all_chunks = []

    for file_path in DOCUMENT_FILES:
        if not os.path.exists(file_path):
            print(f"Warning: missing file: {file_path}")
            continue

        text = read_file(file_path)
        raw_chunks = chunk_text(text)

        for chunk in raw_chunks:
            cleaned = clean_chunk(chunk)
            if cleaned:
                all_chunks.append(cleaned)

    return all_chunks


def load_questions(file_path: str) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def normalize_for_matching(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def keyword_tokens(question: str) -> List[str]:
    stop_words = {
        "what", "when", "where", "who", "which", "how", "many", "much",
        "did", "does", "was", "were", "is", "are", "in", "on", "at", "of",
        "for", "to", "from", "the", "a", "an", "with", "by", "and",
        "this", "that", "it", "as", "its", "into", "under"
    }
    tokens = re.findall(r"[A-Za-z0-9']+", question.lower())
    return [t for t in tokens if t not in stop_words and len(t) > 1]


def retrieve_vector(
    question: str,
    chunks: List[str],
    embedder,
    chunk_embeddings: np.ndarray
) -> List[Tuple[str, float]]:
    question_embedding = embedder.encode([question], convert_to_numpy=True)
    scores = cosine_similarity(question_embedding, chunk_embeddings).flatten()

    top_indices = scores.argsort()[::-1][:TOP_K]
    return [(chunks[i], float(scores[i])) for i in top_indices]


def sentence_score(question: str, sentence: str) -> float:
    q_tokens = set(keyword_tokens(question))
    s_norm = normalize_for_matching(sentence)
    s_tokens = set(re.findall(r"[A-Za-z0-9']+", s_norm))

    overlap = len(q_tokens & s_tokens)
    lexical = overlap / len(q_tokens) if q_tokens else 0.0

    bonus = 0.0

    if re.search(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b", sentence):
        bonus += 0.2
    if re.search(r"\b(19|20)\d{2}\b", sentence):
        bonus += 0.2
    if re.search(r"\$\s?\d", sentence) or re.search(r"₹\s?\d", sentence):
        bonus += 0.2
    if re.search(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", sentence):
        bonus += 0.15

    if "|" in sentence:
        bonus -= 0.5
    if len(sentence.split()) > 45:
        bonus -= 0.3
    if sentence.count("(") > 3:
        bonus -= 0.15

    return lexical + bonus


def extract_candidate_answer(question: str, sentence: str) -> str:
    q = question.lower()

    if q.startswith("what year") or q.startswith("when"):
        full_date = re.search(
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b",
            sentence
        )
        if full_date:
            return full_date.group(0)

        year = re.search(r"\b(19|20)\d{2}\b", sentence)
        if year:
            return year.group(0)

    if "how much" in q or "gross" in q or "budget" in q or "profit" in q:
        money = re.search(
            r"(\$\s?\d[\d.,]*\s?(?:billion|million)?)|(₹\s?\d[\d.,]*\s?(?:crore|lakh|billion|million)?)",
            sentence,
            flags=re.IGNORECASE
        )
        if money:
            return money.group(0).strip()

    if "how many" in q:
        count = re.search(
            r"\b(?:over\s+|more than\s+|nearly\s+|about\s+)?\d+(?:\.\d+)?\s?(?:million|billion|crore|lakh|seasons|weeks|days|cities|countries|viewers)?\b",
            sentence,
            flags=re.IGNORECASE
        )
        if count:
            return count.group(0).strip()

    if q.startswith("who"):
        patterns = [
            r"directed by ([A-Z][a-z]+(?: [A-Z][a-z]+)+)",
            r"produced by ([A-Z][a-z]+(?: [A-Z][a-z]+)+)",
            r"written by ([A-Z][a-z]+(?: [A-Z][a-z]+)+)",
            r"starring ([A-Z][a-z]+(?: [A-Z][a-z]+)+)",
            r"played by ([A-Z][a-z]+(?: [A-Z][a-z]+)+)",
            r"portrayed by ([A-Z][a-z]+(?: [A-Z][a-z]+)+)",
            r"plays [^.,;:]{0,40}? \(([A-Z][a-z]+(?: [A-Z][a-z]+)+)\)",
        ]

        for pattern in patterns:
            match = re.search(pattern, sentence)
            if match:
                return match.group(1).strip()

        names = re.findall(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)+\b", sentence)
        if names:
            return names[0].strip()

    if q.startswith("where"):
        place = re.search(r"\b(?:in|at|on)\s+([A-Z][a-zA-Z]+(?:,?\s+[A-Z][a-zA-Z]+)*)", sentence)
        if place:
            return place.group(1).strip()

    if "studio" in q or "banner" in q or "network" in q:
        org_patterns = [
            r"\b(Marvel Studios)\b",
            r"\b(Walt Disney Studios Motion Pictures)\b",
            r"\b(Touchstone Pictures)\b",
            r"\b(Disney Channel)\b",
            r"\b(Yash Raj Films)\b",
            r"\b(DreamWorks Pictures)\b",
        ]
        for pattern in org_patterns:
            match = re.search(pattern, sentence)
            if match:
                return match.group(1)

    sentence = sentence.strip(" .")
    if len(sentence.split()) <= 12:
        return sentence

    return sentence


def postprocess_answer(answer: str) -> str:
    answer = answer.strip()
    answer = re.sub(r"\s+", " ", answer)

    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()

    bad_exact = {
        "",
        "not found",
        "answer",
        "context",
        "question",
        "unanswerable"
    }

    if answer.lower() in bad_exact:
        return "Not found"

    answer = answer.strip(" .,;:")

    if len(answer.split()) > 12:
        answer = " ".join(answer.split()[:12]).strip()

    if not answer:
        return "Not found"

    return answer


def extractive_answer(question: str, retrieved_chunks: List[Tuple[str, float]]) -> str:
    candidate_sentences = []

    for chunk, chunk_score in retrieved_chunks:
        for sentence in split_into_sentences(chunk):
            score = sentence_score(question, sentence) + (0.20 * chunk_score)
            candidate_sentences.append((sentence, score))

    if not candidate_sentences:
        return "Not found"

    candidate_sentences.sort(key=lambda x: x[1], reverse=True)
    best_sentence, best_score = candidate_sentences[0]

    if best_score < 0.18:
        return "Not found"

    answer = extract_candidate_answer(question, best_sentence)
    return postprocess_answer(answer)


def generate_answer(
    question: str,
    retrieved_chunks: List[Tuple[str, float]],
    tokenizer,
    model,
    device: str
) -> str:
    context_text = " ".join([chunk for chunk, _ in retrieved_chunks])

    prompt = (
        "Answer the question using only the given context.\n"
        "Return only a short factual answer.\n"
        "Do not explain.\n"
        "If the answer is unclear, return the shortest likely answer phrase from the context.\n\n"
        f"Context: {context_text}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=2,
            early_stopping=True
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return postprocess_answer(decoded)


def main():
    print("Loading documents...")
    chunks = load_documents()
    print(f"Loaded {len(chunks)} chunks.")

    if not chunks:
        print("No chunks found. Check your input files.")
        return

    print("Loading sentence embedding model...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Encoding document chunks...")
    chunk_embeddings = embedder.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    tokenizer = None
    model = None
    device = "cpu"

    if USE_FLAN_FALLBACK:
        print("Loading FLAN-T5 model...")
        model_name = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        print(f"Using device: {device}")
    else:
        print("FLAN fallback disabled. Using extractive answering only.")

    questions = load_questions(QUESTIONS_FILE)
    print(f"Loaded {len(questions)} questions.")

    answers = []

    for idx, question in enumerate(questions, start=1):
        retrieved_chunks = retrieve_vector(question, chunks, embedder, chunk_embeddings)
        answer = extractive_answer(question, retrieved_chunks)

        if answer == "Not found" and USE_FLAN_FALLBACK and model is not None:
            answer = generate_answer(question, retrieved_chunks, tokenizer, model, device)

        answers.append(answer)

        print(f"[{idx}/{len(questions)}] {question}")
        print("Top retrieved chunks:")
        for rank, (chunk, score) in enumerate(retrieved_chunks, start=1):
            preview = chunk[:120] + "..." if len(chunk) > 120 else chunk
            print(f"  {rank}. score={score:.4f} | {preview}")
        print(f"Answer: {answer}\n")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for ans in answers:
            f.write(ans + "\n")

    print(f"Done. Answers saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()