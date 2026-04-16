import os
import re
import string
from typing import List, Tuple

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

BASE_DIR = "/home/ehr/Desktop/anupama/NLP_ASSIGNMENT-Building-RAG"
QUESTIONS_FILE = os.path.join(BASE_DIR, "questions.txt")
OUTPUT_FILE = os.path.join(BASE_DIR, "system_output_tfidf.txt")

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


def split_into_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    cleaned = []
    for s in sentences:
        s = s.strip()
        if len(s.split()) >= 3:
            cleaned.append(s)
    return cleaned


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


def retrieve_tfidf(
    question: str,
    chunks: List[str],
    vectorizer: TfidfVectorizer,
    doc_matrix
) -> List[Tuple[str, float]]:
    question_vec = vectorizer.transform([question])
    scores = cosine_similarity(question_vec, doc_matrix).flatten()

    top_indices = scores.argsort()[::-1][:TOP_K]
    return [(chunks[i], float(scores[i])) for i in top_indices]


def keyword_tokens(question: str) -> List[str]:
    stop_words = {
        "what", "when", "where", "who", "which", "how", "many", "much",
        "did", "does", "was", "were", "is", "are", "in", "on", "at", "of",
        "for", "to", "from", "the", "a", "an", "with", "by", "and"
    }
    tokens = re.findall(r"[A-Za-z0-9']+", question.lower())
    return [t for t in tokens if t not in stop_words and len(t) > 1]


def sentence_score(question: str, sentence: str) -> float:
    q_norm = normalize_for_matching(question)
    s_norm = normalize_for_matching(sentence)

    q_tokens = set(keyword_tokens(question))
    s_tokens = set(re.findall(r"[A-Za-z0-9']+", s_norm))

    overlap = len(q_tokens & s_tokens)

    bonus = 0.0

    # Reward answer-like patterns
    if re.search(r"\b\d{4}\b", sentence):
        bonus += 0.3
    if re.search(r"\$\s?\d", sentence) or re.search(r"₹\s?\d", sentence):
        bonus += 0.3
    if re.search(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", sentence):
        bonus += 0.2

    # Penalize nav/list junk
    if "|" in sentence:
        bonus -= 0.5
    if sentence.count("(") > 3:
        bonus -= 0.2
    if len(sentence.split()) > 45:
        bonus -= 0.3

    # Reward direct lexical overlap
    lexical = 0.0
    if q_tokens:
        lexical = overlap / len(q_tokens)

    return lexical + bonus


def extract_candidate_answer(question: str, sentence: str) -> str:
    q = question.lower()

    # Year / date questions
    if q.startswith("what year") or q.startswith("when"):
        date_match = re.search(
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b",
            sentence
        )
        if date_match:
            return date_match.group(0)

        year_match = re.search(r"\b(19|20)\d{2}\b", sentence)
        if year_match:
            return year_match.group(0)

    # Money questions
    if "how much" in q or "gross" in q or "budget" in q or "profit" in q:
        money_match = re.search(
            r"(\$\s?\d[\d.,]*\s?(?:billion|million)?)|(₹\s?\d[\d.,]*\s?(?:crore|lakh|billion|million)?)",
            sentence,
            flags=re.IGNORECASE
        )
        if money_match:
            return money_match.group(0).strip()

    # Count questions
    if "how many" in q:
        count_match = re.search(
            r"\b(?:over\s+|more than\s+|nearly\s+|about\s+)?\d+(?:\.\d+)?\s?(?:million|billion|crore|lakh|seasons|weeks|days|cities|countries|viewers)?\b",
            sentence,
            flags=re.IGNORECASE
        )
        if count_match:
            return count_match.group(0).strip()

    # Who questions
    if q.startswith("who"):
        # Prefer "X directed", "X produced", etc.
        patterns = [
            r"directed by ([A-Z][a-z]+(?: [A-Z][a-z]+)+)",
            r"produced by ([A-Z][a-z]+(?: [A-Z][a-z]+)+)",
            r"written by ([A-Z][a-z]+(?: [A-Z][a-z]+)+)",
            r"starring ([A-Z][a-z]+(?: [A-Z][a-z]+)+)",
            r"plays? [^.,;:]{0,40}? ([A-Z][a-z]+(?: [A-Z][a-z]+)+)",
            r"portrays? [^.,;:]{0,40}? ([A-Z][a-z]+(?: [A-Z][a-z]+)+)",
        ]
        for pattern in patterns:
            m = re.search(pattern, sentence)
            if m:
                return m.group(1).strip()

        # fallback: first likely person name
        names = re.findall(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)+\b", sentence)
        if names:
            return names[0].strip()

    # Where questions
    if q.startswith("where"):
        m = re.search(r"\b(?:in|at|on)\s+([A-Z][a-zA-Z]+(?:,?\s+[A-Z][a-zA-Z]+)*)", sentence)
        if m:
            return m.group(1).strip()

    # Which studio / banner / network
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
            m = re.search(pattern, sentence)
            if m:
                return m.group(1)

    # General direct sentence cleanup fallback
    sentence = sentence.strip(" .")
    if len(sentence.split()) <= 12:
        return sentence

    return sentence


def extractive_answer(question: str, retrieved_chunks: List[Tuple[str, float]]) -> str:
    candidate_sentences = []

    for chunk, chunk_score in retrieved_chunks:
        for sent in split_into_sentences(chunk):
            score = sentence_score(question, sent) + (0.15 * chunk_score)
            candidate_sentences.append((sent, score))

    if not candidate_sentences:
        return "Not found"

    candidate_sentences.sort(key=lambda x: x[1], reverse=True)
    best_sentence, best_score = candidate_sentences[0]

    if best_score < 0.15:
        return "Not found"

    answer = extract_candidate_answer(question, best_sentence)
    answer = postprocess_answer(answer)

    return answer


def postprocess_answer(answer: str) -> str:
    answer = answer.strip()
    answer = re.sub(r"\s+", " ", answer)

    bad_exact = {
        "not found",
        "answer:",
        "context:",
        "question:",
        "high school musical",
        "avengers: endgame",
        "real steel",
        "war",
    }

    if not answer:
        return "Not found"

    if answer.lower() in bad_exact:
        return "Not found"

    if len(answer.split()) > 12:
        answer = " ".join(answer.split()[:12])

    answer = answer.strip(" .,;:")

    if not answer:
        return "Not found"

    return answer


def generate_answer_with_flan(
    question: str,
    retrieved_chunks: List[Tuple[str, float]],
    tokenizer,
    model,
    device: str
) -> str:
    context_text = " ".join([chunk for chunk, _ in retrieved_chunks])

    prompt = (
        "Answer the question using only the context.\n"
        "Return only a short answer phrase.\n"
        "Do not explain.\n\n"
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

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return postprocess_answer(decoded)


def main():
    print("Loading documents...")
    chunks = load_documents()
    print(f"Total chunks: {len(chunks)}")

    if not chunks:
        print("No document chunks found. Please check your input files.")
        return

    print("Building TF-IDF index...")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_df=0.9,
        min_df=1
    )
    doc_matrix = vectorizer.fit_transform(chunks)

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
        print("FLAN fallback disabled. Using extractive QA only.")

    questions = load_questions(QUESTIONS_FILE)
    print(f"Questions loaded: {len(questions)}")

    answers = []

    for i, question in enumerate(questions, start=1):
        retrieved_chunks = retrieve_tfidf(question, chunks, vectorizer, doc_matrix)
        answer = extractive_answer(question, retrieved_chunks)

        if answer == "Not found" and USE_FLAN_FALLBACK and model is not None:
            answer = generate_answer_with_flan(question, retrieved_chunks, tokenizer, model, device)

        answers.append(answer)

        print(f"[{i}/{len(questions)}] {question}")
        print("Top retrieved chunks:")
        for rank, (chunk, score) in enumerate(retrieved_chunks, start=1):
            preview = chunk[:120] + "..." if len(chunk) > 120 else chunk
            print(f"  {rank}. score={score:.4f} | {preview}")
        print(f"Answer: {answer}\n")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for answer in answers:
            f.write(answer + "\n")

    print(f"Done. Answers saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()