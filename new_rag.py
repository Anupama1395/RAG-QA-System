import os
import re
import string
from typing import Dict, List, Optional, Tuple

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
BASE_DIR = "/data/home/sai/Desktop/RAG-QA-System-main/RAG-QA-System-main"
QUESTIONS_FILE = os.path.join(BASE_DIR, "questions.txt")
OUTPUT_FILE = os.path.join(BASE_DIR, "system_output_tfidf.txt")
DOC_DIR = os.path.join(BASE_DIR, "documents")

DOCUMENT_FILES = [
    os.path.join(DOC_DIR, "avengers_endgame.txt"),
    os.path.join(DOC_DIR, "real_steel.txt"),
    os.path.join(DOC_DIR, "high_school_musical.txt"),
    os.path.join(DOC_DIR, "war_2019.txt"),
    os.path.join(DOC_DIR, "alice_in_wonderland.txt"),
    os.path.join(DOC_DIR, "bugonia.txt"),
]

TOP_K = 2
MAX_INPUT_TOKENS = 768
MAX_NEW_TOKENS = 6
MODEL_NAME = "google/flan-t5-base"
USE_GENERATOR = True

MAX_SENTENCES_PER_CHUNK = 3
OVERLAP_SENTENCES = 1
TITLE_MATCH_BOOST = 0.6


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\[[0-9]+\]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_chunk(chunk: str) -> str:
    chunk = re.sub(r"\|.*?\|", " ", chunk)
    chunk = re.sub(r"\bContents\b", " ", chunk, flags=re.IGNORECASE)
    chunk = re.sub(r"\bReferences\b", " ", chunk, flags=re.IGNORECASE)
    chunk = re.sub(r"\bExternal links\b", " ", chunk, flags=re.IGNORECASE)
    chunk = re.sub(r"\bSee also\b", " ", chunk, flags=re.IGNORECASE)
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


def chunk_text(
    text: str,
    max_sentences: int = MAX_SENTENCES_PER_CHUNK,
    overlap_sentences: int = OVERLAP_SENTENCES
) -> List[str]:
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks = []
    step = max(1, max_sentences - overlap_sentences)

    for start in range(0, len(sentences), step):
        chunk = " ".join(sentences[start:start + max_sentences]).strip()
        if chunk:
            chunks.append(chunk)

    return chunks


def normalize_for_matching(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_doc_title(file_path: str) -> str:
    name = os.path.splitext(os.path.basename(file_path))[0]
    return name.replace("_", " ").strip()


def get_title_variants(title: str) -> List[str]:
    variants = set()
    norm = normalize_for_matching(title)
    if norm:
        variants.add(norm)

    no_year = re.sub(r"\b(19|20)\d{2}\b", "", norm).strip()
    no_year = re.sub(r"\s+", " ", no_year).strip()
    if no_year:
        variants.add(no_year)

    return list(variants)

KEEP_TABLE_KEYS = {
    "Directed by",
    "Screenplay by",
    "Based on",
    "Produced by",
    "Starring",
    "Cinematography",
    "Edited by",
    "Music by",
    "Production companies",
    "Distributed by",
    "Release dates",
    "Running time",
    "Countries",
    "Language",
    "Budget",
    "Box office",
}


def normalize_table_value(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip()

    # Add commas for long person/company lists if they were flattened
    value = re.sub(r"([a-z]) ([A-Z])", r"\1, \2", value)

    # Remove extra parenthetical ISO-like dates if too noisy
    value = re.sub(r"\(\s*\d{4}-\d{2}-\d{2}\s*\)", "", value)

    value = re.sub(r"\s+", " ", value).strip(" .,;:")
    return value


def extract_table_facts(text: str, title: str) -> List[str]:
    if "--- TABLE DATA ---" not in text:
        return []

    table_part = text.split("--- TABLE DATA ---", 1)[1]
    facts = []

    for raw_line in table_part.splitlines():
        line = raw_line.strip()
        if not line or "|" not in line:
            continue

        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) < 2:
            continue

        key = parts[0]
        value = normalize_table_value(" ".join(parts[1:]))

        if key in KEEP_TABLE_KEYS and value:
            facts.append(f"{title}. {key} {value}.")

    return facts
def load_documents() -> List[Dict[str, str]]:
    all_chunks = []

    for file_path in DOCUMENT_FILES:
        if not os.path.exists(file_path):
            print(f"Warning: missing file: {file_path}")
            continue

        title = get_doc_title(file_path)
        text = read_file(file_path)
        raw_chunks = chunk_text(text)

        for chunk in raw_chunks:
            cleaned = clean_chunk(chunk)
            if cleaned:
                all_chunks.append({
                    "text": f"{title}. {cleaned}",
                    "raw_text": cleaned,
                    "source": file_path,
                    "title": title,
                })

    return all_chunks


def load_questions(file_path: str) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def detect_target_document(question: str, document_files: List[str]) -> Optional[str]:
    q_norm = normalize_for_matching(question)

    for path in document_files:
        title = get_doc_title(path)
        for variant in get_title_variants(title):
            if variant and variant in q_norm:
                return path

    return None
def retrieve_tfidf(
    question: str,
    chunk_records: List[Dict[str, str]],
    vectorizer: TfidfVectorizer,
    doc_matrix
) -> List[Dict[str, object]]:
    question_vec = vectorizer.transform([question])
    target_doc = detect_target_document(question, DOCUMENT_FILES)

    if target_doc is not None:
        candidate_indices = [
            i for i, rec in enumerate(chunk_records)
            if rec["source"] == target_doc
        ]
    else:
        candidate_indices = list(range(len(chunk_records)))

    # Fallback if routed document has no chunks
    if not candidate_indices:
        candidate_indices = list(range(len(chunk_records)))

    candidate_matrix = doc_matrix[candidate_indices]
    scores = cosine_similarity(question_vec, candidate_matrix).flatten()

    top_local = scores.argsort()[::-1][:TOP_K]

    results = []
    for local_idx in top_local:
        global_idx = candidate_indices[local_idx]
        rec = chunk_records[global_idx]
        results.append({
            "text": rec["text"],
            "raw_text": rec["raw_text"],
            "title": rec["title"],
            "source": rec["source"],
            "score": float(scores[local_idx]),
        })

    return results


def keyword_tokens(question: str) -> List[str]:
    stop_words = {
        "what", "when", "where", "who", "which", "how", "many", "much",
        "did", "does", "was", "were", "is", "are", "in", "on", "at", "of",
        "for", "to", "from", "the", "a", "an", "with", "by", "and", "film",
        "movie"
    }
    tokens = re.findall(r"[A-Za-z0-9']+", question.lower())
    return [t for t in tokens if t not in stop_words and len(t) > 1]


def sentence_score(question: str, sentence: str) -> float:
    q = question.lower()
    s = sentence.lower()

    q_tokens = set(keyword_tokens(question))
    s_tokens = set(re.findall(r"[A-Za-z0-9']+", s))

    overlap = len(q_tokens & s_tokens)
    score = 0.0

    if q_tokens:
        score += overlap / len(q_tokens)

    boosts = [
        (["director", "directed"], ["directed by"], 1.2),
        (["producer", "produced"], ["produced by"], 1.2),
        (["writer", "written", "screenplay"], ["written by", "screenplay by"], 1.2),
        (["starring", "stars", "cast"], ["starring"], 1.0),
        (["music", "composer", "composed"], ["music by", "score by", "composed by"], 1.0),
        (["distribut"], ["distributed by"], 1.0),
        (["budget"], ["budget"], 1.0),
        (["gross", "box office"], ["box office", "grossed"], 1.0),
        (["running time", "runtime"], ["running time", "minutes"], 1.0),
        (["language"], ["language", "english", "hindi"], 0.8),
        (["country"], ["united states", "india", "united kingdom"], 0.8),
        (["based on"], ["based on"], 1.0),
        (["release", "released", "premiere"], ["released", "premiered"], 0.8),
    ]

    for q_words, s_phrases, bonus in boosts:
        if any(w in q for w in q_words) and any(p in s for p in s_phrases):
            score += bonus

    if re.search(r"\b(19|20)\d{2}\b", sentence):
        score += 0.2
    if re.search(r"\$\s?\d", sentence):
        score += 0.3
    if re.search(r"\b\d+\s+minutes?\b", s):
        score += 0.4
    if re.search(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", sentence):
        score += 0.2

    if "|" in sentence:
        score -= 0.5
    if len(sentence.split()) > 45:
        score -= 0.3

    return score


def extract_candidate_answer(question: str, sentence: str) -> str:
    q = question.lower()
    s = sentence.strip()

    patterns = []

    if "direct" in q:
        patterns += [r"directed by ([^.]+?)(?:\.|$)"]

    if "produc" in q:
        patterns += [r"produced by ([^.]+?)(?:\.|$)"]

    if "wrote" in q or "written" in q or "screenplay" in q or "writer" in q:
        patterns += [r"(?:written by|screenplay by) ([^.]+?)(?:\.|$)"]

    if "star" in q or "starring" in q or "cast" in q:
        patterns += [r"starring ([^.]+?)(?:\.|$)"]

    if "music" in q or "composer" in q or "composed" in q:
        patterns += [r"(?:music by|score by|composed by) ([^.]+?)(?:\.|$)"]

    if "distribut" in q:
        patterns += [r"distributed by ([^.]+?)(?:\.|$)"]

    if "language" in q:
        patterns += [r"\b(English|Hindi|French|Spanish|Japanese|Korean)\b"]

    if "country" in q:
        patterns += [r"\b(United States|India|United Kingdom|Canada|Australia)\b"]

    if "running time" in q or "runtime" in q:
        patterns += [r"\b(\d+\s+minutes?)\b"]

    if "budget" in q or "gross" in q or "box office" in q:
        patterns += [r"(\$\s?\d[\d.,]*\s?(?:million|billion)?)",
                     r"(₹\s?\d[\d.,]*\s?(?:crore|lakh|million|billion)?)"]

    if "based on" in q:
        patterns += [r"based on ([^.]+?)(?:\.|$)"]

    if q.startswith("when") or q.startswith("what year"):
        patterns += [
            r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4})\b",
            r"\b((?:19|20)\d{2})\b"
        ]

    if q.startswith("where"):
        patterns += [r"\b(?:in|at|on)\s+([A-Z][A-Za-z]+(?:,?\s+[A-Z][A-Za-z]+)*)"]

    for pattern in patterns:
        m = re.search(pattern, s, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip(" .,;:")

    if q.startswith("who"):
        names = re.findall(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)+\b", s)
        if names:
            return names[0].strip()

    s = s.strip(" .")
    if len(s.split()) <= 14:
        return s

    return "Not found"


def extractive_answer(question: str, retrieved_chunks: List[Dict[str, object]]) -> Tuple[str, float]:
    candidate_sentences = []

    for chunk in retrieved_chunks:
        chunk_score = float(chunk["score"])
        for sent in split_into_sentences(str(chunk["raw_text"])):
            score = sentence_score(question, sent) + (0.15 * chunk_score)
            candidate_sentences.append((sent, score))

    if not candidate_sentences:
        return "Not found", 0.0

    candidate_sentences.sort(key=lambda x: x[1], reverse=True)
    best_sentence, best_score = candidate_sentences[0]

    answer = extract_candidate_answer(question, best_sentence)
    answer = postprocess_answer(answer)

    if answer == "Not found":
        return "Not found", best_score

    return answer, best_score


def postprocess_answer(answer: str) -> str:
    answer = answer.strip()
    answer = answer.split("\n")[0].strip()
    answer = re.sub(r"^(answer|response)\s*:\s*", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"^(it was|it is|the answer is)\s+", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"\s+", " ", answer)
    answer = answer.strip(" .,;:")

    bad_patterns = [
        "return only",
        "use only the context",
        "question:",
        "context:",
    ]
    for bp in bad_patterns:
        if bp in answer.lower():
            return "Not found"

    bad_exact = {"", "not found", "answer", "context", "question", "unanswerable"}
    if answer.lower() in bad_exact:
        return "Not found"

    if len(answer.split()) > 18:
        answer = " ".join(answer.split()[:18]).strip(" .,;:")

    return answer if answer else "Not found"


def build_context(retrieved_chunks: List[Dict[str, object]]) -> str:
    parts = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        parts.append(f"Source {i} ({chunk['title']}): {chunk['raw_text']}")
    return "\n".join(parts)


def generate_answer_with_flan(
    question: str,
    retrieved_chunks: List[Dict[str, object]],
    tokenizer,
    model,
    device: str
) -> str:
    context_text = build_context(retrieved_chunks)
    #"If the answer is not in the context, return Not found.\n\n"
    prompt = (
        "Answer the question using only the context below.\n"
        "Return only the short answer phrase.\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n"
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
            num_beams=4,
            early_stopping=True,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return postprocess_answer(decoded)


def choose_final_answer(
    question: str,
    extractive_ans: str,
    extractive_score: float,
    generative_ans: str
) -> str:
    if generative_ans != "Not found":
        return generative_ans

    if extractive_ans != "Not found" and extractive_score >= 0.25:
        return extractive_ans

    return "Not found"

def retrieve_dense(
    question: str,
    chunk_records: List[Dict[str, str]],
    dense_model,
    chunk_embeddings
) -> List[Dict[str, object]]:
    target_doc = detect_target_document(question, DOCUMENT_FILES)

    if target_doc is not None:
        candidate_indices = [
            i for i, rec in enumerate(chunk_records)
            if rec["source"] == target_doc
        ]
    else:
        candidate_indices = list(range(len(chunk_records)))

    if not candidate_indices:
        candidate_indices = list(range(len(chunk_records)))

    query_emb = dense_model.encode_query(question, convert_to_tensor=True)
    cand_embs = chunk_embeddings[candidate_indices]
    scores = util.cos_sim(query_emb, cand_embs)[0]

    top_local = scores.argsort(descending=True)[:TOP_K]

    results = []
    for local_idx in top_local:
        local_idx = int(local_idx)
        global_idx = candidate_indices[local_idx]
        rec = chunk_records[global_idx]
        results.append({
            "text": rec["text"],
            "raw_text": rec["raw_text"],
            "title": rec["title"],
            "source": rec["source"],
            "score": float(scores[local_idx]),
        })

    return results
def main():
    print("Loading documents...")
    chunk_records = load_documents()
    print(f"Total chunks: {len(chunk_records)}")

    if not chunk_records:
        print("No document chunks found.")
        return

    print("Building TF-IDF index...")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 3),
        max_df=0.95,
        min_df=1,
        sublinear_tf=True
    )

    chunk_texts = [rec["text"] for rec in chunk_records]
    doc_matrix = vectorizer.fit_transform(chunk_texts)

    tokenizer = None
    model = None
    device = "cpu"
    dense_model = SentenceTransformer(os.path.join(BASE_DIR, "retriever_model"))
    chunk_embeddings = dense_model.encode_document(
    [rec["text"] for rec in chunk_records],
    convert_to_tensor=True
    )
    if USE_GENERATOR:
        try:
            print(f"Loading generator model: {MODEL_NAME}")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            model.eval()
            print(f"Using device: {device}")
        except Exception as e:
            print(f"Generator could not be loaded: {e}")
            tokenizer = None
            model = None

    questions = load_questions(QUESTIONS_FILE)
    print(f"Questions loaded: {len(questions)}")

    answers = []

    for i, question in enumerate(questions, start=1):
        retrieved_chunks = retrieve_dense(question, chunk_records, dense_model, chunk_embeddings)

        extractive_ans, extractive_score = extractive_answer(question, retrieved_chunks)

        generative_ans = "Not found"
        if model is not None and tokenizer is not None:
            generative_ans = generate_answer_with_flan(
                question, retrieved_chunks, tokenizer, model, device
            )

        final_answer = choose_final_answer(
            question,
            extractive_ans,
            extractive_score,
            generative_ans
        )

        answers.append(final_answer)

        print(f"[{i}/{len(questions)}] {question}")
        print("Top retrieved chunks:")
        for rank, chunk in enumerate(retrieved_chunks, start=1):
            preview = str(chunk["raw_text"])
            preview = preview[:140] + "..." if len(preview) > 140 else preview
            print(f"  {rank}. title={chunk['title']} | score={float(chunk['score']):.4f} | {preview}")
        print(f"Extractive : {extractive_ans} (score={extractive_score:.4f})")
        print(f"Generative : {generative_ans}")
        print(f"Final      : {final_answer}\n")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for ans in answers:
            f.write(ans + "\n")

    print(f"Done. Answers saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()