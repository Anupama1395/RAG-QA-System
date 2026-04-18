import os
import re
import torch
from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

BASE_DIR = "/data/home/sai/Desktop/RAG-QA-System-main/RAG-QA-System-main"
QUESTIONS_FILE = os.path.join(BASE_DIR, "questions.txt")
OUTPUT_FILE = os.path.join(BASE_DIR, "system_output_closedbook.txt")

MODEL_NAME = "google/flan-t5-base"
MAX_INPUT_TOKENS = 256
MAX_NEW_TOKENS = 8


def load_questions(file_path: str) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def postprocess_answer(answer: str) -> str:
    answer = answer.strip()
    answer = answer.split("\n")[0].strip()
    answer = re.sub(r"^(answer|response)\s*:\s*", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"^(it was|it is|the answer is)\s+", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"\s+", " ", answer).strip(" .,;:")

    bad_patterns = [
        "return only",
        "use only",
        "question:",
        "context:",
    ]
    for bp in bad_patterns:
        if bp in answer.lower():
            return "Not found"

    if not answer:
        return "Not found"

    if len(answer.split()) > 8:
        answer = " ".join(answer.split()[:8]).strip(" .,;:")

    return answer if answer else "Not found"


def generate_closed_book_answer(question: str, tokenizer, model, device: str) -> str:
    prompt = (
    "Answer the question from your own knowledge.\n"
    "Return only a short answer phrase.\n"
    "Do not explain.\n\n"
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


def main():
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"Using device: {device}")

    questions = load_questions(QUESTIONS_FILE)
    print(f"Questions loaded: {len(questions)}")

    answers = []

    for i, question in enumerate(questions, start=1):
        answer = generate_closed_book_answer(question, tokenizer, model, device)
        answers.append(answer)

        print(f"[{i}/{len(questions)}] {question}")
        print(f"Answer: {answer}\n")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for answer in answers:
            f.write(answer + "\n")

    print(f"Done. Answers saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()