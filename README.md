# RAG-QA-System
Required imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

to run new_rag.py 

first run python build_retriever_train.py
next run python train_retriever.py
now python new_rag.py

to evaluate you can run evaluate_tfidf.py but change the file name with the correct rag output file

all other scripts can just be ran normally with python

web_pages.py can be used to get new wikipedia movie summaries
