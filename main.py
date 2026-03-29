from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import pandas as pd
import warnings
import spacy
import sys
import re
import os
warnings.filterwarnings("ignore")

nlp = spacy.load("en_core_web_sm")

folder_path = Path("corpus")

documents = []
filenames = []

for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        with open(os.path.join(folder_path, filename), "r", encoding = "utf-8") as f:
            raw = f.read()
            cleaned = re.sub(r'[\n\t\r]+', ' ', raw)
            cleaned = re.sub(r'[•·⁃\uf0b7\uf0d8\xa0\uf0b7]+', ' ', cleaned)
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            documents.append(cleaned)
            filenames.append(filename.split(".")[0])

def chunk_by_lecture(text: str, filename: str):
    chunks = re.split(r'(?=Lecture\s+\d+)', text, flags=re.IGNORECASE)
    chunks = [c.strip() for c in chunks if c.strip()]
    
    chunk_texts = []
    chunk_names = []
    
    for chunk in chunks:
        match = re.match(r'Lecture\s+(\d+)', chunk, flags=re.IGNORECASE)
        if match:
            lecture_num = match.group(1)
            chunk_names.append(f"{filename} lecture_{lecture_num}")
        else:
            chunk_names.append(f"{filename} intro")
    
        chunk_texts.append(chunk)
    
    return chunk_texts, chunk_names

all_chunks = []
all_chunk_names = []

for filename, doc in zip(filenames, documents):
    chunks, names = chunk_by_lecture(doc, filename)
    all_chunks.extend(chunks)
    all_chunk_names.extend(names)

def tokenize(text: str):

    pattern = re.compile(r'\b[a-zA-Z][a-zA-Z0-9]{2,}\b')
    filtered = ' '.join(pattern.findall(text))

    doc = nlp(filtered)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and len(token.lemma_)>2]

    return tokens

tfidf = TfidfVectorizer(
    tokenizer = tokenize,
    ngram_range=(1,3)
)

term_doc = tfidf.fit_transform(all_chunks)

vocab = tfidf.get_feature_names_out()

term_doc_df = pd.DataFrame(term_doc.toarray(), columns = vocab, index = all_chunk_names)

def search(query: str, tfidf: TfidfVectorizer, tfidf_matrix: pd.DataFrame, filenames: list[str]):

    query_vec = tfidf.transform([query])

    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    top_doc = similarities.argsort()[::-1][0]

    print(f"Query: {query}")
    print(f"Answer: Check {filenames[top_doc]}")

def main():

    if len(sys.argv) > 2:
        raise Exception("Input exceeds allowed length. Wrap your query in a string")
    query = sys.argv[1]

    search(query, tfidf, term_doc_df, all_chunk_names)

if __name__ == "__main__":
    main()