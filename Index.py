from collections import defaultdict
import json

def build_inverted_index(corpus_file, do_stemming=True):
    """
    Build an inverted index from a Scifact corpus.jsonl file.
    
    Returns:
    --------
    index: dict, str -> dict, where index[token] = {doc_id: term_frequency, ...}
    doc_lengths: dict, doc_id -> int (length of doc in tokens)
    N: int, total number of documents
    """
    index = defaultdict(lambda: defaultdict(int))  # token -> {docID -> term freq}
    doc_lengths = defaultdict(int)
    N = 0
    
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            doc_json = json.loads(line)
            doc_id = str(doc_json["doc_id"])  # ensure doc_id is a string
            # Combine title + abstract (or just abstract if you want)
            text_content = (doc_json.get("title", "") + " " +
                            doc_json.get("abstract", "")).strip()
            
            tokens = preprocess_text(text_content, do_stemming=do_stemming)
            doc_lengths[doc_id] = len(tokens)
            
            # Count term frequencies for this doc
            tf_doc = defaultdict(int)
            for tok in tokens:
                tf_doc[tok] += 1
            
            # Update the main index
            for tok, freq in tf_doc.items():
                index[tok][doc_id] = freq

            N += 1

    return index, doc_lengths, N

import math

def compute_idf(index, N, min_df=1):
    """
    Compute IDF for each token in the index.
    Using standard formula: idf = log((N - df + 0.5) / (df + 0.5))
    or you can use a simpler log(N/df).
    
    Returns:
    --------
    idf: dict, token -> idf value
    """
    idf = {}
    for token, posting in index.items():
        df = len(posting)  # number of docs containing this token
        if df < min_df:  # optionally remove very rare tokens
            continue
        # IDF formulas vary; use whichever you like.  For BM25:
        #  idf_t = log((N - df + 0.5) / (df + 0.5) + 1)
        # For a simpler TF-IDF style:
        #  idf_t = log(N / df)
        idf_t = math.log((N - df + 0.5)/(df + 0.5) + 1)
        idf[token] = idf_t
    return idf
