# **CSI 4107 Assignment 1 - Information Retrieval System**


## Names and Student Numbers

Bruno Kazadi Kazadi () <br>
Bernardo Caiado () <br>
Jun Ning (300286811)


## Distribution of Work
Bruno Kazadi Kazadi
- Step3 :
- 
Bernardo Caiado
- Stpe2 :
- 
Jun Ning
- Step 1: Preprocessing
- report 

## **Overview**
This assignment implements an Information Retrieval system for the SciFact dataset, focusing on both a traditional Vector Space Model (VSM) with TF–IDF and an advanced BM25 ranking approach. The system accepts a set of queries, identifies the most relevant documents using the inverted index, and ranks them based on similarity scores. Finally, the performance is evaluated using Mean Average Precision (MAP) and other metrics via the standard trec_eval tool.


## Functionality of Programs


This code implements a basic Information Retrieval (IR) pipeline in three main stages: preprocessing, indexing, and retrieval/ranking. Below is a detailed outline of how each function and step works.

#### Preprocessing

Tokenize and Clean Text (tokenize_and_remove_punctuations):
Removes punctuation and digits from the input text, then lowercases and tokenizes it into words.
Stopword Removal and Stemming (preprocess_text):
Uses the NLTK stopwords list to filter out common English words. Also removes very short tokens (length ≤ 2) to reduce noise. Finally, applies the Porter Stemmer to normalize words (e.g., “running,” “runs,” become “run”).
Reading Input Data

#### Reading the Corpus (read_corpus):
Loads each document from the corpus.jsonl file, extracts the _id, title, and text fields, concatenates the textual content, and preprocesses it to produce a list of tokens. Stores these tokens in a dictionary keyed by doc_id.

Reading Relevance Judgments (get_relevance):
Loads the ground-truth relevance judgments from a TSV file (e.g., test.tsv). Only documents with a relevance score > 0 are considered relevant for each query. This helps in evaluating retrieval performance later.

Reading Queries (read_queries):
Loads and preprocesses queries from queries.jsonl. Only queries that appear in the relevance file (i.e., “valid queries”) are kept. Each query’s text is converted to a list of tokens after the same preprocessing steps as the corpus.
Indexing

#### Build Inverted Index (build_inverted_index):
Creates a dictionary mapping each unique token to the set of document IDs in which that token appears. This allows quick identification of candidate documents that contain a given query term.
Compute IDF (calculate_idf):

TF–IDF Vectors:
For each document, the code computes term frequencies (calculate_tf) and multiplies these by IDF values to produce a TF–IDF vector (calculate_tfidf). This vector is stored in doc_tfidf for later retrieval.
Retrieval & Ranking

#### Query Processing:
For each query, compute the TF–IDF vector in the same manner as the documents.
Candidate Document Retrieval:
Based on the inverted index, gather all documents containing at least one query token. This set is a candidate pool for ranking.

Cosine Similarity (cosine_similarity):
Compare each candidate document’s TF–IDF vector with the query’s TF–IDF vector. The code calculates their dot product over the terms and divides by the product of their vector norms. The result is a score in the range [0, 1], where 1 indicates perfect alignment.

Ranking & Output:
The code sorts candidate documents in descending order of similarity score and keeps the top 100 for each query. These are written to Results.txt in TREC format, which lists the query ID, document ID, rank, and score.
Program Flow (Main Function)

#### Command-Line Arguments: The script expects three arguments: corpus.jsonl, queries.jsonl, and test.tsv.
Load Relevance File to identify the queries of interest.
Read & Preprocess Corpus; build a dictionary of doc_id → tokens.
Read Queries (for the query IDs in the relevance file) and preprocess them.
Construct Inverted Index, compute IDF, and build TF–IDF vectors for all documents.
#### For Each Query:
Compute query’s TF–IDF vector.
Find candidate docs from the inverted index, compute their cosine similarity, rank them, and store the top 100.
Write Rankings to Results.txt in TREC format.

#### Output & Evaluation
The final output is the file Results.txt, which contains up to 100 documents per query, sorted by descending similarity.

These results can be evaluated against the relevance judgments using the trec_eval tool or other IR evaluation methods (computing Mean Average Precision).


## How to Run 

### **Install Dependencies**
```
pip install ntlk
```

### **Run the Information Retrieval System** 
- which will produce the Results.txt

```
python  ir_system.py corpus.jsonl queries.jsonl formated-test.tsv 

```

### **Evaluate Performance**

```
trec_eval formated-test.tsv Results.txt

```


## Algorithms, Data Structures, and Optimizations 

Preprocessing

* Algorithms: Tokenize text, remove stopwords, and apply Porter stemming.
* Data Structures: Primarily lists (for storing tokens) and sets (for stopwords).
* Optimizations: Convert all text to lowercase and discard very short tokens (length < 3).
  
Indexing

* Algorithms: Build an inverted index (term → set of doc IDs). Compute TF for each document and IDF globally, then combine for TF–IDF.
* Data Structures: Dictionaries (dict or defaultdict) for the inverted index and for storing TF–IDF vectors.
* Optimizations: Use sets to eliminate duplicates and accelerate IDF calculation.

Retrieval & Ranking

* Algorithms: Vectorize the query (TF–IDF), retrieve candidate docs from the inverted index, then rank by cosine similarity.
* Data Structures: Dictionaries for query and document vectors; lists for storing doc scores.
* Optimizations: Only score documents that match at least one query term. Sort by similarity and keep the top 100.

## Vocabulary

### How big was the vocabulary?

### Sample of 100 tokens from the vocabulary

### **First 10 Answers for First Two Queries**

## **Results & Discussion**
- Mean Average Precision (MAP) score: 

0.500859425397324  ![image](https://github.com/user-attachments/assets/6a7a233e-add8-412c-8e10-b9bdcb5934c9)


## **References** : 
- **TREC Eval**:  https://github.com/cvangysel/pytrec_eval
