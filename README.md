# **CSI 4107 Assignment 1 - Information Retrieval System**


## Names and Student Numbers

Bruno Kazadi (300210848) <br>
Bernardo Caiado (300130165) <br>
Jun Ning (300286811)


## Distribution of Work
Bruno Kazadi
- Step3 : Retrieval and Ranking

Bernardo Caiado
- Stpe2 : Indexing  

Jun Ning
- Step 1: Preprocessing
  
## **Overview**
This assignment implements an Information Retrieval system for the SciFact dataset, focusing on both a traditional Vector Space Model (VSM) with TF–IDF and an advanced BM25 ranking approach. The system accepts a set of queries, identifies the most relevant documents using the inverted index, and ranks them based on similarity scores. Finally, the performance is evaluated using Mean Average Precision (MAP) and other metrics via the standard trec_eval tool.


## Functionality of Programs


This code implements a basic Information Retrieval (IR) pipeline in three main stages: preprocessing, indexing, and retrieval/ranking. Below is a detailed outline of how each function and step works.

#### Preprocessing

Tokenize and Clean Text (tokenize_and_remove_punctuations):
Removes punctuation and digits from the input text, then lowercases and tokenizes it into words.
Stopword Removal and Stemming (preprocess_text):
Uses the NLTK stopwords list to filter out common English words. Also removes very short tokens (length ≤ 2) to reduce noise. Finally, applies the Porter Stemmer to normalize words (e.g., “running,” “runs,” become “run”).

### Reading Input Data 

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

1. Verify that Python is installed on your system. If not, download and install it from the Python official website.
2. Download trec_eval Obtain the trec_eval tool from the TREC official website or from the references below 
3. Extract the trec_eval Package The downloaded file is a .tar archive. Decompress it using a utility like tar (on Linux/macOS) or 7-Zip (on Windows).

4. Compile trec_eval
    - On POSIX systems:
    `cd trec_eval-9.0.7 
    make`
    - On MinGW/GCC:
    `gcc -o trec_eval trec_eval.c`
5. the other option for using python to work on trec_eval is to have pip install pytrec_eval intsalled
    - pip install pytrec_eval
      
6. The Scifact dataset is available [here](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip).
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
**NOTE**: **If you encounter any issues while following the steps to set up the system, please feel free to email jning016@uottawa.ca for assistance**

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

We determined the vocabulary size by running len(inverted_index), which has 37,975 terms.

Below is a sample of the 100 most frequently occurring tokens from the vocabulary:
'cell', 'result', 'study', 'increase', 'protein', 'suggest', 'factor', 'associate', 'gene', 'role', 'expression', 'human', 'control', 'disease', 'effect', 'function', 'patient', 'data', 'level', 'identify', 'mechanism', 'conclusion', 'induce', 'method', 'analysis', 'model', 'response', 'specific', 'demonstrate', 'type', 'activity', 'treatment', 'compare', 'target', 'development', 'signal', 'cancer', 'reduce', 'require', 'regulate', 'report', 'process', 'base', 'pathway', 'receptor', 'significantly', 'change', 'finding', 'indicate', 'involve', 'present', 'risk', 'cause', 'potential', 'remain', 'age', 'clinical', 'mice', 'activation', 'determine', 'reveal', 'system', 'complex', 'relate', 'group', 'evidence', 'population', 'develop', 'tissue', 'bind', 'measure', 'express', 'number', 'mediate', 'observe', 'year', 'form', 'dna', 'decrease', 'know', 'growth', 'vivo', 'dependent', 'background', 'review', 'outcome', 'tumor', 'objective', 'early', 'lead', 'regulation', 'investigate', 'test', 'activate', 'interaction', 'occur', 'cellular', 'molecular', 'design', 'major'.

The extracted tokens are predominantly scientific and medical in nature, with a strong emphasis on neuroscience, diffusion MRI analysis, and brain development research. Words like "cell", "protein", and "gene" are commonly associated with biomedical studies, whereas terms such as "analysis", "model", and "response" suggest statistical or computational methodologies. Additionally, the absence of common stopwords (e.g., "the", "is", "and") indicates that effective preprocessing techniques were applied to refine the vocabulary. Furthermore, all tokens appear in lowercase, demonstrating that text normalization was implemented to maintain uniformity across the dataset.

This structured vocabulary will be instrumental in analyzing key trends, identifying significant relationships, and drawing insights from the dataset, particularly within the domains of medical imaging, neuroscience, and computational biology.


### **First 10 Answers for First Two Queries**
#### First 10 Answers for the First Query:
![image](https://github.com/user-attachments/assets/ca4ca659-5c0b-4418-a7bb-60b17f735930)



#### First 10 Answers for the Second Query: 
![image](https://github.com/user-attachments/assets/50968093-68ac-42ca-bd5c-4bd1dc555bbc)

## Discussion of Results

####  Comparison of the Two Queries

The two queries have yielded different sets of documents, with varying relevance scores. The first query returned documents with relatively low scores, with the highest being 0.1163, while the second query produced documents with much higher scores, reaching up to 0.4184. This indicates that the second query likely had more highly relevant documents available in the dataset.

####  Score Distribution

For the first query, the scores are tightly clustered between 0.0544 and 0.1163, suggesting that no document was overwhelmingly relevant compared to others. In contrast, the second query shows a much wider range, from 0.1582 to 0.4184, implying that some documents were significantly more relevant than others.

####  Ranking and Document IDs

The ranking of documents follows the expected pattern, with documents receiving higher scores being placed at the top. The distribution of document IDs appears to be random, meaning that no specific pattern is visible in their assignment.

####  System Performance

The results suggest that "myIRsystem" performs well in ranking documents based on their relevance scores. However, the difference in score ranges between the two queries suggests that either the first query was more ambiguous or the document collection had fewer relevant documents for it.

####  Potential Improvements

Query Expansion: The first query could benefit from query expansion techniques to retrieve more relevant documents.

Relevance Feedback: Implementing user feedback mechanisms could refine the ranking further.

TF-IDF or BM25 Optimization: The scoring function might need optimization to ensure better differentiation in cases like the first query.


## **Results**
- Mean Average Precision (MAP) score with titles and text: 

0.500859425397324  ![image](https://github.com/user-attachments/assets/6a7a233e-add8-412c-8e10-b9bdcb5934c9)

- Mean Average Precision (MAP) score with titles only:
![image](https://github.com/user-attachments/assets/651fe010-3745-4ca2-ae81-ec496b9fadc1)


The MAP score for titles and text is 0.500859425397324. This indicates that the information retrieval system performs moderately well when both titles and text are utilized for ranking relevant documents.
The relatively high MAP score suggests that incorporating text into the retrieval process enhances the system's ability to rank documents relevant to the queries effectively.
Mean Average Precision (MAP) for Titles Only:

The MAP score for titles only is 0.3790, which is notably lower than the MAP score for titles and text.
This decrease in performance indicates that the absence of additional textual data limits the system's ability to provide more precise rankings, as it relies solely on title information.
Comparison of Metrics:

Other metrics like recip_rank (0.2979), P@10 (0.4667), and iprec_at_recall_0.50 (0.3849) further support that titles and text together result in better performance compared to titles alone. For instance:
The reciprocal rank (recip_rank) is slightly low, indicating room for improvement in retrieving the top-most relevant documents.
The Precision@10 (P@10) for titles alone is below 0.5, implying that less than half of the top 10 retrieved documents are relevant.
Conclusion:

The analysis suggests that including both titles and text is essential for achieving better MAP scores and overall performance.
Future improvements can focus on further enriching the text features or exploring other ranking techniques to enhance precision and recall.


## **References** : 
- **TREC Eval**:  https://github.com/cvangysel/pytrec_eval
