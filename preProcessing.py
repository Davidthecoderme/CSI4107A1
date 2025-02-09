import json
import re
import nltk
# pip install nltk
# nltk.download('stopwords')
# nltk.download('punkt')

# Make sure to have "pip install openpyxl" installed.

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# For writing Excel files
from openpyxl import Workbook

INPUT_FILE = "./scifact/corpus.jsonl"
OUTPUT_FILE = "output_tokens.xlsx"  # changed to .xlsx

# 1. Initialize stop words and Porter stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    """
    1) Convert all to lowercase
    2) Keep only alphabetical characters (remove punctuation and numbers)
    3) Tokenize
    4) Remove stop words
    5) Extract word stems using the Porter Stemmer
    """
    text = text.lower()
    tokens = re.findall(r"[a-z]+", text)
    filtered_tokens = [t for t in tokens if t not in stop_words]
    stemmed_tokens = [stemmer.stem(t) for t in filtered_tokens]
    return stemmed_tokens

def main():
    # Create a new Workbook
    wb = Workbook()
    ws = wb.active  # Get the active worksheet
    ws.title = "Tokens"

    # Write header row (optional)
    ws.append(["_id", "tokens"])

    with open(INPUT_FILE, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)

            doc_id = doc.get("_id", "")
            text = doc.get("text", "")
            tokens = preprocess_text(text)

            # Convert the list of tokens to a single string for Excel
            tokens_str = " ".join(tokens)

            # Append a new row: doc_id in the first column, tokens in the second
            ws.append([doc_id, tokens_str])

    # Save the workbook
    wb.save(OUTPUT_FILE)

    # Print a confirmation message
    print(f"Preprocessed data is saved to '{OUTPUT_FILE}'")

if __name__ == "__main__":
    main()
