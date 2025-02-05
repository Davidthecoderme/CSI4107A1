import jsonlines
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """Preprocess text: lowercase, remove punctuation, stopwords, and apply lemmatization."""
    if not text:
        return ""

    # Convert text to lowercase and remove punctuation
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    # Split text into words
    tokens = text.split()
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def process_corpus(input_file, output_file):
    """Process the test corpus file and save the cleaned version."""
    processed_data = []

    with jsonlines.open(input_file) as reader:
        for obj in reader:
            # Check for missing fields and skip invalid entries
            if "_id" not in obj or "title" not in obj or "text" not in obj:
                print(f"❌ Skipping invalid entry: {obj}")
                continue

            # Extract fields
            doc_id = obj["_id"]
            text = obj["title"] + " " + obj["text"]  # Combine title and text

            # Preprocess the combined text
            cleaned_text = preprocess_text(text)
            processed_data.append({"doc_id": doc_id, "text": cleaned_text})

    # Save the processed data
    with jsonlines.open(output_file, mode='w') as writer:
        writer.write_all(processed_data)

    print(f"✅ Preprocessing complete. Processed {len(processed_data)} documents. Output saved to {output_file}")

if __name__ == "__main__":
    # File paths for the test corpus
    input_file = "test_corpus.jsonl"  # Input test file
    output_file = "processed_test_corpus.jsonl"  # Output processed file

    # Process the test corpus
    process_corpus(input_file, output_file)
