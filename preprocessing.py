import re
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Initialize stemmer and stopwords
ps = PorterStemmer()
stopwords_set = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text) # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation)) # Remove punctuation
    text = text.strip()
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stopwords_set]
    return ' '.join(words)

def get_vectorizers(data):
    # NOTE: In a real application, these vectorizers would be trained once and saved.
    # For this project, we'll re-train them for simplicity in this script.
    # In the Streamlit app, we'll load pre-trained vectorizers.

    count_vectorizer = CountVectorizer()
    count_vectorizer.fit(data)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(data)

    return count_vectorizer, tfidf_vectorizer

# Example usage (for testing purposes)
if __name__ == '__main__':
    sample_text = "This is a sample text with some numbers like 123 and punctuation!"
    processed_text = preprocess_text(sample_text)
    print(f"Original: {sample_text}")
    print(f"Processed: {processed_text}")

    sample_data = ["This is the first document.", "This document is the second document.", "And this is the third one."]
    cv, tfidf = get_vectorizers(sample_data)
    print("\nCountVectorizer vocabulary:", cv.vocabulary_)
    print("TfidfVectorizer vocabulary:", tfidf.vocabulary_)


