# ==============================
# Imports
# ==============================

import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer



# ==============================
# Initialize NLP Tools
# ==============================


# English stop words
stop_words = set(
    stopwords.words("english")
)


# Lemmatization tool
lemmatizer = WordNetLemmatizer()


# Stemming tool (Porter Algorithm)
stemmer = PorterStemmer()



# ==============================
# Text Preprocessing Function
# ==============================


def preprocess_text(text):


    # ------------------------------
    # 1. Normalization
    # Convert text to lowercase
    # ------------------------------

    text = text.lower()



    # ------------------------------
    # 2. Remove punctuation
    # Keep only letters, numbers, spaces
    # ------------------------------

    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )



    # ------------------------------
    # 3. Tokenization
    # Split text into words
    # ------------------------------

    tokens = word_tokenize(text)



    # ------------------------------
    # 4. Stopwords Removal
    # Remove common words:
    # the, is, are, etc.
    # ------------------------------

    tokens = [

        word

        for word in tokens

        if word not in stop_words

    ]



    # ------------------------------
    # 5. Lemmatization
    # Convert words to dictionary form
    # running -> running/run
    # cars -> car
    # ------------------------------

    tokens = [

        lemmatizer.lemmatize(word)

        for word in tokens

    ]



    # ------------------------------
    # 6. Porter Stemming
    # Reduce words to root form
    # studies -> studi
    # running -> run
    # ------------------------------

    tokens = [

        stemmer.stem(word)

        for word in tokens

    ]



    # ------------------------------
    # Return processed text
    # ------------------------------

    return " ".join(tokens)

# ==============================
# Test
# ==============================

if __name__ == "__main__":

    sample_text = """
    Running runners are studying Information Retrieval systems!
    The cars were parked outside the universities, and people
    were searching for relevant documents.
    """

    print("Original Text:")
    print(sample_text)

    print("\nProcessed Text:")
    print(preprocess_text(sample_text))