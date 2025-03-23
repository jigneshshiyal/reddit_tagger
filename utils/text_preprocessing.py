import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download necessary NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")

# 1. Remove punctuation and numbers
def remove_punctuation_number(text_list):
    return [re.sub(r'[^\w\s]', '', re.sub(r'\d+', '', text)) for text in text_list]

# 2. Convert text to lowercase
def lowercase_text(text_list):
    return [text.lower() for text in text_list]

# 3. Remove stopwords
def remove_stopwords(text_list):
    stop_words = set(stopwords.words('english'))
    return [" ".join([word for word in text.split() if word.lower() not in stop_words]) for text in text_list]

# 4. Tokenization (words only)
def tokenize_text(text_list):
    return [word_tokenize(text) for text in text_list]

# 5. Stemming - Reduce words to their root form
def stem_text(text_list):
    stemmer = PorterStemmer()
    return [" ".join([stemmer.stem(word) for word in text.split()]) for text in text_list]

# 6. Lemmatization - Reduce words to their base form
def lemmatize_text(text_list):
    lemmatizer = WordNetLemmatizer()
    return [" ".join([lemmatizer.lemmatize(word) for word in text.split()]) for text in text_list]


