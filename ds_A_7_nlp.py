import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import math
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('punkt',      quiet=True)
nltk.download('stopwords',  quiet=True)
nltk.download('wordnet',    quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
new_doc = "I had high hopes for the latest smartwatch, but the battery life is disappointing. It barely lasts a day even with minimal use."
sample_documents = [
    "Natural Language Processing (NLP) is a facinating field of Artificial Intelligence.",
    "Text Analytics involves preprocessing steps such as tokenization and stemming."
]
def preprocess(doc):
    tokens = word_tokenize(doc)
    # POS tagging (optional, shown for completeness)
    pos_tags = pos_tag(tokens)

    # lowercase & alphabetic & remove stopwords
    stops = set(stopwords.words('english'))
    filtered = [w.lower() for w in tokens if w.isalpha() and w.lower() not in stops]

    # stemming & lemmatization
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stems  = [stemmer.stem(w)    for w in filtered]
    lemmas = [lemmatizer.lemmatize(w) for w in filtered]

    return {
        "tokens": tokens,
        "pos_tags": pos_tags,
        "filtered": filtered,
        "stems": stems,
        "lemmas": lemmas
    }
p_new = preprocess(new_doc)

print("Original tokens:   ", p_new['tokens'])
print("POS Tags:          ", p_new['pos_tags'])
print("Filtered Tokens:   ", p_new['filtered'])
print("Stems:             ", p_new['stems'])
print("Lemmas:            ", p_new['lemmas'])
processed = [preprocess(doc) for doc in sample_documents]
def compute_tf(tokens):
    tf = {}
    n = len(tokens)
    for w in tokens:
        tf[w] = tf.get(w, 0) + 1
    return {w: count / n for w, count in tf.items()}
def compute_idf(list_of_token_lists):
    idf = {}
    N = len(list_of_token_lists)
    all_terms = set(w for tokens in list_of_token_lists for w in tokens)
    for term in all_terms:
        df = sum(1 for tokens in list_of_token_lists if term in tokens)
        # add-1 smoothing
        idf[term] = math.log((N / (1 + df))) + 1
    return idf
def compute_tfidf(tf, idf):
    return {w: tf_val * idf.get(w, 0) for w, tf_val in tf.items()}
# Gather filtered token lists
all_filtered = [p['filtered'] for p in processed]
idf_manual = compute_idf(all_filtered)
for i, p in enumerate(processed, start=1):
    tf_manual    = compute_tf(p['filtered'])
    tfidf_manual = compute_tfidf(tf_manual, idf_manual)
    print(f"\n--- DOCUMENT {i} ---")
    print("Filtered Tokens:", p['filtered'])
    print(" TF:", tf_manual)
    print(" IDF:", {w: idf_manual[w] for w in p['filtered']})
    print(" TF–IDF:", tfidf_manual)
all_docs = [new_doc] + sample_documents
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_mat = vectorizer.fit_transform(all_docs)
tfidf_df = pd.DataFrame(
    tfidf_mat.toarray(),
    index=["New Doc", "Doc 1", "Doc 2"],
    columns=vectorizer.get_feature_names_out()
)

print("\nTF–IDF Matrix (scikit-learn):")
print(tfidf_df)