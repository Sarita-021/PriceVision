from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

def build_tfidf_ridge(
    max_features=50000,
    ngram_range=(1, 2),
    alpha=1.0
):
    """
    TF-IDF + Ridge regression pipeline
    """
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words="english"
    )

    ridge = Ridge(alpha=alpha, random_state=42)

    pipeline = Pipeline([
        ("tfidf", tfidf),
        ("ridge", ridge)
    ])

    return pipeline
