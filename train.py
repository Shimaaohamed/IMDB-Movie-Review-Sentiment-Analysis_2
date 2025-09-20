# train.py
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from preprocessing import load_and_preprocess

def train_and_save(path="data/IMDB Dataset.csv",
                   model_path="model_imdb.pkl",
                   test_size=0.2, random_state=42):
    # load & preprocess
    df = load_and_preprocess(path, text_col="review", label_col="sentiment")
    X = df["review"]
    y = df["sentiment"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Pipeline: TF-IDF + LogisticRegression
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    print("Training model...")
    pipeline.fit(X_train, y_train)
    print("Done training. Saving model to", model_path)
    joblib.dump(pipeline, model_path)
    print("✅ Model saved.")

    # Optional: quick validation score
    val_score = pipeline.score(X_val, y_val)
    print(f"Validation accuracy: {val_score:.4f}")

if __name__ == "__main__":
    train_and_save()
