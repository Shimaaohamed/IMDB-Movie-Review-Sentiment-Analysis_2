# test.py
import joblib
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from preprocessing import load_and_preprocess
from sklearn.model_selection import train_test_split
import pandas as pd

def evaluate(model_path="model_imdb.pkl", data_path="data/IMDB Dataset.csv"):
    # load data (preprocessed)
    df = load_and_preprocess(data_path, text_col="review", label_col="sentiment")
    X = df["review"]
    y = df["sentiment"]

    # use fixed split same as train script
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # load model
    model = joblib.load(model_path)

    print("Predicting on test set...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                       index=["neg_true","pos_true"],
                       columns=["neg_pred","pos_pred"]))

if __name__ == "__main__":
    evaluate()
