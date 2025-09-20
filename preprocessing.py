# preprocessing.py
import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# تنزيل بيانات NLTK اللازمة (مرّة أولى)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    """نظف النص: lowercase, ازالة روابط، HTML، علامات ترقيم، أرقام، stopwords، وlemmatize."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    # احذف علامات الترقيم
    text = text.translate(str.maketrans('', '', string.punctuation))
    # احذف أرقام
    text = re.sub(r'\d+', '', text)
    # توكن ثم ازالة stopwords و lemmatize
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in stop_words]
    return " ".join(tokens)

def load_and_preprocess(path="data/IMDB Dataset.csv", text_col="review", label_col="sentiment"):
    """
    يقرأ CSV، يطبق تنظيف، ويعيد DataFrame جاهز.
    يتوقع وجود عمود 'review' و 'sentiment'.
    """
    df = pd.read_csv(path)
    # حذف الصفوف الفارغة في الأعمدة المطلوبة
    df = df.dropna(subset=[text_col, label_col]).reset_index(drop=True)
    # تنظيف النصوص
    df[text_col] = df[text_col].astype(str).apply(clean_text)
    return df
