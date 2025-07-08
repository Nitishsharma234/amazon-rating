# === Imports ===
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import nltk
import warnings
warnings.filterwarnings("ignore")

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# === Streamlit Config ===
st.set_page_config(page_title="Amazon Alexa Sentiment Dashboard", layout="wide")

# === NLTK Setup ===
nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()

# === Load and Train Models ===
@st.cache_resource
def train_models():
    data = pd.read_csv("amazon_alexa.tsv", sep="\t", quoting=3)
    data.dropna(inplace=True)

    corpus = []
    for review in data['verified_reviews']:
        review = re.sub('[^a-zA-Z]', ' ', review)
        review = review.lower().split()
        review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
        corpus.append(' '.join(review))

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(corpus).toarray()
    y = data['feedback'].values

    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.3, random_state=15)
    scaler = MinMaxScaler()
    X_train_scl = scaler.fit_transform(X_train)
    X_test_scl = scaler.transform(X_test)

    rf_model = RandomForestClassifier()
    rf_model.fit(X_train_scl, y_train)

    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train_scl, y_train)

    return rf_model, xgb_model, vectorizer, scaler, X_train_scl, X_test_scl, y_train, y_test, data, corpus

# === Get models & data ===
rf_model, xgb_model, vectorizer, scaler, X_train, X_test, y_train, y_test, df, corpus = train_models()

# === Page UI ===
st.title("🧠 Amazon Alexa Review Classifier")
st.markdown("Predict sentiment & explore data insights with charts and models (TF-IDF + SMOTE + Bigrams).")

# === Sidebar ===
with st.sidebar:
    st.header("📊 Model and Input")
    review = st.text_area("✍️ Enter your review here")
    model_choice = st.selectbox("Choose model", ("XGBoost", "Random Forest"))
    threshold = st.slider("Prediction Threshold", 0.5, 0.9, 0.72, 0.01)
    show_dashboard = st.checkbox("📈 Show Dashboard")

# === Sentiment Prediction ===
if review:
    def clean_review(text):
        review = re.sub('[^a-zA-Z]', ' ', text)
        review = review.lower().split()
        review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
        return ' '.join(review)

    cleaned = clean_review(review)
    vect = vectorizer.transform([cleaned]).toarray()
    scaled = scaler.transform(vect)

    if model_choice == "XGBoost":
        proba = xgb_model.predict_proba(scaled)[0]
    else:
        proba = rf_model.predict_proba(scaled)[0]

    pred = 1 if proba[1] > threshold else 0
    sentiment = "😊 Positive (1)" if pred == 1 else "☹️ Negative (0)"
    st.success(f"Prediction: {sentiment}")
    st.info(f"Confidence: Positive={proba[1]:.2f}, Negative={proba[0]:.2f} | Threshold: {threshold}")

# === Dashboard Section ===
if show_dashboard:
    st.subheader("📊 Review Rating Distribution")
    st.bar_chart(df['rating'].value_counts().sort_index())

    st.subheader("☁️ WordCloud of All Reviews")
    all_reviews = " ".join(corpus)
    wordcloud = WordCloud(background_color="white", max_words=100).generate(all_reviews)
    fig_wc, ax_wc = plt.subplots()
    ax_wc.imshow(wordcloud, interpolation='bilinear')
    ax_wc.axis('off')
    st.pyplot(fig_wc)

    st.subheader("📈 Model Accuracy Comparison")
    acc_rf = accuracy_score(y_test, rf_model.predict(X_test))
    acc_xgb = accuracy_score(y_test, xgb_model.predict(X_test))

    acc_df = pd.DataFrame({
        'Model': ['Random Forest', 'XGBoost'],
        'Accuracy': [acc_rf, acc_xgb]
    })

    fig_acc, ax_acc = plt.subplots()
    sns.barplot(x='Model', y='Accuracy', data=acc_df, palette='coolwarm', ax=ax_acc)
    ax_acc.set_ylim(0.7, 1.0)
    for i, val in enumerate(acc_df['Accuracy']):
        ax_acc.text(i, val + 0.005, f"{val:.2f}", ha='center')
    st.pyplot(fig_acc)

    st.subheader("📉 Confusion Matrix")
    model = rf_model if model_choice == "Random Forest" else xgb_model
    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)
