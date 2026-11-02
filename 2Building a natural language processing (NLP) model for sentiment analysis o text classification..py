import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

data = {
    "text": [
        "I love this product",
        "Amazing quality and performance",
        "Very satisfied with the purchase",
        "Best product I have used",
        "Excellent service and support",
        "Terrible quality",
        "Worst experience ever",
        "Very disappointed with the product",
        "Awful customer service",
        "Waste of money",
        "It is okay",
        "Average product",
        "Nothing special about it",
        "Decent quality",
        "Can be improved"
    ],
    "label": [
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "neutral",
        "neutral",
        "neutral",
        "neutral",
        "neutral"
    ]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in string.punctuation]

    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    return " ".join(tokens)

df["cleaned_text"] = df["text"].apply(preprocess_text)

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(df["cleaned_text"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

model = MultinomialNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

def predict_sentiment(text):
    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)
    return prediction[0]

new_text = "This is the best product I have ever bought"

print("\nTest Sentence:")
print(new_text)

print("\nPredicted Sentiment:")
print(predict_sentiment(new_text))
