import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("data/data_with_labels.csv")

df["text"] = df["title"].fillna("") + " " + df["h1"].fillna("") + " " + df["snippet"].fillna("")

X_train, X_test, y_train, y_test = train_test_split(df["text"], df["website_score"], test_size=0.2, random_state=42)

model = make_pipeline(
    TfidfVectorizer(max_features=1000),
    Ridge()
)
model.fit(X_train, y_train)
preds = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, preds))
