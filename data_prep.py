import pandas as pd
import random

input_path = "seo_data.csv"

def generate_scores(df):
    random.seed(42)
    df = df.copy()
    df["website_score"] = [random.randint(60, 90) for _ in range(len(df))]
    df["keyword_score"] = [random.randint(5, 30) for _ in range(len(df))]
    df["content_score"] = [random.randint(50, 95) for _ in range(len(df))]
    return df

if __name__ == "__main__":
    df = pd.read_csv(input_path)
    df = generate_scores(df)
    df.to_csv("data_with_labels.csv", index=False)