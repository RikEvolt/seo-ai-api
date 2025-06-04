import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, GlobalMaxPool1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle
import os

# Load dữ liệu
csv_file = "data/data_with_labels.csv"
df = pd.read_csv(csv_file)

# Kết hợp nội dung văn bản
df['combined_text'] = df['title'].fillna('') + ' ' + df['snippet'].fillna('') + ' ' + df['h1'].fillna('')
texts = df['combined_text'].astype(str)

# Gán nhãn nhị phân: 1 nếu có từ "SEO" trong title + snippet
df['keywords'] = df['title'].fillna('') + ' ' + df['snippet'].fillna('')
labels = df['keywords'].str.contains('SEO', case=False).astype(int)

# Tiền xử lý văn bản
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(texts)  # <-- ĐÂY mới là nơi tokenizer được định nghĩa!
sequences = tokenizer.texts_to_sequences(texts)
data = pad_sequences(sequences, maxlen=200)

# Chia dữ liệu train/val
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

# Xây dựng mô hình
model = Sequential([
    Embedding(input_dim=5000, output_dim=128),
    LSTM(64, return_sequences=True),
    GlobalMaxPool1D(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Huấn luyện
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Tạo thư mục nếu chưa có
os.makedirs("models", exist_ok=True)

# Lưu mô hình và tokenizer
model.save("models/keyword_model.h5")

with open("models/Tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("✅ Đã huấn luyện xong mô hình và lưu tokenizer thành công.")
