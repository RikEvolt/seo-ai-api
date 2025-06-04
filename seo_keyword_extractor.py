import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from utils.text_cleaner import clean_text

text = "SEO là một phần quan trọng trong chiến lược nội dung và tiêu đề bài viết. Từ khóa cần được tối ưu."
text_clean = clean_text(text)

# Giả lập tokenizer như khi huấn luyện
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts([text_clean])
seq = tokenizer.texts_to_sequences([text_clean])
seq_pad = pad_sequences(seq, maxlen=200)

model = tf.keras.models.load_model("models/keyword_model.h5")
pred = model.predict(seq_pad)

keywords = [kw for kw, idx in tokenizer.word_index.items() if idx < 20]  # Top 20 từ
for kw in keywords:
    print(f"✔ Keyword: {kw} (approx)")
