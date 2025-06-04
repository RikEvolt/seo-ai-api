import pandas as pd

# Đọc file gốc
input_path = "django.csv"      # <-- đổi tên file của bạn tại đây
output_path = "seo_articles.csv"

# Load dữ liệu
df = pd.read_csv(input_path)

# Gộp nội dung SEO lại
def merge_text(row):
    parts = [str(row.get(col, "")) for col in ['title', 'h1', 'h2', 'body_text']]
    return " ".join([p.strip() for p in parts if p.strip() != ""])

# Tạo cột mới body_text (gộp)
df['body_text'] = df.apply(merge_text, axis=1)

# Tạo cột label rỗng (để bạn thêm thủ công từ khóa chính)
df['label_keywords'] = ""

# Giữ lại 2 cột cần thiết
df_out = df[['body_text', 'label_keywords']]

# Lưu file
df_out.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"✅ File đã tạo: {output_path}")
