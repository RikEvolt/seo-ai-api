def clean_text(text):
    import re
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text