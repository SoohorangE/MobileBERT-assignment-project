import pandas as pd
import nltk
from nltk.corpus import stopwords
from langdetect import detect, LangDetectException
import re

# NLTK의 stopwords 다운로드
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # 특수문자 제거 및 소문자 변환
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    # 불용어 제거
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def is_english(text):
    try:
        return detect(text) == 'en'  # 영어 감지
    except LangDetectException:
        return False

# 원본 CSV 파일 읽기
df = pd.read_csv('spam_emails_data.csv')

# 'label' 컬럼 값이 'Ham' 또는 'Spam'인 행만 필터링
filtered_df = df[df['label'].isin(['Ham', 'Spam'])]

# 'label' 값을 0과 1로 변환
filtered_df['label'] = filtered_df['label'].map({'Ham': 0, 'Spam': 1})

# 'text' 컬럼의 길이가 1024자 이하인 행만 필터링
filtered_df = filtered_df[filtered_df['text'].apply(lambda x: len(x) <= 1024 if isinstance(x, str) else False)]

# 영어로 작성된 텍스트만 필터링
filtered_df = filtered_df[filtered_df['text'].apply(is_english)]

# 텍스트 정제 및 불용어 제거
filtered_df['text'] = filtered_df['text'].apply(clean_text)

# 'label'과 'text' 컬럼이 같은 중복된 행을 제거
filtered_df = filtered_df.drop_duplicates(subset=['label', 'text'])

# 필터링된 데이터를 새로운 CSV 파일로 저장
filtered_df.to_csv('spam_emails_data_filtered.csv', index=False)

print("필터링된 CSV 파일이 저장되었습니다.")
