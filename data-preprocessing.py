import pandas as pd
from langdetect import detect, LangDetectException

# 원본 CSV 파일 읽기
df = pd.read_csv('spam_emails_data.csv')

# 'label' 컬럼 값이 'Ham' 또는 'Spam'인 행만 필터링
filtered_df = df[df['label'].isin(['Ham', 'Spam'])]

# 'text' 컬럼의 길이가 1024자 이하인 행만 필터링
filtered_df = filtered_df[filtered_df['text'].apply(lambda x: len(x) <= 1024 if isinstance(x, str) else False)]

# 영어로 작성된 텍스트만 필터링
def is_english(text):
    try:
        return detect(text) == 'en'  # 텍스트가 영어인 경우 True 반환
    except LangDetectException:
        return False  # 언어 감지에 실패하면 False 반환

filtered_df = filtered_df[filtered_df['text'].apply(is_english)]

# 라벨을 'Ham'이면 0, 'Spam'이면 1로 변환
filtered_df['label'] = filtered_df['label'].apply(lambda x: 0 if x == 'Ham' else 1)

# 필터링된 데이터를 새로운 CSV 파일로 저장
filtered_df.to_csv('spam_emails_data-filter.csv', index=False)

print("필터링된 CSV 파일이 저장되었습니다.")