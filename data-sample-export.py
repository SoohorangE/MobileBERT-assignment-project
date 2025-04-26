import pandas as pd

# 1. filter.csv 파일 읽기
df = pd.read_csv('spam_emails_data_filtered.csv', encoding='cp949')

# 2. label 값이 0인 데이터와 1인 데이터 추출
label_0 = df[df['label'] == 0]
label_1 = df[df['label'] == 1]

# 3. 각 label에 대해 19000개씩 샘플링
sample_label_0 = label_0.sample(n=19000, random_state=42)
sample_label_1 = label_1.sample(n=19000, random_state=42)

# 4. 샘플링된 데이터 합치고 섞기
sampled_data = pd.concat([sample_label_0, sample_label_1]).sample(frac=1, random_state=42).reset_index(drop=True)

# 5. 나머지 데이터 추출 (나머지 데이터를 추출)
remaining_label_0 = label_0.drop(sample_label_0.index)  # label 0 나머지
remaining_label_1 = label_1.drop(sample_label_1.index)  # label 1 나머지

# 6. 나머지 데이터를 합치고 섞기
remaining_data = pd.concat([remaining_label_0, remaining_label_1]).sample(frac=1, random_state=42).reset_index(drop=True)

# 7. 샘플링된 데이터 확인
print("샘플링된 데이터의 일부:")
print(sampled_data.head())

# 8. 나머지 데이터 확인
print("나머지 데이터의 일부:")
print(remaining_data.head())

# 9. 샘플링된 데이터를 새로운 CSV 파일로 저장
sampled_data.to_csv('spam_emails_sampled_filter.csv', index=False, encoding='cp949')

# 10. 나머지 데이터를 새로운 CSV 파일로 저장
remaining_data.to_csv('spam_emails_remaining_filter.csv', index=False, encoding='cp949')
