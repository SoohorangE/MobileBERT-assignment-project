import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import MobileBertForSequenceClassification, MobileBertTokenizer

# ======  1. Device 설정 ======
GPU = torch.cuda.is_available()
device = torch.device("cuda" if GPU else "cpu")
print("Using device:", device)

# ======  2. 데이터 로드 ======
data_path = "spam_emails_remaining_filter.csv"
df = pd.read_csv(data_path, encoding="cp949")

# 샘플 하나 선택
sample_idx = 4 ## 0부터 1번 데이터 :  예시 데이터 : 32(스팸), 4번(정상)
sample_text = df['text'].loc[sample_idx]
sample_label = df['label'].loc[sample_idx]  # 0=정상, 1=스팸

print(f"\n[예제 {sample_idx}] (Label: {sample_label})")
print(sample_text)

# ======  3. 모델 및 토크나이저 로드 ======
tokenizer = MobileBertTokenizer.from_pretrained("mobilebert-uncased", do_lower_case=True)
model = MobileBertForSequenceClassification.from_pretrained("mobilebert_custom_model.pt")
model.to(device)
model.eval()

# ======  4. 입력 데이터 변환 ======
inputs = tokenizer(sample_text, return_tensors="pt", truncation=True, max_length=256)
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

# ======  5. 모델 실행 및 Attention Weights 가져오기 ======
with torch.no_grad():
    output = model.mobilebert(input_ids, attention_mask=attention_mask, output_attentions=True)

# 모든 Attention Layers에서 가중치 가져오기
attentions = torch.stack(output.attentions)  # Shape: (num_layers, batch, num_heads, seq_len, seq_len)

# 마지막 레이어에서 첫 번째 Attention Head 선택
layer_idx = -1  # 마지막 레이어
head_idx = 0  # 첫 번째 Attention Head
attn_matrix = attentions[layer_idx][0, head_idx].cpu().numpy()  # (seq_len, seq_len)

# 토큰 변환
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

# ======  6. Attention 가중치 시각화 ======
plt.figure(figsize=(10, 8))
sns.heatmap(attn_matrix, xticklabels=tokens, yticklabels=tokens, cmap="coolwarm")
plt.xlabel("Key Tokens")
plt.ylabel("Query Tokens")
plt.title(f"MobileBERT Attention Weights (Example {sample_idx})")
plt.show()

# ======  7. 중요 단어 출력 (각 토큰의 평균 Attention 점수 계산) ======
importance_scores = attn_matrix.mean(axis=0)  # 각 토큰별 평균 중요도 계산
top_n = 5  # 중요 단어 5개 추출

# 토큰과 중요도를 함께 정렬하여 상위 n개 출력
important_tokens_attention = sorted(zip(tokens, importance_scores), key=lambda x: x[1], reverse=True)[:top_n]

print("\n🔍 Attention 가중치 기준 중요 단어 TOP 5:")
for token, score in important_tokens_attention:
    print(f"{token}: {score:.4f}")

# ======  8. Fully Connected Network (FCN) 가중치 분석 ======
with torch.no_grad():
    full_output = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
    print("예측값:", full_output[0])

#  (1) [CLS] 벡터 가져오기 (최종 레이어에서 나오는 벡터)
cls_embedding = full_output.hidden_states[-1][:, 0, :].cpu().numpy()  # (1, hidden_dim)

# [CLS] 벡터에서 가장 큰 값의 인덱스 찾기g
top_cls_idx = np.argmax(cls_embedding)

#  (2) Fully Connected Layer 가중치 가져오기
# Fully Connected Network (FCN) 가중치 가져오기 (detached)
fc_weights = model.classifier.weight.detach().cpu().numpy()  # (num_classes, hidden_dim)
fc_bias = model.classifier.bias.detach().cpu().numpy()  # (num_classes,)

#  (3) 스팸 vs 정상 판별을 위한 로짓 계산
logits = np.dot(fc_weights, cls_embedding.T).flatten() + fc_bias  # (num_classes,)
spam_prob = np.exp(logits[1]) / (np.exp(logits[0]) + np.exp(logits[1]))  # Softmax 적용

#  (4) Fully Connected Layer에서 중요한 특성 확인
feature_importance = np.abs(fc_weights[1])  # 스팸(1) 클래스의 가중치 절댓값

# 가장 중요한 특성 5개 찾기
top_features = np.argsort(feature_importance)[-5:]

print("\n Fully Connected Network 분석 ")
print("최종 [CLS] 벡터:", cls_embedding[0][:10])
print("FCN 가중치 W (Spam class):", fc_weights[1][:10])  # 일부 가중치 출력
print("FCN Bias:", fc_bias)
print(f"\n📊 스팸 확률: {spam_prob:.4f}")

# FCN에서 중요한 특성 TOP 5 (가중치 기준) 출력
print("\n🔍 FCN에서 중요한 특성 TOP 5 (가중치 기준):")
for idx in top_features:
    if idx < len(tokens):  # 토큰 범위를 벗어나지 않도록 조건 추가
        print(f"특성 {idx}에 해당하는 단어: {tokens[idx]}")  # 여기서 idx에 해당하는 단어 출력