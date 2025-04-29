import torch
import pandas as pd
import numpy as np

from transformers import MobileBertForSequenceClassification, MobileBertTokenizer
from tqdm import tqdm

#GPU = torch.cuda.is_available()
GPU = torch.backends.mps.is_available()

device = torch.device("mps" if GPU else "cpu")
print("Using device:", device)

data_path = "spam_emails_remaining_filter.csv"
df = pd.read_csv(data_path, encoding="cp949")

data_X = df['text'].astype(str).tolist()
labels = df['label'].values

print(len(data_X))

tokenizer = MobileBertTokenizer.from_pretrained("mobilebert-uncased", do_lower_case=True)
inputs = tokenizer(data_X, truncation=True, max_length=256, add_special_tokens=True, padding="max_length")

input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

batch_size = 8

test_inputs = torch.tensor(input_ids)
test_labels = torch.tensor(labels)
test_masks = torch.tensor(attention_mask)
test_data = torch.utils.data.TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = torch.utils.data.RandomSampler(test_data)
test_dataloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

model = MobileBertForSequenceClassification.from_pretrained("mobilebert_custom_model.pt")
model.to(device)

model.eval()

test_pred = []
test_true = []

for batch in tqdm(test_dataloader, desc="Inferencing Full Dataset"):
    batch_ids, batch_mask, batch_labels = batch

    batch_ids = batch_ids.to(device)
    batch_mask = batch_mask.to(device)
    batch_labels = batch_labels.to(device)

    with torch.no_grad():
        output = model(batch_ids, attention_mask=batch_mask)

    logits = output.logits
    pred = torch.argmax(logits, dim=1)

    test_pred.extend(pred.cpu().numpy())
    test_true.extend(batch_labels.cpu().numpy())

test_accuracy = np.sum(np.array(test_pred) == np.array(test_true)) / len(test_pred)
print("전체 데이터 64,147건에 대한 스팸/정상 정확도 : ", test_accuracy)