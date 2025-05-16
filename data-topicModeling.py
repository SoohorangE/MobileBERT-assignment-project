import pandas as pd
import re
import nltk
from gensim import corpora
from gensim.models import LdaModel
import pyLDAvis.gensim_models
import pyLDAvis

# nltk 데이터 다운로드
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# 데이터 불러오기
data = pd.read_csv("spam_emails_model_predict.csv")

custom_stopwords = set(['escapenumber', 'escapelong'] + stopwords.words('english'))

# 전처리 함수
# 불용어 제거 (2글자 이상), english + 문장에 이스케이프 된 단어 escapenumber, escapelong 제거
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in custom_stopwords and len(t) > 2]
    return tokens

# 전처리 적용
data['clean_text'] = data['text'].apply(preprocess)

# Gensim LDA 모델링 함수
def topic_modeling_gensim(texts, n_topics=5, html_filename="lda_vis.html"):
    # dictionary와 corpus 만들기
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # LDA 모델 학습
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=n_topics, random_state=42)

    # 토픽 출력
    topics = lda.print_topics(num_words=10)
    for idx, topic in topics:
        print(f"Topic {idx + 1}: {topic}")

    # pyLDAvis 시각화
    vis = pyLDAvis.gensim_models.prepare(lda, corpus, dictionary)
    pyLDAvis.save_html(vis, html_filename)

    return lda

print("== 토픽 ==")
lda_spam = topic_modeling_gensim(data['clean_text'], n_topics=6, html_filename="spam_topics.html")