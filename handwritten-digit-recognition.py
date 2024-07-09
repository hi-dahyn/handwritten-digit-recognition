# 이 프로젝트는 코랩에서 실행되었습니다

!pip install nltk

import tensorflow as tf

# GPU 사용 설정
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices("GPU")[0], True)

with tf.device("/GPU:0"):
    x = tf.constant(1.0, dtype=tf.float32)
    y = tf.constant(2.0, dtype=tf.float32)
    z = x + y


import re
import nltk
import seaborn as sns
import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
from nltk.corpus import reuters
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix

# 시그모이드 함수 정의
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 벡터화 함수 정의
def vectorize_data(data):
    return vectorizer.transform(data)

# NLTK에서 로이터 뉴스 말뭉치 데이터를 다운로드.
nltk.download("reuters")

# 로이터 뉴스 말뭉치 데이터의 카테고리 목록을 확인.
categories = reuters.categories()
print("로이터 뉴스 말뭉치 카테고리 목록:")
print(categories)

# 특정 카테고리의 파일 목록을 확인하고 파일의 내용을 출력.
category = "wheat"
file_ids = reuters.fileids(category)
print(f"\n'{category}' 카테고리의 파일 목록:")
print(file_ids)

# 특정 파일의 내용을 확인.
file_id = file_ids[0]
file_content = reuters.raw(file_id)
print(f"\n'{file_id}' 파일 내용:")
print(file_content)

# 데이터 전처리 함수
def preprocess_text(text):
    # URL 제거
    text = re.sub(r"http(s)?://[\w./?%&=~_|#-]+", "", text)
    # 이모티콘 제거
    text = re.sub(r"[^\w\s]|_", "", text)
    # 단어 길이가 2이하인 단어 제거
    text = ' '.join([word for word in text.split() if len(word) >= 2])

    # 불용어 제거
    try:
        stopwords = nltk.corpus.stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")
        stopwords = nltk.corpus.stopwords.words("english")

    text = ' '.join([word for word in text.split() if word not in stopwords])

    # 문자열을 토큰화
    tokens = text.split()
    # Lowercase and remove special characters
    tokens = [token.lower() for token in tokens]
    # 문자열로 다시 결합
    text = ' '.join(tokens)
    return text

x_data = [reuters.raw(file_id) for file_id in reuters.fileids(categories)]

# 데이터 전처리 적용
x_data = [preprocess_text(doc) for doc in x_data]

# x_data의 내용을 확인
for i, doc in enumerate(x_data[:5]):  # 처음 5개의 문서만 출력하도록 설정
    print(f"문서 {i + 1}:\n{doc}\n")

# 데이터 형태 확인
print("x_data의 데이터 형태:")
print(f"데이터 타입: {type(x_data)}")
print(f"데이터 개수: {len(x_data)}")
print(f"샘플 문서 예시:\n{x_data[0]}")  # 첫 번째 문서 출력

# 데이터 분포 확인
word_counts = {}
for doc in x_data:
    words = doc.split()
    for word in words:
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1

# 데이터 분포 출력
print("\n단어 빈도수 분포:")
print(f"고유한 단어 개수: {len(word_counts)}")
print(f"전체 단어 개수: {sum(word_counts.values())}")
print(f"상위 10개 단어 빈도수:")
top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
for word, count in top_words:
    print(f"{word}: {count}")

# categories를 레이블 데이터로 사용
y = [reuters.categories(file_id)[0] for file_id in reuters.fileids()]

#전처리된 데이터를 변수에 저장
x = x_data

# 데이터를 학습 데이터와 테스트 데이터로 나눔.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 학습 데이터와 테스트 데이터의 크기 확인
print(f"학습 데이터 개수: {len(x_train)}")
print(f"테스트 데이터 개수: {len(x_test)}")

# 학습 데이터와 테스트 데이터의 첫 번째 문서 및 레이블 출력
print(f"\n첫 번째 학습 데이터 문서:\n{x_train[0]}")
print(f"첫 번째 학습 데이터 레이블: {y_train[0]}")
print(f"\n첫 번째 테스트 데이터 문서:\n{x_test[0]}")
print(f"첫 번째 테스트 데이터 레이블: {y_test[0]}")

x_train = np.array(x_train)
y_train = np.array(y_train)

# 개수 출력
print(x_train.shape)
print(y_train.shape)

# 텍스트 데이터를 벡터화하기 위해 CountVectorizer를 사용.
with tf.device("/GPU:0"):
    vectorizer = CountVectorizer()
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

# 로지스틱 회귀 분석 모델을 생성하고 학습.
with tf.device("/GPU:0"):
    Lmodel = LogisticRegression(multi_class="multinomial", max_iter=200)
    Lmodel.fit(x_train_vec, y_train)

# 학습된 모델로 예측을 수행.
with tf.device("/GPU:0"):
    y_pred = Lmodel.predict(x_test_vec)

#가중치 출력
w = Lmodel.coef_
w = w.swapaxes(0,1)
print(w.shape)
print(w)

#로지스틱 회귀 모델 예측
print(y_pred.shape)

# accuracy 구하기
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# F1 score 구하기
f1 = f1_score(y_test, y_pred,average='macro')
print(f1)

from sklearn.metrics import log_loss

y_pred = Lmodel.predict_proba(x_test_vec)[:, :60]

# log_loss 계산
loss = log_loss(y_test, y_pred, labels=range(60))
print(loss)

from sklearn.metrics import precision_score, recall_score

# 모델의 예측 결과 (y_pred)와 실제 레이블 (y_test)을 사용하여 정밀도와 재현율을 계산
precision = precision_score(y_test, Lmodel.predict(x_test_vec), average='macro')
recall = recall_score(y_test, Lmodel.predict(x_test_vec), average='macro')

# 정밀도와 재현율 출력
print("정밀도 (Precision):", precision)
print("재현율 (Recall):", recall)

def hyperparameter_tuning(x_train_vec, y_train):

    # 로지스틱 회귀 분석 모델을 학습하기 위한 하이퍼파라미터의 범위를 설정.
    param_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
    }

    # GridSearchCV를 사용하여 하이퍼파라미터를 최적화.
    with tf.device("/GPU:0"):
        grid_search = GridSearchCV(
            LogisticRegression(),
            param_grid=param_grid,
            cv=5,
            scoring="accuracy",
        )
        grid_search.fit(x_train_vec, y_train)

    # 최적의 하이퍼파라미터를 반환.
    return grid_search.best_params_

if __name__ == "__main__":
    # 하이퍼파라미터 튜닝을 수행.
    best_hyperparameters = hyperparameter_tuning(x_train_vec, y_train)

def evaluate_model(x_test_vec, y_test, best_hyperparameters):
    # 최적의 모델을 생성.
    best_model = LogisticRegression(**best_hyperparameters)

    # 모델을 학습.
    best_model.fit(x_train_vec, y_train)

    return best_model

from sklearn.metrics import f1_score as calculate_f1_score

if __name__ == "__main__":
    # 모델을 평가.
    best_model = evaluate_model(x_test_vec, y_test, best_hyperparameters)

    # 모델을 사용하여 예측을 수행.
    y_pred = best_model.predict(x_test_vec)

    # 정확도 계산
    accuracy2 = accuracy_score(y_test, y_pred)
    print("정확도 (Accuracy):", accuracy2)

    # 정밀도 계산
    precision2 = precision_score(y_test, y_pred, average='macro')
    print("정밀도 (Precision):", precision2)

    # 재현율 계산
    recall2 = recall_score(y_test, y_pred, average='macro')
    print("재현율 (Recall):", recall2)

    # F1 점수 계산
    f2 = calculate_f1_score(y_test, y_pred, average="weighted")
    print("F1 점수 (F1 Score):", f2)

# regularization_strength: 정규화 강도
coefficients = np.random.rand(x_train_vec.shape[1])  # 랜덤 초기화
regularization_strength = 0.1

# 학습 파라미터 설정
learning_rate = 0.01
num_epochs = 100

# 혼동 행렬(Confusion Matrix)을 계산산.
with tf.device("/GPU:0"):
  cm = confusion_matrix(y_test, y_pred)

# 혼동 행렬을 시각화.
plt.figure(figsize=(20, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("혼동 행렬 (Confusion Matrix)")
plt.xlabel("예측 레이블")
plt.ylabel("실제 레이블")
plt.show()

# 분류 보고서를 출력.
print("\n분류 보고서:")
print(classification_report(y_test, y_pred))

# 테스트용 텍스트 문장을 생성.
test_sentences = [
    "This is a positive review.",
    "The product is not good.",
    "I like this book.",
    "The movie was terrible.",
    "The weather is nice today.",
]

# 텍스트 데이터를 벡터화.
x_test_custom = vectorizer.transform(test_sentences)

# 모델로 테스트 데이터에 대한 예측을 수행.
y_pred_custom = best_model.predict(x_test_custom)

# 예측 결과 및 실제 레이블 출력 및 시각화
for i, sentence in enumerate(test_sentences):
    print(f"텍스트: {sentence}")
    print(f"예측 레이블: {y_pred_custom[i]}")
    print("--------------")

# 그래프 X
# 예측 결과를 시각화
plt.figure(figsize=(8, 6))
plt.bar(test_sentences, y_pred_custom)
plt.xlabel("텍스트 문장")
plt.ylabel("예측 레이블")
plt.title("텍스트 문장별 예측 결과")
plt.xticks(rotation=15)
plt.show()
