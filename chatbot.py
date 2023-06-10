import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SimpleChatBot:
    def __init__(self, filepath):
        self.questions, self.answers = self.load_data(filepath)
        self.vectorizer = TfidfVectorizer()
        self.question_vectors = self.vectorizer.fit_transform(
            self.questions)  # 질문을 TF-IDF로 변환

    def load_data(self, filepath):
        data = pd.read_csv(filepath)
        questions = data['Q'].tolist()  # 질문열만 뽑아 파이썬 리스트로 저장
        answers = data['A'].tolist()   # 답변열만 뽑아 파이썬 리스트로 저장
        return questions, answers

    def find_best_answer(self, input_sentence):
        input_vector = self.vectorizer.transform([input_sentence])
        similarities = cosine_similarity(
            input_vector, self.question_vectors)  # 코사인 유사도 값들을 저장

        best_match_index = similarities.argmax()   # 유사도 값이 가장 큰 값의 인덱스를 반환
        return self.answers[best_match_index]

########################################################
# Levenshtein Distance
########################################################


def calc_distance(a, b):
    ''' 레벤슈타인 거리 계산하기 '''
    if a == b:
        return 0  # 같으면 0을 반환
    a_len = len(a)  # a 길이
    b_len = len(b)  # b 길이
    if a == "":
        return b_len
    if b == "":
        return a_len
    # 2차원 표 (a_len+1, b_len+1) 준비하기 --- (※1)
    # matrix 초기화의 예 : [[0, 1, 2, 3], [1, 0, 0, 0, 0], [2, 0, 0, 0, 0], [3, 0, 0, 0, 0], [4, 0, 0, 0, 0]]
    # [0, 1, 2, 3]
    # [1, 0, 0, 0]
    # [2, 0, 0, 0]
    # [3, 0, 0, 0]
    matrix = [[] for i in range(a_len+1)]  # 리스트 컴프리헨션을 사용하여 1차원 초기화
    for i in range(a_len+1):  # 0으로 초기화
        matrix[i] = [0 for j in range(b_len+1)]  # 리스트 컴프리헨션을 사용하여 2차원 초기화
    # 0일 때 초깃값을 설정
    for i in range(a_len+1):
        matrix[i][0] = i
    for j in range(b_len+1):
        matrix[0][j] = j
    # 표 채우기 --- (※2)
    # print(matrix, '----------')
    for i in range(1, a_len+1):
        ac = a[i-1]
        # print(ac, '=============')
        for j in range(1, b_len+1):
            bc = b[j-1]
            # print(bc)
            # 파이썬 조건 표현식 예:) result = value1 if condition else value2
            cost = 0 if (ac == bc) else 1
            matrix[i][j] = min([
                matrix[i-1][j] + 1,     # 문자 제거: 위쪽에서 +1
                matrix[i][j-1] + 1,     # 문자 삽입: 왼쪽 수에서 +1
                matrix[i-1][j-1] + cost  # 문자 변경: 대각선에서 +1, 문자가 동일하면 대각선 숫자 복사
            ])
        #     print(matrix)
        # print(matrix, '----------끝')
    return matrix[a_len][b_len]


# CSV 파일 경로를 지정하세요.
filepath = 'ChatbotData.csv'

# 간단한 챗봇 인스턴스를 생성합니다.
chatbot = SimpleChatBot(filepath)

#################################################################################
# 1. 학습데이터의 질문과 chat의 질문의 유사도를 레벤슈타인 거리를 이용해 구하기
#################################################################################
print("#####################################################################")
print('1. 학습데이터의 질문과 chat의 질문의 유사도를 레벤슈타인 거리를 이용해 구하기')
print("#####################################################################")
input_sentence_1 = input('You: ')

data_1 = pd.read_csv(filepath)
questions_1 = data_1['Q'].tolist()

# 레벤슈타인 거리값저장을 위한 컬럼 추가
data_1['distanceValue'] = 0

# 인덱스 번호저장을 위한 컬럼 추가
data_1['index'] = 0

# 각행의 학습데이터에 거리값과 인덱스 값 대입
for i in range(len(data_1)):
    data_1.loc[i, 'index'] = i
    data_1.loc[i, 'distanceValue'] = calc_distance(
        input_sentence_1, data_1.loc[i, 'Q'])

# 데이터프레임의 레벤슈타인 거리값과 인덱스 값 출력확인
for i in range(len(data_1)):
    print(data_1.loc[i, "distanceValue"], data_1.loc[i, "index"])

#################################################################################
# 2. chat의 질문과 레벤슈타인 거리와 가장 유사한 학습데이터의 질문의 인덱스를 구하기
#################################################################################
minRecord = data_1.loc[data_1["distanceValue"].idxmin()]
print("#####################################################################")
print("2. chat의 질문과 레벤슈타인 거리와 가장 유사한 학습데이터의 질문의 인덱스를 구하기")
print(minRecord)
print("인덱스 :", minRecord['index'])
print("#####################################################################")


#################################################################################
# 3. 학습 데이터의 인덱스의 답을 chat의 답변을 채택한 뒤 출력
#################################################################################
print("#####################################################################")
print("3. 학습 데이터의 인덱스의 답을 chat의 답변을 채택한 뒤 출력")
print("답 :", minRecord['A'])
print("#####################################################################")


# 추후 while문을 활용하여 챗봇과 대화를 할수 있음.

# '종료'라는 단어가 입력될 때까지 챗봇과의 대화를 반복합니다.
# while True:
#     input_sentence = input('You: ')
#     if input_sentence.lower() == '종료':
#         break
#     response = chatbot.find_best_answer(input_sentence)
#     print('Chatbot:', response)
