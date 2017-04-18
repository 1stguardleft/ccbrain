import sys, csv
from gensim.models import word2vec
from konlpy.tag import Twitter
from konlpy.utils import pprint

#트윗 nlp 초기화
twitter = Twitter()

#결과 파일처리를 위한 포인터 생성
#rf = open("./result_w2v.dat", 'w')

#list로 쉽게 사용하기 위해 읽는 옵션을 rt로 한다.
with open('./sample.csv','rt') as csvfile:
#reader를 들여쓰지 않으면 에러가 난다....
    reader = csv.reader(csvfile, delimiter=',')
    samp_lst = list(reader)

x = False

if x == True:
#0번 col은 장애 접수 번호
    print (samp_lst[0][0])
#3번 col은 처리자 
    print (samp_lst[0][3])
    print (twitter.nouns(samp_lst[0][2]))
    print ('test complete')

i      = 0
c_list = []
low    = []
tmp    = []
while i < 500: # nlow:
#while i < 100: # nlow:
    j = 0
    #단순 장애 접수 번호로 스킵한다.
    #low.append(samp_lst[i][0]) 
    low.append(samp_lst[i][3])
    tmp = twitter.nouns(samp_lst[i][2])
    while j < len(tmp): 
        low.append(tmp[j])
        j += 1

    c_list.append(low)
#    rf.write(str(c_list[i]))
#    rf.write("\n")
    low = []
    i += 1
    if i%100 == 0: 
        print ('process row : ',i)

#rf.close()

# 벡터 처리
print('work to word2vec') 
sentences = c_list
model = word2vec.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
model.init_sims(replace=True)

print(sys.argv[1],' 유사단어')
print(model.most_similar(sys.argv[1]))

model.save('result_w2v.model')

