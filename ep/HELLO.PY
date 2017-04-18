import csv
from konlpy.tag import Twitter

#트윗 nlp 초기화
twitter = Twitter()

#결과 파일처리를 위한 포인터 생성
rf = open("./result.dat", 'w')

#list로 쉽게 사용하기 위해 읽는 옵션을 rt로 한다.
with open('./sample.csv','rt') as csvfile:
#reader를 들여쓰지 않으면 에러가 난다....
    reader = csv.reader(csvfile, delimiter=',')
    samp_lst = list(reader)

#0번 col은 장애 접수 번호
print (samp_lst[0][0])
#3번 col은 처리자 
print (samp_lst[0][3])
print (twitter.nouns(samp_lst[0][2]))
print ('test complete')

i      = 0
c_list = []
low    = []
while i < 500: # nlow:
    low.append(samp_lst[i][0])
    low.append(samp_lst[i][3])
    low.append(twitter.nouns(samp_lst[i][2]))

    c_list.append(low)
    rf.write(str(c_list[i]))
    rf.write("\n")
    low = []
    i += 1
    print ('process row : ',i)

rf.close()

