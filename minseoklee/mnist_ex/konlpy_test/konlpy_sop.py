from konlpy.tag import Kkma
from konlpy.utils import pprint
import read_csv as rc
import file_save as fs
import json as json

sop_all = rc.read_csv('../../ep/sample.csv')

kkma = Kkma()
data_save = []
for sop in sop_all :
  data_save.append({"title": kkma.pos(sop[1]), "detail": kkma.pos(sop[2]), "respondedor": sop[3]})

fs.save('./result/konlpy_pos_result.json', json.dumps(data_save))
