def save(file_pos, str) :
  out = open(file_pos, 'w')
  print(str, file=out)
  out.close()