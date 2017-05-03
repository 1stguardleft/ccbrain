import csv
def read_csv(file_name) :
  matrix = []
  f = open(file_name, 'r')
  csvReader = csv.reader(f)
  for row in csvReader:
    matrix.append(row)
  f.close()
  return matrix