train_labels = []
train_data = []
features = set()

def readTrainingData():
  file = './data/train.dat'
  with open(file) as f1:
    for lines in f1:
      lines = lines.replace('\n',' ').replace('\t',' ')
      lines = ' '.join(lines.split())
      document = lines.split(' ')
      train_labels.append(document[0])
      train_data.append(document[1:])
      for i in range(1,len(document)):
        features.add(document[i])

test_data = []
def readTestData():
  file = './data/test.dat'
  with open(file) as f1:
    for lines in f1:
      lines = lines.replace('\n', ' ').replace('\t', ' ')
      lines = ' '.join(lines.split())
      document = lines.split(' ')
      test_data.append(document)

from scipy.sparse import  coo_matrix
import numpy as np

def create_COO_Training():
  file = './data/train.dat'
  row_id = 0
  rows = []
  cols= []
  data = []
  with open(file) as f1:
    for lines in f1:
      lines = lines.replace('\n',' ').replace('\t',' ')
      lines = ' '.join(lines.split())
      document = lines.split(' ')
      train_labels.append(document[0])
      for i in range(1,len(document)):
        rows.append(row_id)
        cols.append(document[i])
        data.append(1)
      row_id += 1
  rows = np.array(rows)
  cols = np.array(cols)
  data = np.array(data)
  print(len(rows))
  print(len(cols))
  print(len(data))
  train_coo_matrix = coo_matrix((data,(rows,cols)), shape=(800,100001))
  return train_coo_matrix

def main():
  # readTrainingData()
  # readTestData()
  train_coo_matrix = create_COO_Training()
  print(train_coo_matrix.toarray())

if __name__ == '__main__':
  main()
