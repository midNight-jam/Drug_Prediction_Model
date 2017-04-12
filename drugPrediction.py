train_labels = []

#
# def readTrainingData():
#   file = './data/train.dat'
#   with open(file) as f1:
#     for lines in f1:
#       lines = lines.replace('\n',' ').replace('\t',' ')
#       lines = ' '.join(lines.split())
#       document = lines.split(' ')
#       train_labels.append(document[0])
#       train_data.append(document[1:])
#       for i in range(1,len(document)):
#         features.add(document[i])
#
# test_data = []
# def readTestData():
#   file = './data/test.dat'
#   with open(file) as f1:
#     for lines in f1:
#       lines = lines.replace('\n', ' ').replace('\t', ' ')
#       lines = ' '.join(lines.split())
#       document = lines.split(' ')
#       test_data.append(document)

from scipy.sparse import coo_matrix
import numpy as np

def create_Test_csr():
  file = './data/test.dat'
  row_id = 0
  rows = []
  cols = []
  data = []
  with open(file) as f1:
    for lines in f1:
      lines = lines.replace('\n', ' ').replace('\t', ' ')
      lines = ' '.join(lines.split())
      document = lines.split(' ')
      for i in range(0, len(document)):
        rows.append(row_id)
        cols.append(document[i])
        data.append(1)
      row_id += 1
  rows = np.array(rows)
  cols = np.array(cols)
  data = np.array(data)
  return coo_matrix((data, (rows, cols)), shape=(800, 100001)).tocsr()

def create_Train_csr():
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
  return coo_matrix((data,(rows,cols)), shape=(800,100001)).tocsr()

from sklearn.naive_bayes import BernoulliNB
def apply_bernoulliNB(train_csr):
  bnb_clf = BernoulliNB()
  bnb_clf.fit(train_csr,train_labels)
  # print(train_csr[2])
  print(bnb_clf.predict(train_csr[0]))
  print(bnb_clf.predict(train_csr[18]))
  print(bnb_clf.predict(train_csr[19]))
  print(bnb_clf.predict(train_csr[26]))
  print(bnb_clf.predict(train_csr[27]))

from sklearn.svm import SVC
def apply_svc(train_csr, test_csr):
  svc = SVC()
  svc.fit(train_csr,train_labels)
  test_pred = svc.predict(test_csr)
  print(test_pred)

def main():
  train_csr = create_Train_csr()
  test_csr = create_Test_csr()
  # print(train_labels)
  # print(train_csr.toarray())
  # print(test_csr.toarray())
  apply_svc(train_csr,train_csr)

if __name__ == '__main__':
  main()
