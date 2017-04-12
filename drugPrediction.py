train_labels = []

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
  return coo_matrix((data, (rows, cols)), shape=(350, 100001)).tocsr()

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


def write_to_file(pred_results, file_name):
  f = open(file_name,'w')
  for p in pred_results:
    f.write(p+'\n')
  f.close()

from sklearn.naive_bayes import BernoulliNB
def apply_bernoulliNB(train_csr, test_csr):
  bnb_clf = BernoulliNB()
  bnb_clf.fit(train_csr,train_labels)
  pred_results = bnb_clf.predict(test_csr)
  print('-----------------------BernaouliNB-----------------------')
  print(pred_results)
  print('----------------------------------------------')
  write_to_file(pred_results, 'results_bernouliNB.dat')

from sklearn.ensemble import RandomForestClassifier
def apply_randomforest(train_csr, test_csr):
  random_forest = RandomForestClassifier()
  random_forest.fit(train_csr, train_labels)
  pred_results = random_forest.predict(test_csr)
  print('-----------------------RandomForest-----------------------')
  print(pred_results)
  print('----------------------------------------------')
  write_to_file(pred_results,'results_randomForest.dat')

from sklearn.tree import DecisionTreeClassifier
def apply_DecisionTree(train_csr, test_csr):
  decision_tree = DecisionTreeClassifier()
  decision_tree.fit(train_csr, train_labels)
  pred_results = decision_tree.predict(test_csr)
  print('-----------------------decision tree-----------------------')
  print(pred_results)
  print('----------------------------------------------')
  write_to_file(pred_results,'results_decisionTree.dat')

from sklearn.linear_model import SGDClassifier
def apply_SGD(train_csr, test_csr):
  sgd = SGDClassifier()
  sgd.fit(train_csr, train_labels)
  pred_results = sgd.predict(test_csr)
  print('-----------------------stochaistic Gradient Descent-----------------------')
  print(pred_results)
  print('----------------------------------------------')
  write_to_file(pred_results,'results_SGD.dat')

from sklearn.linear_model import Perceptron
def apply_Perceptron(train_csr, test_csr):
  perceptron = Perceptron()
  perceptron.fit(train_csr, train_labels)
  pred_results = perceptron.predict(test_csr)
  print('-----------------------Perceptron-----------------------')
  print(pred_results)
  print('----------------------------------------------')
  write_to_file(pred_results,'results_Perceptron.dat')

from sklearn.linear_model import LogisticRegression
def apply_LogReg(train_csr, test_csr):
  logReg = LogisticRegression()
  logReg.fit(train_csr, train_labels)
  pred_results = logReg.predict(test_csr)
  print('-----------------------Logistical Regression -----------------------')
  print(pred_results)
  print('----------------------------------------------')
  write_to_file(pred_results,'results_logReg.dat')

from sklearn.svm import LinearSVC
def apply_LinearSVC(train_csr, test_csr):
  linSVC = LinearSVC()
  linSVC.fit(train_csr, train_labels)
  pred_results = linSVC.predict(test_csr)
  print('-----------------------Linear SVC-----------------------')
  print(pred_results)
  print('----------------------------------------------')
  write_to_file(pred_results,'results_linearSVC.dat')

def main():
  train_csr = create_Train_csr()
  test_csr = create_Test_csr()
  print(train_labels)
  # print(train_csr.toarray())
  # print(test_csr.toarray())
  # apply_bernoulliNB(train_csr,test_csr)
  # apply_randomforest(train_csr,test_csr)
  # apply_DecisionTree(train_csr,test_csr)
  apply_SGD(train_csr,test_csr)
  apply_Perceptron(train_csr,test_csr)
  # apply_LogReg(train_csr,test_csr)
  # apply_LinearSVC(train_csr,test_csr)

if __name__ == '__main__':
  main()
