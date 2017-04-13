
from scipy.sparse import coo_matrix
import numpy as np

train_labels = [] # to hold the train data set provided labels

def create_Train_csr():
  file = './data/train.dat'
  row_id = 0  # row index variable
  rows = [] # place holder array for holding the row index
  cols= [] # place holder array for holding the column index
  data = [] # place holder array for holding the data on the ith row & ith column position
  with open(file) as f1:
    for lines in f1:
      lines = lines.replace('\n',' ').replace('\t',' ')
      lines = ' '.join(lines.split())
      document = lines.split(' ')
      train_labels.append(document[0]) # reading and storing the labels from the training data set
      for i in range(1,len(document)):
        rows.append(row_id)
        cols.append(document[i])
        data.append(1)  # as we have binary data thus we are using 1
      row_id += 1 # increasing the row number for next row
  rows = np.array(rows) # converting to numpy array for passing to coo_matrix
  cols = np.array(cols)
  data = np.array(data)
  # creating a coo_matrix representation using the rows, cols & data numpy array and then converting it to
  # a csr_matrix to return
  # we have used 800 * 100001 as our shape for csr matrix
  # Why 800 ? beacuse these are the numbers of documents in the train data
  # Why 100001 ? because we have 100000 features in our training set and one extra I added because I was getting a
  # column exceeded index error
  return coo_matrix((data,(rows,cols)), shape=(800,100001)).tocsr()

def create_Test_csr():
  file = './data/test.dat'
  row_id = 0
  rows = []
  cols = []
  data = []
  with open(file) as f1:
    for lines in f1:
      lines = lines.replace('\n', ' ').replace('\t', ' ') # cleaning of line for newline & tab character
      lines = ' '.join(lines.split())
      document = lines.split(' ') # taking out the features
      for i in range(0, len(document)):
        rows.append(row_id)
        cols.append(document[i])
        data.append(1)
      row_id += 1
  rows = np.array(rows)
  cols = np.array(cols)
  data = np.array(data)
  # we have used 350 * 100001 as our shape for csr matrix
  # Why 350 ? beacuse these are the numbers of documents in the test data
  # Why 100001 ? because we have 100000 features in our training set and one extra I added because I was getting a
  # column exceeded index error
  return coo_matrix((data, (rows, cols)), shape=(350, 100001)).tocsr()

#  a utility method to write the classification results to a file
def write_to_file(pred_results, file_name):
  f = open(file_name,'w')
  for p in pred_results:
    f.write(p+'\n')
  f.close()

# using RandomForestClassifier for classification
from sklearn.ensemble import RandomForestClassifier
def apply_randomforest(train_csr, test_csr):
  random_forest = RandomForestClassifier()
  random_forest.fit(train_csr, train_labels)
  pred_results = random_forest.predict(test_csr)
  print('-----------------------RandomForest-----------------------')
  print(pred_results)
  print('----------------------------------------------')
  write_to_file(pred_results,'results_randomForest.dat')

# using DecisionTreeClassifier for classification
from sklearn.tree import DecisionTreeClassifier
def apply_DecisionTree(train_csr, test_csr):
  decision_tree = DecisionTreeClassifier()
  decision_tree.fit(train_csr, train_labels)
  pred_results = decision_tree.predict(test_csr)
  print('-----------------------decision tree-----------------------')
  print(pred_results)
  print('----------------------------------------------')
  write_to_file(pred_results,'results_decisionTree.dat')

# using SGDClassifier for classification (not tuned)
from sklearn.linear_model import SGDClassifier
def apply_SGD(train_csr, test_csr):
  sgd = SGDClassifier()
  sgd.fit(train_csr, train_labels)
  pred_results = sgd.predict(test_csr)
  print('-----------------------stochaistic Gradient Descent-----------------------')
  print(pred_results)
  print('----------------------------------------------')
  write_to_file(pred_results, 'results_SGD.dat')

# using SGDClassifier for classification but tuned with parameters
def apply_SGD_parmasControlled(train_csr, test_csr):
  sgd = SGDClassifier(loss='hinge', penalty='l2')
  sgd.fit(train_csr, train_labels)
  # we have tried to tune SGD with passing the parameters.
  # The parameters to pay attention is "eta0", it is noted that SGD preforms for higher value of 0.6 if the number of
  # features are high, as in our case 100000. This has resulted in the getting high F1 score as compared to original
  # plain SGD
  SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1, eta0=0.6,
                fit_intercept=True, l1_ratio=0.15, learning_rate='optimal', loss='hinge',
                n_iter=5, n_jobs=1, penalty='l2', power_t=0.5, random_state=None,
                shuffle=True, verbose=0, warm_start=False)
  pred_results = sgd.predict(test_csr)
  print('-----------------------stochaistic Gradient Descent Controlled-----------------------')
  print(pred_results)
  print('----------------------------------------------')
  write_to_file(pred_results,'results_SGD_params_0.6.dat')

from sklearn.linear_model import Perceptron
def apply_Perceptron(train_csr, test_csr):
  perceptron = Perceptron()
  perceptron.fit(train_csr, train_labels)
  pred_results = perceptron.predict(test_csr)
  print('-----------------------Perceptron-----------------------')
  print(pred_results)
  print('----------------------------------------------')
  write_to_file(pred_results,'results_Perceptron.dat')

def main():
  train_csr = create_Train_csr() # converting train data to csr matrix
  test_csr = create_Test_csr()  # converting test data to csr matrix
  apply_randomforest(train_csr,test_csr)
  apply_DecisionTree(train_csr,test_csr)
  apply_SGD(train_csr, test_csr)
  apply_Perceptron(train_csr, test_csr)
  apply_SGD_parmasControlled(train_csr, test_csr)

if __name__ == '__main__':
  main()