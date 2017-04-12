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


def main():
  readTrainingData()
  print(len(features))

  readTestData()
  print(test_data[1])


if __name__ == '__main__':
  main()
