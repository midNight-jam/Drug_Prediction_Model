# Drug_Prediction_Model

Develop predictive models that can determine, given a particular compound, whether it is active (1) or not (0). A molecule can be represented by several thousands of binary features which represent their topological shapes and other characteristics important for binding.

# Data Description
The training dataset consists of 800 records and the test dataset consists of 350 records.
We provide you with the training class labels and the test labels are held out. The attributes
are binary and are presented in a sparse matrix format within train.dat and test.dat. Note
that, unlike the CSR matrices we saw before, the values are not listed in the file, since they
are always 1.
# train.dat
Training set (a sparse binary matrix, patterns in lines, features in columns, with
class label 1 or 0 in the first column).
# test.dat
Testing set (a sparse binary matrix, patterns in lines, features in columns, no
class label provided).
# format.dat
A sample submission with 350 entries randomly chosen to be 0 or 1.
