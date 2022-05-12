"""
This script runs age prediction with linear regression using labeled nii.gz brain MRI images.
First, the volume (number of voxels) feature for each brain segmentation label is obtained. The 20 volume features
that have highest correlation with age are selected. These features are used to train a multivariate linear regression
model to predict age.
"""

import glob
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import os
import csv
import pickle

# extract age data from target images
age_data = np.genfromtxt('/Users/stephenpark/Desktop/mia_project2/train_age.csv', delimiter=',')
age_data = np.delete(age_data, (0), axis=0)

# list of label images
label_list = glob.glob('/Users/stephenpark/Desktop/mia_project2/label_output/*.nii.gz')

# get label map - list of labels that correspond to brain region
filename = '/Users/stephenpark/Desktop/mia_project2/label.txt'
with open(filename, "r") as f:
    lines = f.readlines()
    labelmap = []
    for line in lines:
        temp = line.split()
        labelmap.append(temp[0])
label_map = [int(x) for x in labelmap]

### regression model

# build input matrix
X = np.zeros((252,133)) # (images, features)

label_dict = {}
for i in range(len(label_list)):
    label_im = nib.load(label_list[i])
    label_data0 = label_im.get_fdata()
    label_data = label_data.astype(np.uint8)  # set labels as integers
    (label, count) = np.unique(label_data0, return_counts=True)
    for j in range(len(label)):
        if label[j] in label_map:
            label_dict[label[j]] = count[j]

    X[i, :] = list(label_dict.values())

# generate regression dataset
label_nm = [os.path.basename(x) for x in glob.glob('/Users/stephenpark/Desktop/mia_project2/label_output/*.nii.gz')]
age_list = age_data[:,1] # age list
y = []
for i in range(len(label_nm)):
    temp = label_nm[i].split('_')
    id = int(temp[1])
    y_tmp = age_list[id-1]
    y.append(y_tmp)

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)


### feature selection based on correlation
# select 20 features - was chosen because it had lowest MAE
fs = SelectKBest(score_func=f_regression, k=20)
fs.get_support().astype(int)
# learn relationship from training data
fs.fit(X_train, y_train)
# transform train input data
X_train_fs = fs.transform(X_train)
# transform test input data
X_test_fs = fs.transform(X_test)

# fit the model
model = LinearRegression()
model.fit(X_train_fs, y_train)
# make predictions of y_test_fs using X_test_fs
yhat = model.predict(X_test_fs)
# evaluate predictions based on mean absolute error
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)

# save the model
filename = 'trained_model.sav'
pickle.dump(model, open(filename, 'wb'))



### final test
# list of label images
label_list_test = glob.glob('/Users/stephenpark/Desktop/mia_project2/registration_output/*.nii.gz')
X_test_eval = np.zeros((15,133))
label_dict = {}
for i in range(len(label_list_test)):
    label_im = nib.load(label_list_test[i])
    label_data0 = label_im.get_fdata()
    label_data = label_data.astype(np.uint8)  # set labels as integers
    (label, count) = np.unique(label_data0, return_counts=True)
    for j in range(len(label)):
        if label[j] in label_map:
            label_dict[label[j]] = count[j]

    X_test_eval[i, :] = list(label_dict.values())

# select the 20 features
X_test_eval_fs = fs.transform(X_test_eval)

# load trained model
filename = 'trained_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.predict(X_test_eval_fs)

# create csv file
img_name = [os.path.basename(x) for x in glob.glob('/Users/stephenpark/Desktop/mia_project2/registration_output/*.nii.gz')]

with open('age_prediction.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(img_name)
    writer.writerow(result)