import pandas as pd
import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
import string
from sklearn.impute import SimpleImputer

from functools import partial

from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

counter = 1

def read_dataset():

    train_csv = open('datasets/challenge1_train.csv',"r")
    test_csv = open('datasets/challenge1_test.csv',"r")

    train = pd.read_csv(train_csv)
    test  =  pd.read_csv(test_csv)

    return train, test

def converter(x,feature):

    if x == "" or pd.isna(x): return np.nan
    elif feature == 'f0' : return int(x,16)
    elif feature == 'f1' : return x
    elif feature == 'f2' : return x
    elif feature == 'f3' : return ord(x) - 96
    elif feature == 'f4' : return x
    elif feature == 'f5' : return x
    elif feature == 'f6' : return x
    elif feature == 'f7' : return int(x,16)
    elif feature == 'f8' : return x
    elif feature == 'f9' : return x
    elif feature == 'f10': return x
    elif feature == 'f11': return x
    elif feature == 'f12': return x
    elif feature == 'f13': return x
    elif feature == 'f14': return ord(x) - 96
    elif feature == 'f15': return int(x,16)
    elif feature == 'f16': return ord(x.lower()) - 96
    elif feature == 'f17': return x
    elif feature == 'f18': return x
    elif feature == 'f19': return x
    elif feature == 'f20': return x
    elif feature == 'f21': return ord(x.lower()) - 96
    elif feature == 'f22': return x
    elif feature == 'f23': return int(x,16)
    elif feature == 'f24' and all(c in string.hexdigits for c in x): return int(x,16)

def normalize_data(train,test):
    train_label, train_feature = np.split(train, [2], axis = 1)
    test_label, test_feature =  np.split(test, [1], axis = 1)

    for i in range(25):
        feature = 'f' + str(i)
        train_feature[feature] = train_feature[feature].apply(lambda x: converter(x,feature),1)
        test_feature[feature]  = test_feature[feature].apply(lambda x: converter(x,feature),1)

    # Use LabelEncoder to encode feature "f9"
    #label_encoder = preprocessing.LabelEncoder()

    #train_feature['f9'] = label_encoder.fit_transform(train_feature['f9'].fillna('0'))
    #test_feature['f9'] = label_encoder.fit_transform(test_feature['f9'].fillna('0'))

    one_hot_train = pd.get_dummies(train_feature['f9'], prefix='f')
    one_hot_test = pd.get_dummies(test_feature['f9'], prefix='f')

    train_feature = train_feature.join(one_hot_train).drop(['f9'],axis=1)
    test_feature  = test_feature.join(one_hot_test).drop(['f9'],axis=1)

    min_max_scaler = preprocessing.MinMaxScaler()
    train_feature = pd.DataFrame(min_max_scaler.fit_transform(train_feature), columns=train_feature.columns, index=train_feature.index)
    test_feature  = pd.DataFrame(min_max_scaler.fit_transform(test_feature), columns=test_feature.columns, index=test_feature.index)

    imp_mean_train = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_mean_train.fit(train_feature)
    train_feature = pd.DataFrame(imp_mean_train.transform(train_feature))

    imp_mean_test = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_mean_test.fit(test_feature)
    test_feature = pd.DataFrame(imp_mean_test.transform(test_feature))

    print(train_feature.head())


    train_label.join(train_feature).to_csv('datasets/transformed_train.csv', index = False)
    test_label.join(test_feature).to_csv('datasets/transformed_test.csv', index = False)

    return train_label, train_feature, test_label, test_feature

def naive_bayes(train_label, train_feature, test_label, test_feature):
    model = LogisticRegression()
    model.fit(train_feature, train_label['target'].astype('int'))
    predicted = model.predict(test_feature)

    print(model.score(train_feature, train_label['target'].astype('int')))

    print(pd.DataFrame(predicted).head(50))
    print(train_label['target'].head(50))





#train, test = read_dataset()
#train_label, train_feature, test_label, test_feature = normalize_data(train,test)
train_csv = open('datasets/transformed_train.csv',"r")
test_csv = open('datasets/transformed_test.csv',"r")

train = pd.read_csv(train_csv)
test  =  pd.read_csv(test_csv)
train_label, train_feature = np.split(train, [2], axis = 1)
test_label, test_feature =  np.split(test, [1], axis = 1)

naive_bayes(train_label,train_feature, test_label, test_feature)
