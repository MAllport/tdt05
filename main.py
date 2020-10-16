import pandas as pd
import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
import string

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from functools import partial
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve, ShuffleSplit, train_test_split
from joblib import Parallel, delayed

from plots import plot_learning_curve

counter = 1

def read_dataset():

    train_csv = open('datasets/challenge1_train.csv',"r")
    test_csv = open('datasets/challenge1_test.csv',"r")

    _, y, X = np.split(pd.read_csv(train_csv), [1,2], axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    return X_train, X_test, y_train, y_test

def converter(x,feature):
    if x == "" or pd.isna(x): return np.nan
    elif feature == 'f0' and all(c in string.hexdigits for c in x): return int(x,16)
    elif feature == 'f1' : return x
    elif feature == 'f2' : return x
    elif feature == 'f3' : return ord(x) - 96
    elif feature == 'f4' : return x
    elif feature == 'f5' : return x
    elif feature == 'f6' : return x
    elif feature == 'f7' and all(c in string.hexdigits for c in x): return int(x,16)
    elif feature == 'f8' : return x
    elif feature == 'f9' : return x
    elif feature == 'f10': return x
    elif feature == 'f11': return x
    elif feature == 'f12': return x
    elif feature == 'f13': return x
    elif feature == 'f14': return ord(x) - 96
    elif feature == 'f15' and all(c in string.hexdigits for c in x): return int(x,16)
    elif feature == 'f16': return ord(x.lower()) - 96
    elif feature == 'f17': return x
    elif feature == 'f18': return x
    elif feature == 'f19': return x
    elif feature == 'f20': return x
    elif feature == 'f21': return ord(x.lower()) - 96
    elif feature == 'f22': return x
    elif feature == 'f23' and all(c in string.hexdigits for c in x): return int(x,16)
    elif feature == 'f24' and all(c in string.hexdigits for c in x): return int(x,16)

def converter_categorical(x,feature):

    if x == "" or pd.isna(x): return np.nan
    elif feature == 'f0' and all(c in string.hexdigits for c in x): return int(x,16)
    elif feature == 'f1' : return x
    elif feature == 'f2' : return x
    elif feature == 'f3' : return x
    elif feature == 'f4' : return x
    elif feature == 'f5' : return x
    elif feature == 'f6' : return x
    elif feature == 'f7' and all(c in string.hexdigits for c in x): return int(x,16)
    elif feature == 'f8' : return x
    elif feature == 'f9' : return x
    elif feature == 'f10': return x
    elif feature == 'f11': return x
    elif feature == 'f12': return x
    elif feature == 'f13': return x
    elif feature == 'f14': return x
    elif feature == 'f15' and all(c in string.hexdigits for c in x): return int(x,16)
    elif feature == 'f16': return x
    elif feature == 'f17': return x
    elif feature == 'f18': return x
    elif feature == 'f19': return x
    elif feature == 'f20': return x
    elif feature == 'f21': return x
    elif feature == 'f22': return x
    elif feature == 'f23' and all(c in string.hexdigits for c in x): return int(x,16)
    elif feature == 'f24' and all(c in string.hexdigits for c in x): return int(x,16)

def normalize_data(train,test):
    for i in range(25):
        feature = 'f' + str(i)
        train[feature] = train[feature].apply(lambda x: converter(x,feature),1)
        test[feature]  = test[feature].apply(lambda x: converter(x,feature),1)

    # Use LabelEncoder to encode feature "f9"
    #label_encoder = preprocessing.LabelEncoder()

    #train['f9'] = label_encoder.fit_transform(train['f9'].fillna('0'))
    #test['f9'] = label_encoder.fit_transform(test['f9'].fillna('0'))

    one_hot_train = pd.get_dummies(train['f9'], prefix='f')
    one_hot_test = pd.get_dummies(test['f9'], prefix='f')

    train = train.join(one_hot_train).drop(['f9'],axis=1)
    test  = test.join(one_hot_test).drop(['f9'],axis=1)

    min_max_scaler = preprocessing.MinMaxScaler()
    train = pd.DataFrame(min_max_scaler.fit_transform(train), columns=train.columns, index=train.index)
    test  = pd.DataFrame(min_max_scaler.fit_transform(test), columns=test.columns, index=test.index)

    imp_mean_train = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_mean_train.fit(train)
    train = pd.DataFrame(imp_mean_train.transform(train))

    imp_mean_test = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_mean_test.fit(test)
    test = pd.DataFrame(imp_mean_test.transform(test))

    return train, test

def normalize_data_it(X,y):
    print(y.head())
    print(y['target'].astype('int').dtype)
    for i in range(25):
        feature = 'f' + str(i)
        X[feature] = X[feature].apply(lambda x: converter_categorical(x,feature),1)



    X = X.join(pd.get_dummies(X['f3'], prefix='f3')).drop(['f3'],axis=1)
    X = X.join(pd.get_dummies(X['f4'], prefix='f4')).drop(['f4'],axis=1)
    X = X.join(pd.get_dummies(X['f5'], prefix='f5')).drop(['f5'],axis=1)
    X = X.join(pd.get_dummies(X['f6'], prefix='f6')).drop(['f6'],axis=1)
    X = X.join(pd.get_dummies(X['f8'], prefix='f8')).drop(['f8'],axis=1)
    X = X.join(pd.get_dummies(X['f9'], prefix='f9')).drop(['f9'],axis=1)
    X = X.join(pd.get_dummies(X['f13'], prefix='f13')).drop(['f13'],axis=1)
    X = X.join(pd.get_dummies(X['f14'], prefix='f14')).drop(['f14'],axis=1)
    X = X.join(pd.get_dummies(X['f16'], prefix='f16')).drop(['f16'],axis=1)
    X = X.join(pd.get_dummies(X['f17'], prefix='f17')).drop(['f17'],axis=1)
    X = X.join(pd.get_dummies(X['f19'], prefix='f19')).drop(['f19'],axis=1)
    X = X.join(pd.get_dummies(X['f20'], prefix='f20')).drop(['f20'],axis=1)
    X = X.join(pd.get_dummies(X['f21'], prefix='f21')).drop(['f21'],axis=1)

    min_max_scaler = preprocessing.MinMaxScaler()
    X = pd.DataFrame(min_max_scaler.fit_transform(X), columns=X.columns, index=X.index)

    imp_mean_train = IterativeImputer(missing_values=np.nan, estimator = BayesianRidge())
    imp_mean_train.fit(X)   
    X = pd.DataFrame(imp_mean_train.transform(X))

    return X

def normalize_data_categorical(X):
    for i in range(25):
        feature = 'f' + str(i)
        X[feature] = X[feature].apply(lambda x: converter_categorical(x,feature),1)



    X = X.join(pd.get_dummies(X['f3'], prefix='f3')).drop(['f3'],axis=1)
    X = X.join(pd.get_dummies(X['f4'], prefix='f4')).drop(['f4'],axis=1)
    X = X.join(pd.get_dummies(X['f5'], prefix='f5')).drop(['f5'],axis=1)
    X = X.join(pd.get_dummies(X['f6'], prefix='f6')).drop(['f6'],axis=1)
    X = X.join(pd.get_dummies(X['f8'], prefix='f8')).drop(['f8'],axis=1)
    X = X.join(pd.get_dummies(X['f9'], prefix='f9')).drop(['f9'],axis=1)
    X = X.join(pd.get_dummies(X['f12'], prefix='f12')).drop(['f12'],axis=1)
    X = X.join(pd.get_dummies(X['f13'], prefix='f13')).drop(['f13'],axis=1)
    X = X.join(pd.get_dummies(X['f14'], prefix='f14')).drop(['f14'],axis=1)
    X = X.join(pd.get_dummies(X['f16'], prefix='f16')).drop(['f16'],axis=1)
    X = X.join(pd.get_dummies(X['f17'], prefix='f17')).drop(['f17'],axis=1)
    X = X.join(pd.get_dummies(X['f19'], prefix='f19')).drop(['f19'],axis=1)
    X = X.join(pd.get_dummies(X['f20'], prefix='f20')).drop(['f20'],axis=1)
    X = X.join(pd.get_dummies(X['f21'], prefix='f21')).drop(['f21'],axis=1)

    min_max_scaler = preprocessing.MinMaxScaler()
    X = pd.DataFrame(min_max_scaler.fit_transform(X), columns=X.columns, index=X.index)

    imp_mean_train = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_mean_train.fit(X)
    X = pd.DataFrame(imp_mean_train.transform(X))

    return X

def naive_bayes(y_train, X_train, y_test, X_test):
    model = GaussianNB
    model.fit(X_train, y_train['target'].astype('int'))
    predicted = model.predict(X_test)

def logistic_plot(X_train, X_test, y_train, y_test):
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))

    title = "Learning Curves (Logistic)"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    estimator = LogisticRegression(max_iter = 100000)
    plot_learning_curve(estimator, title, X_train, y_train['target'].astype('int'), axes=axes[:, 0], ylim=(0.7, 1.01),
                        cv=cv, n_jobs=4)


X_train, X_test, y_train, y_test = read_dataset()

X_train_categorical = normalize_data_categorical(X_train.copy())
X_test_categorical = normalize_data_categorical(X_test.copy())

X_train_it = normalize_data_it(X_train.copy(), y_train)
X_test_it = normalize_data_it(X_test.copy(), y_train)

X_train, X_test = normalize_data(X_train, X_test)

#y_train.join(X_train_categorical).to_csv('datasets/transformed_train_new.csv', index = False)
#y_test.join(X_test_categorical).to_csv('datasets/transformed_test_new.csv', index = False)

#y_train.join(X_train).to_csv('datasets/transformed_train.csv', index = False)
#y_test.join(X_test).to_csv('datasets/transformed_test.csv', index = False)


#_, y, X = np.split(pd.read_csv(train_csv), [1,2], axis = 1)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print("Training with X_train")
#logistic_plot(X_train, X_test, y_train, y_test)
print("Training with X_train_categorical")
#logistic_plot(X_train_categorical, X_test_categorical, y_train, y_test)
print("Training with X_train_iterative")
logistic_plot(X_train_it, X_test_it, y_train, y_test)

plt.show()