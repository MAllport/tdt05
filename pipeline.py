import pandas as pd
import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt

import string
from tempfile import TemporaryDirectory

# from functools import partial

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, average_precision_score, precision_score, \
    recall_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split, cross_val_score


def converter(x,feature):

    if x == "" or pd.isna(x): return np.nan
    elif feature == 'f0' : return int(x,16)
    elif feature == 'f1' : return x
    elif feature == 'f2' : return x
    elif feature == 'f3' : return x # categorical
    elif feature == 'f4' : return x # ordinal categories 1-6
    elif feature == 'f5' : return x
    elif feature == 'f6' : return x # ordinal 0.0 - 0.5
    elif feature == 'f7' : return int(x,16)
    elif feature == 'f8' : return x # ordinal 1 - 12
    elif feature == 'f9' : return x # categorical
    elif feature == 'f10': return x
    elif feature == 'f11': return x
    elif feature == 'f12': return x # ordinal 1-7
    elif feature == 'f13': return x
    elif feature == 'f14': return x # categorical
    elif feature == 'f15': return int(x,16)
    elif feature == 'f16': return x # categorical
    elif feature == 'f17': return x # ordinal 0-3
    elif feature == 'f18': return x
    elif feature == 'f19': return x # ordinal 1-3
    elif feature == 'f20': return x # ordinal 0-4
    elif feature == 'f21': return x # categorical
    elif feature == 'f22': return x
    elif feature == 'f23': return int(x,16)
    elif feature == 'f24' and all(c in string.hexdigits for c in x): return int(x,16) # hex + ignore


def evaluate_model(targets, predicted, scores):
    prc = precision_score(targets, predicted)
    rc  = recall_score(targets, predicted)
    auc = roc_auc_score(targets, predicted)
    aps = average_precision_score(targets, predicted)
    acs = accuracy_score(targets, predicted)

    print(f"{prc=}")
    print(f"{rc=}")
    print(f"{auc=}")
    print(f"{aps=}")
    print(f"{acs=}")
    print(f"{scores=}")


    plot_ROC(targets, predicted, auc)

def plot_ROC(targets, predicted, auc):
    fpr, tpr, thresholds = roc_curve(targets, predicted)
    plt.plot(fpr, tpr) 
    plt.axis("Square")
    plt.xlabel('False Positive Rate') 
    plt.ylabel('True Positive Rate') 
    plt.title(f'ROC curve: score={auc}')
    plt.show()



def main():

    with open('datasets/challenge1_train.csv') as train_csv:
        df = pd.read_csv(train_csv)

    labels = df['target']
    features = df.drop(columns=['target', 'id'])

    # TODO: preprocess test features when we want to predict on test set
    for i in range(25):
        feature = 'f' + str(i)
        features[feature] = features[feature].apply(lambda x: converter(x,feature))

    # print(features.describe())
    # print(features.dtypes)
    # print(features['f0'])

    numeric_features     = ['f2', 'f18']
    hexadecimal_features = ['f0', 'f7', 'f15', 'f23', 'f24']
    boolean_features     = ['f1', 'f10', 'f11', 'f13', 'f22']
    ordinal_features     = ['f4', 'f6', 'f8', 'f12', 'f17', 'f19', 'f20']
    categorical_features = ['f3', 'f9', 'f14', 'f16', 'f21']

    # TODO: iterative imputer, gridsearch, standard vs minmax scaler

    numeric_transformer = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler()
    )

    boolean_transformer = make_pipeline(
        SimpleImputer(strategy='mean'),
        StandardScaler()
    )

    ordinal_transformer = make_pipeline(
        SimpleImputer(strategy='mean'),
        StandardScaler()
    )

    categorical_transformer = make_pipeline(
        SimpleImputer(strategy='constant', fill_value='missing'),
        OneHotEncoder(handle_unknown='ignore')
    )

    preprocessor = make_column_transformer(
        (numeric_transformer, numeric_features + hexadecimal_features),
        # (hexadecimal_transformer, hexadecimal_features),
        (boolean_transformer, boolean_features),
        (ordinal_transformer, ordinal_features),
        (categorical_transformer, categorical_features),
    )

    with TemporaryDirectory() as cachedir:
        classifier = make_pipeline(preprocessor, LogisticRegression(max_iter = 1000000), memory=cachedir)

        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

        classifier.fit(X_train, y_train)

        print("model score: %.3f" % classifier.score(X_test, y_test))

        predicted = classifier.predict(X_test)
        targets = y_test

        scores = cross_val_score(classifier, X_test, y_test, cv=5, scoring='f1_macro')
        
        evaluate_model(targets, predicted, scores)

if __name__ == "__main__":
    main()

