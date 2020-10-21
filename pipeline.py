import pandas as pd
import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
import string
import seaborn as sns
import imblearn
import scipy

from tempfile import TemporaryDirectory

# from functools import partial

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, average_precision_score, precision_score, recall_score, balanced_accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.base import TransformerMixin, BaseEstimator
from itertools import combinations
from sklearn.svm import SVC

from scipy.stats import chi2_contingency,spearmanr

numeric_features     = ['f2', 'f18']
hexadecimal_features = ['f0', 'f7', 'f15', 'f23', 'f24']
boolean_features     = ['f1', 'f10', 'f11', 'f13', 'f22']
ordinal_features     = ['f4', 'f6', 'f8', 'f12', 'f17', 'f19', 'f20']
categorical_features = ['f3', 'f9', 'f14', 'f16', 'f21']


def evaluate_model(targets, predicted):

    metrics = {
        'accuracy': accuracy_score,
        'balanced accuracy': balanced_accuracy_score,
        'recall' : recall_score,
        'precision': precision_score,
        'average precision': average_precision_score,
        'AUC': roc_auc_score,
        'F1': f1_score,
    }

    for metric, fun in metrics.items():
        print(f"{metric} = {fun(targets, predicted):.2f}")

    print(classification_report(targets, predicted))

    # plot_ROC(targets, predicted, average_precision_score(targets, predicted))


def plot_ROC(targets, predicted, auc):

    fpr, tpr, thresholds = roc_curve(targets, predicted)
    plt.plot(fpr, tpr) 
    plt.axis("Square")
    plt.xlabel('False Positive Rate') 
    plt.ylabel('True Positive Rate') 
    plt.title(f'ROC curve: score={auc}')
    plt.show()


def fit_model(pipeline, X_train, X_test, y_train, y_test):

    pipeline.fit(X_train, y_train)
    print("model score: %.3f" % pipeline.score(X_test, y_test))

    return pipeline

    
def fit_gridsearch(pipeline, X_train, y_train):


    kf = KFold(n_splits=5, shuffle=False)

    #params = {
    #    'n_estimators'      : [200,700],
    #    'max_depth'         : [8, 9, 10, 11, 12],
    #    'max_features'      : ['sqrt', 'log2']
    #    #'criterion' :['gini']
    #}

    #logreg_params = {'randomforestclassifier__' + key: params[key] for key in params}

    #grid_imba = GridSearchCV(
    #    pipeline,
    #    param_grid=logreg_params,
    #    cv=kf,
    #    scoring='balanced_accuracy',
    #    return_train_score=True,
    #    n_jobs=7
    #)

    params = [
        {
            'kernel': ['rbf'],
            'gamma': [1e-3, 1e-4],
            'C': [1, 10, 100, 1000]
        },
        {
            'kernel': ['linear'],
            'C': [1, 10, 100, 1000]
        }
    ]

    params = [{'svc__' + key: d[key] for key in d} for d in params]

    grid_imba = GridSearchCV(
        pipeline,
        param_grid=params,
        cv=kf,
        scoring='f1',
        return_train_score=True,
        n_jobs=7
    )

    grid_imba.fit(X_train, y_train)

    print(grid_imba.best_params_)
    print(f"Best score ({grid_imba.scoring}): ", end='')
    print(grid_imba.best_score_)

    return grid_imba

def conv_hex(x):
    try:
        return int(x, 16)
    except ValueError:
        return np.nan
    except TypeError:
        return np.nan

def spearman(df):
    print("Spearmans \n")
    comb = combinations(list(df), 2)
    for col1, col2 in comb:
        coef, p = spearmanr(df[col1], df[col2], nan_policy="omit")
        # interpret the significance
        alpha = 0.05
        if p < alpha:
            print("Comparing " + col1 + " and " + col2)
            print('Spearmans correlation coefficient: %.3f' % coef)
            print('Samples are correlated (reject H0) p=%.3f' % p)

def chi2(df):
    print("Chi2 \n")
    comb = combinations(list(df), 2)
    for col1, col2 in comb:
        p = chi2_contingency(pd.crosstab(df[col1], df[col2]))[1]
        # interpret the significance
        alpha = 0.05
        if p < alpha:
            print("Comparing " + col1 + " and " + col2)
            print('Samples are correlated (reject H0) p=%.3f' % p)

def analyze_corr(labels,features):

    le = LabelEncoder()
    for i in range(25):
        feature = "f" + str(i)
        if feature in hexadecimal_features:
            features[feature] = features[feature].apply(lambda x: conv_hex(x))
        if feature in categorical_features:
            #features[feature] = pd.get_dummies(features[feature])
            features[feature] = le.fit_transform(features[feature].astype(str))    

    min_max_scaler = MinMaxScaler()
    features = pd.DataFrame(min_max_scaler.fit_transform(features), columns=features.columns, index=features.index)
    df_num = features[features.columns & (numeric_features + ordinal_features)]
    df_cat = features[features.columns & (categorical_features + ordinal_features)]

    spearman(features)
    chi2(df_cat)

    sns.pairplot(df_num, kind="scatter")
    plt.savefig("numpairs.png")

    sns.pairplot(df_cat, kind="scatter")
    plt.savefig("catpairs.png")

    correlations = features.corr(method="spearman")

    fig, ax = plt.subplots(figsize=(15,15))
    heatmap = sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',
            square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})
    plt.savefig('heatmap.png')

    print("Correlation between numeric columns : ")
    print(df_num.corr())

def main():

    with open('datasets/challenge1_train.csv') as train_csv:
        df = pd.read_csv(train_csv, skipinitialspace=True)

    labels = df['target']
    features = df.drop(columns=['target', 'id'])
    # print(df.duplicated().any())

    missing_values = df.isna().sum()
    fig, ax = plt.subplots(1,2, figsize=(15, 12))
    sns.barplot(missing_values.values, missing_values.index, ax=ax[0])
    ax[0].set_title("Barplot of missing values for each feature")
    ax[0].set_xlabel('# of missing values')
    ax[0].set_ylabel('Feature')

    sns.distplot(missing_values.values, ax=ax[1], kde=False)
    ax[1].set_title("Distribution of missing values for each feature")
    ax[1].set_xlabel('# of missing values')
    ax[1].set_ylabel('Bin count')

    analyze_corr(labels, features.copy())


    for i in range(25):
        feature = "f" + str(i)
        if feature in hexadecimal_features:
            features[feature] = features[feature].apply(lambda x: conv_hex(x))
    
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

        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, stratify=labels)

        imba_pipeline = make_pipeline(
            preprocessor,
            SMOTE(),
<<<<<<< HEAD
            RandomForestClassifier(n_estimators=500, random_state=0),
=======
            # RandomOverSampler(),
            # RandomForestClassifier(random_state=0),
            # SVC(kernel="linear", C=0.025),
            SVC(),
>>>>>>> 0268053116558cbb18a481f6e55c9cab36ef657a
            memory=cachedir
        )

        # preprocessor.fit_transform(X_train, y_train)

<<<<<<< HEAD
        f = X_train.columns
        importances = model.named_steps["randomforestclassifier"].feature_importances_
        indices = np.argsort(importances)

        plt.figure(figsize=(10,15))
        plt.title('Feature Importances')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [f[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.show()

=======
        model = fit_model(imba_pipeline, X_train, X_test, y_train, y_test)
        # model = fit_gridsearch(imba_pipeline, X_train, y_train)
>>>>>>> 0268053116558cbb18a481f6e55c9cab36ef657a

        # predicted = model.predict(X_test)
        # evaluate_model(y_test, predicted)
        # pd.DataFrame(predicted).to_csv("predictions.csv")


if __name__ == "__main__":
    main()

