import pandas as pd
from bayes_opt import BayesianOptimization
from bayes_opt.util import Colours
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from preprocessing_pipeline import preprocessing_pipeline


def get_cv():
    return TimeSeriesSplit(n_splits=5)


def get_pipeline(model):
    return Pipeline([('Vectorizer',
                      CountVectorizer(tokenizer=preprocessing_pipeline,
                                      max_df=0.9,
                                      min_df=0.07)), ('Classifier', model)])


def load_data():
    news = pd.read_csv('./data/bigboi.csv', parse_dates=["Date"], index_col=0)
    labeled_spy = pd.read_csv('./data/labeled_spy_data.csv',
                              parse_dates=["Date"],
                              index_col=0)
    news = news.set_index('Date')
    labeled_spy = labeled_spy.set_index('Date')
    #remove non-business days from news dataframe
    news = news[news.index.dayofweek < 5]

    #process data for matching dates
    #labels = list(labeled_spy.loc[labeled_spy.index.intersection(news.index)]['Drop'])
    labels = labeled_spy.loc[news.index]['Drop']
    actual_fucking_values = labels.notnull()
    labels = list(labels.loc[actual_fucking_values])
    news = news.loc[actual_fucking_values]
    return news.Text, labels


def svc_cv(C, gamma, data, labels):
    """SVC cross validation.

    This function will instantiate a SVC classifier with parameters C and
    gamma. Combined with data and labels this will in turn be used to perform
    cross validation. The result of cross validation is returned.

    Our goal is to find combinations of C and gamma that maximizes the roc_auc
    metric.
    """
    estimator = svm.SVC(C=C, gamma=gamma, random_state=20160820)
    cval = cross_val_score(get_pipeline(estimator),
                           data,
                           labels,
                           scoring='roc_auc',
                           cv=get_cv())
    return cval.mean()


def xgb_cv(max_depth, n_estimators, data, labels):
    estimator = XGBClassifier(max_depth=max_depth,
                              n_estimators=n_estimators,
                              verbosity=1,
                              n_jobs=-1,
                              random_state=20160820)
    cval = cross_val_score(get_pipeline(estimator),
                           data,
                           labels,
                           scoring='roc_auc',
                           cv=get_cv())
    return cval.mean()


def optimize_svc(data, labels):
    def svc_crossval(exp_c, exp_gamma):
        C = 10**exp_c
        gamma = 10**exp_gamma
        return svc_cv(C=C, gamma=gamma, data=data, labels=labels)

    optimizer = BayesianOptimization(f=svc_crossval,
                                     pbounds={
                                         "exp_c": (-5, -3),
                                         "exp_gamma": (-2, 0)
                                     },
                                     random_state=20160820,
                                     verbose=2)

    optimizer.maximize(n_iter=40)

    print("Final result:", optimizer.max)


def optimize_xgb(data, labels):
    def xgb_crossval(max_depth, n_estimators):
        n_estimators = int(n_estimators)
        max_depth = int(max_depth)

        return xgb_cv(max_depth, n_estimators, data, labels)

    optimizer = BayesianOptimization(f=xgb_crossval,
                                     pbounds={
                                         "max_depth": (2, 8),
                                         "n_estimators": (50, 400)
                                     },
                                     random_state=20160820,
                                     verbose=2)

    optimizer.maximize(n_iter=40)

    print("Final result:", optimizer.max)


def mlp_cv(hidden_layer_sizes, alpha, data, labels):
    estimator = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                              alpha=alpha,
                              shuffle=False,
                              verbose=False,
                              max_iter=500,
                              random_state=20160820)
    cval = cross_val_score(get_pipeline(estimator),
                           data,
                           labels,
                           scoring='roc_auc',
                           cv=get_cv())
    return cval.mean()


def optimize_mlp(data, labels):
    def mlp_crossval(hidden_layer_one_size, hidden_layer_two_size,
                     hidden_layer_three_size, hidden_layer_four_size, alpha):
        hidden_layer_one_size = int(hidden_layer_one_size)
        hidden_layer_two_size = int(hidden_layer_two_size)
        hidden_layer_three_size = int(hidden_layer_three_size)
        hidden_layer_four_size = int(hidden_layer_four_size)

        return mlp_cv((hidden_layer_one_size, hidden_layer_two_size,
                       hidden_layer_three_size, hidden_layer_four_size), alpha,
                      data, labels)

    optimizer = BayesianOptimization(f=mlp_crossval,
                                     pbounds={
                                         "hidden_layer_one_size": (10, 200),
                                         "hidden_layer_two_size": (10, 200),
                                         "hidden_layer_three_size": (10, 200),
                                         "hidden_layer_four_size": (10, 200),
                                         "alpha": (0.00001, 0.05)
                                     },
                                     random_state=20160820,
                                     verbose=2)

    optimizer.maximize(n_iter=40)

    print("Final result:", optimizer.max)


if __name__ == "__main__":
    data, labels = load_data()

    print(Colours.green("--- Optimizing Neural Network ---"))
    optimize_mlp(data, labels)

    print(Colours.yellow("--- Optimizing SVM ---"))
    optimize_svc(data, labels)

    print(Colours.red("--- Optimizing XGBoost ---"))
    optimize_xgb(data, labels)
