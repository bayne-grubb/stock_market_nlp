import pickle

import pandas as pd
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from preprocessing_pipeline import preprocessing_pipeline


def load_data():
    train_X = pd.read_csv('./data/assessment_training_data.csv',
                          parse_dates=["Date"],
                          index_col=0)
    test_X = pd.read_csv('./data/assessment_testing_data.csv',
                         parse_dates=["Date"],
                         index_col=0)
    labeled_spy = pd.read_csv('./data/labeled_spy_data.csv',
                              parse_dates=["Date"],
                              index_col=0)
    train_X = train_X.set_index('Date')
    test_X = test_X.set_index('Date')
    labeled_spy = labeled_spy.set_index('Date')
    train_X = train_X[train_X.index.dayofweek < 5]
    test_X = test_X[test_X.index.dayofweek < 5]
    train_Y = labeled_spy.loc[train_X.index]['Drop']
    test_Y = labeled_spy.loc[test_X.index]['Drop']
    actual_fucking_values_train = train_Y.notnull()
    actual_fucking_values_test = test_Y.notnull()
    train_Y = list(train_Y.loc[actual_fucking_values_train])
    test_Y = list(test_Y.loc[actual_fucking_values_test])
    train_X = train_X.loc[actual_fucking_values_train]
    test_X = test_X.loc[actual_fucking_values_test]
    return test_X, train_X.Text, test_Y, train_Y, labeled_spy


def calculate_position_result(init_val, final_val, position_size=10):
    return (init_val - final_val) * position_size


def setup_func():
    #do not run, initial setup only
    vectorizer = CountVectorizer(tokenizer=preprocessing_pipeline,
                                 max_df=0.9,
                                 min_df=0.07)
    test_X, train_X, test_Y, train_Y, labeled_spy = load_data()

    xgb_top = XGBClassifier(max_depth=6, n_estimators=50, n_jobs=-1)
    xgb_simple = XGBClassifier(max_depth=4, n_estimators=59, n_jobs=-1)
    mlp_top = MLPClassifier(hidden_layer_sizes=(11, 142, 33, 99),
                            alpha=0.03375,
                            shuffle=False,
                            max_iter=500)
    mlp_simple = MLPClassifier(hidden_layer_sizes=(27, 196, 49, 126),
                               alpha=0.04579)
    svm_top = svm.SVC(C=0.00022594357702209782, gamma=0.04920395356814509)
    svm_simple = svm.SVC(C=0.00001, gamma=1)

    models = [xgb_top, xgb_simple, mlp_top, mlp_simple, svm_top, svm_simple]
    model_names = [
        "XGB_Top", "XGB_Simple", "MLP_Top", "MLP_Simple", "SVM_Top",
        "SVM_Simple"
    ]

    vectorizer.fit(train_X)
    pickle.dump(vectorizer, open("vector.sav", "wb"))
    print("Done Fitting, starting Vectorization")

    train_X = vectorizer.transform(train_X)

    for model, model_name in zip(models, model_names):
        print("Fitting model: " + model_name)
        model.fit(train_X, train_Y)
        pickle.dump(model, open(model_name + ".sav", "wb"))


def retrain_svm_because_it_sucks():
    svm_top = svm.SVC(C=10**-2.856, gamma=10**-1.004)
    svm_simple = svm.SVC(C=10**-1.004, gamma=10**-1.01)
    test_X, train_X, test_Y, train_Y, labeled_spy = load_data()
    vectorizer = pickle.load(open("vector.sav", "rb"))
    train_X = vectorizer.transform(train_X)
    svm_top.fit(train_X, train_Y)
    svm_simple.fit(train_X, train_Y)
    pickle.dump(svm_top, open("SVM_Top.sav", "wb"))
    pickle.dump(svm_simple, open("SVM_Simple.sav", "wb"))


#retrain_svm_because_it_sucks()
setup_func()
vectorizer = pickle.load(open("vector.sav", "rb"))
xgb_top = pickle.load(open("XGB_Top.sav", "rb"))
xgb_simple = pickle.load(open("XGB_Simple.sav", "rb"))
mlp_top = pickle.load(open("MLP_Top.sav", "rb"))
mlp_simple = pickle.load(open("MLP_Simple.sav", "rb"))
svm_top = pickle.load(open("SVM_Top.sav", "rb"))
svm_simple = pickle.load(open("SVM_Simple.sav", "rb"))

test_X, _, test_Y, _, labeled_spy = load_data()
models = [xgb_top, xgb_simple, mlp_top, mlp_simple, svm_top, svm_simple]
model_names = [
    "XGB_Top", "XGB_Simple", "MLP_Top", "MLP_Simple", "SVM_Top", "SVM_Simple"
]

test_X_vectorized = vectorizer.transform(test_X.Text)
xgb_top_predictions = xgb_top.predict(test_X_vectorized)
xgb_simple_predictions = xgb_simple.predict(test_X_vectorized)
mlp_top_predictions = mlp_top.predict(test_X_vectorized)
mlp_simple_predictions = mlp_simple.predict(test_X_vectorized)
svm_top_predictions = svm_top.predict(test_X_vectorized)
svm_simple_predictions = svm_simple.predict(test_X_vectorized)

model_predictions = [
    xgb_top_predictions, xgb_simple_predictions, mlp_top_predictions,
    mlp_simple_predictions, svm_top_predictions, svm_simple_predictions
]


def print_scores(model_predictions, model_names, test_Y):
    for model_prediction, model_name in zip(model_predictions, model_names):
        print("{} Accuracy: {}, and Precision: {}".format(
            model_name, accuracy_score(test_Y, model_prediction),
            precision_score(test_Y, model_prediction)))


print_scores(model_predictions, model_names, test_Y)


def naive_backtest():
    for model, model_name in zip(models, model_names):
        print("Calculating profit for model: " + model_name)

        position_results_list = []
        for date in test_X.index:
            text = test_X.loc[date]["Text"]
            prediction = model.predict(vectorizer.transform([text]))
            #print(prediction[0])
            if prediction[0]:
                #print("Shorting stock on date:" + str(date))
                spy_day = labeled_spy.loc[date]
                openboi = spy_day["Open"]
                closeboi = spy_day["Close"]
                position_result = calculate_position_result(openboi, closeboi)
                #print("Position Result: " + str(position_result))
                position_results_list.append(position_result)
        #print(position_results_list)
        print("Money for model {} is {}".format(model_name,
                                                sum(position_results_list)))


naive_backtest()


def smart_backtest():
    position_results_list = []
    for date in test_X.index:
        predictions = []
        for model, model_name in zip(models, model_names):
            text = test_X.loc[date]["Text"]
            prediction = model.predict(vectorizer.transform([text]))
            predictions.append(prediction)
        if sum(predictions) > 1:
            #print("Shorting stock on date:" + str(date))
            spy_day = labeled_spy.loc[date]
            openboi = spy_day["Open"]
            closeboi = spy_day["Close"]
            if sum(predictions) > 2:
                position_size = 100
            if sum(predictions) > 3:
                position_size = 200
            else:
                position_size = 10
            position_result = calculate_position_result(
                openboi, closeboi, position_size)
            #print("Position Result: " + str(position_result))
            position_results_list.append(position_result)
    #print(position_results_list)
    print("Money for smart_backtesting is {}".format(
        sum(position_results_list)))


smart_backtest()
