import os
import joblib
import lightgbm as lgb
from sklearn.naive_bayes import MultinomialNB
from utils.model_evaluation import evaluate_model
from sklearn.linear_model import LogisticRegression

os.makedirs("models", exist_ok=True)

def naive_bayes_model(X_train, y_train, X_test, y_test):
    model = MultinomialNB()
    
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    nb_evaluation = evaluate_model(y_test, y_pred)

    joblib.dump(model, "models/naive_bayes.pkl")
    return nb_evaluation

def logistic_regression_model(X_train, y_train, X_test, y_test):
    model = LogisticRegression(max_iter=1000)
    
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    lr_evaluation = evaluate_model(y_test, y_pred)
    joblib.dump(model, "models/logistic_regression.pkl")
    return lr_evaluation


def lightgbm_model(X_train, y_train, X_test, y_test):
    model = lgb.LGBMClassifier(n_estimators=100, boosting_type='gbdt')
    
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    lgb_evaluation = evaluate_model(y_test, y_pred)
    joblib.dump(model, "models/lightgbm.pkl")
    return lgb_evaluation
