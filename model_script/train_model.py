from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from utils.text_encoding import categorical_encoding
from utils.text_preprocessing import *
from utils.text_tokenization import *
from model_script.ml_models_functions import *

def text_preprocessing(text_list):
    texts = remove_punctuation_number(text_list)
    texts = lowercase_text(texts)
    texts = remove_stopwords(texts)
    texts = convert_into_bow(texts)
    return texts


def train_model_text_cls(X, y):
    model_evaluation_result = {}

    texts = text_preprocessing(X)
    labels, encoder = categorical_encoding(y)

    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2)

    # train naive bayes model
    nb_evaluation = naive_bayes_model(X_train, y_train, X_test, y_test)
    model_evaluation_result["naive_bayes"] = nb_evaluation

    # train logistic regression model
    lr_evaluation = logistic_regression_model(X_train, y_train, X_test, y_test)
    model_evaluation_result["logistic_regression"] = lr_evaluation

    # train lightgbm model
    lgb_evaluation = lightgbm_model(X_train, y_train, X_test, y_test)
    model_evaluation_result["lightgbm"] = lgb_evaluation

    return model_evaluation_result

