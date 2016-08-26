import json
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline


def build_prediction(train, test):
    # Train
    from collections import Counter
    for_sale = Counter(x["seller_id"] for x in train)
    vectorize = lambda datum: (datum["price"],
                               for_sale[datum["seller_id"],
                               datum["seller_power_status"]])

    X_train_title = [item['title'] for item in train]
    X_train = [vectorize(x) for x in train]
    y_train = [x["top_level_category"] for x in train]
    X_test_title = [item['title'] for item in test]
    X_test = [vectorize(x) for x in test]

    cv = CountVectorizer()
    c = LogisticRegression()

    # usar tfidf vectorizer con preprocessor=lambda x: x['title'].lower()
    # Crear Price Vectorizer usando DirectTransformer

    # make_pipeline(make_union(
    #   TfidfVectorizer(),
    #   Price()
    # ), LogisticRegression())

    X_train_title = cv.fit_transform(X_train_title)

    accuracy = cross_val_score(
        c,                   # The classifier
        X_train_title, y_train,    # Train data, used for cross validation
        scoring="accuracy",  # Evaluate the accuracy of the classifier
        cv=10,               # 10-fold cross validation
    ).mean()
    print "Estimated accuracy: %s" % (accuracy*100)

    cv2 = CountVectorizer()
    X_test_title = cv2.fit_transform(X_test_title)

    c.fit(X_train_title, y_train)
    yhat = c.predict(X_test_title)
    return yhat


class DirectTransformer:
    """Utility for building class-like features from a single-point function, but that may need
    some general configuration first (you usually override __init__ for that)
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([self.transform_one(x) for x in X]).reshape((-1, 1))

    def transform_one(self, x):
raise NotImplementedError


def load_data(path):
    return [json.loads(line) for line in open(path)]


if __name__ == "__main__":
    train = load_data("mlm_items_train.jsonlines")
    test = load_data("mlm_items_test.jsonlines")
    prediction = build_prediction(train, test)
    with open("prediction.jsonlines", "wt") as out:
        for x, label in zip(test, prediction):
            d = {
                "id": x["id"],
                "prediction": label
            }
            out.write(json.dumps(d))
            out.write("\n")