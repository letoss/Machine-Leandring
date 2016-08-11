import json
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron


def build_prediction(train, test):
    # Train
    from collections import Counter
    for_sale = Counter(x["seller_id"] for x in train)
    vectorize = lambda datum: (datum["price"], for_sale[datum["seller_id"]])
    X_train = [vectorize(x) for x in train]
    y_train = [x["top_level_category"] for x in train]
    X_test = [vectorize(x) for x in test]
    c = KNeighborsClassifier(1)
    # c = DecisionTreeClassifier()
    # c = Perceptron()
    c.fit(X_train, y_train)
    yhat = c.predict(X_test)
    return yhat


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
