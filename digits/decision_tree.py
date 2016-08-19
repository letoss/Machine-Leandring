import json
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score


def build_prediction(train, test):
    # Train

    X_train = [x['image'] for x in train]
    y_train = [x["label"] for x in train]
    X_test = [x['image'] for x in test]
    c = DecisionTreeClassifier()

    accuracy = cross_val_score(
        c,                   # The classifier
        X_train, y_train,    # Train data, used for cross validation
        scoring="accuracy",  # Evaluate the accuracy of the classifier
        cv=10,               # 10-fold cross validation
    ).mean()
    print "Estimated accuracy: %s" % (accuracy*100)

    c.fit(X_train, y_train)
    yhat = c.predict(X_test)
    return yhat


def load_data(path):
    return [json.loads(line) for line in open(path)]


if __name__ == "__main__":
    train = load_data("digits_train.jsonlines")
    test = load_data("digits_test.jsonlines")
    prediction = build_prediction(train, test)
    with open("prediction.jsonlines", "wt") as out:
        for x, label in zip(test, prediction):
            d = {
                "id": x["id"],
                "prediction": label
            }
            out.write(json.dumps(d))
            out.write("\n")