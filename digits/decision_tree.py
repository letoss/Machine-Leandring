import json
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA


def build_prediction(train, test):
    # Train
    X_train = [x['image'] for x in train]
    y_train = [x["label"] for x in train]
    X_test = [x['image'] for x in test]

    c = DecisionTreeClassifier()
    pca = PCA()
    p = Pipeline(steps=[
        ('pca', pca),
        ('clf', DecisionTreeClassifier())
    ])

    # X_train = pca.fit_transform(X_train)

    accuracy = cross_val_score(
        p,                   # The pipeline
        X_train, y_train,    # Train data, used for cross validation
        scoring="accuracy",  # Evaluate the accuracy of the classifier
        cv=10,               # 10-fold cross validation
    ).mean()
    print "Estimated accuracy: %s" % (accuracy*100)

    # TODO:
    # Crear un pipeline con:
    # Vectorizer (en este caso es image solo, el vectorizer no haria nada)
    # Reductor dimension (sklearn.manifold.TSNE)
    # clasificator
    # Llamar p.fit(X_train, y_train)
    # llamar p.predict(X_test)

    p.fit(X_train, y_train)
    yhat = p.predict(X_test)
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