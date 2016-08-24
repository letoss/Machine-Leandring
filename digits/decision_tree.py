import json
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, TruncatedSVD


def build_prediction(train, test):
    # Train
    X_train = [x['image'] for x in train]
    y_train = [x["label"] for x in train]
    X_test = [x['image'] for x in test]

    c = DecisionTreeClassifier()
    pca = PCA(n_components=3)
    tsvd = TruncatedSVD(n_components=30)

    # PCA
    p1 = Pipeline(steps=[
        ('pca', pca),
        ('clf', DecisionTreeClassifier())
    ])

    # TSVD
    p2 = Pipeline(steps=[
        ('tsvd', tsvd),
        ('clf', DecisionTreeClassifier())
    ])

    # X_train = pca.fit_transform(X_train)

    accuracy = cross_val_score(
        p2,                   # The pipeline
        X_train, y_train,    # Train data, used for cross validation
        scoring="accuracy",  # Evaluate the accuracy of the classifier
        cv=10,               # 10-fold cross validation
    ).mean()
    print "Estimated accuracy: %s" % (accuracy*100)

    p2.fit(X_train, y_train)
    yhat = p2.predict(X_test)
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


"""
PCA n_components = 2
Estimated accuracy: 81.118473539
real    46m35.815s
user    30m23.487s
sys 0m5.234s

TSVD
n_components = 20
train[:1000]
Estimated accuracy: 100.0
real    0m6.561s
user    0m6.214s
sys 0m0.339s

n_components = 20
train[:10000]
Estimated accuracy: 99.58000979
real    0m17.549s
user    0m17.050s
sys 0m0.480s

n_components = 30
Estimated accuracy: 84.6533297756
real    2m9.632s
user    2m8.271s
sys 0m1.235s

n_components = 40
Estimated accuracy: 84.216700006
real    2m38.248s
user    2m36.739s
sys 0m1.219s

n_components = 50
Estimated accuracy: 83.8850440987
real    3m2.176s
user    3m0.932s
sys 0m1.115s
"""
