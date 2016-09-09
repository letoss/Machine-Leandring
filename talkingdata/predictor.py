import json
import csv
import sys
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline, make_union

PERSONS = {}
PERSONS_TESTS = {}
APP_IDS = {}


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


class OneHotTransformer:
    def __init__(self, func):
        self.f = func

    def fit(self, X, y=None):
        unseen = object()
        seen = set()
        for x in X:
            seen.add(self.f(x))
        self.seen = list(sorted(seen)) + [unseen]
        return self

    def transform(self, X):
        return np.array([self.transform_one(x) for x in X])

    def transform_one(self, x):
        result = [0] * len(self.seen)
        value = self.f(x)
        if value in self.seen:
            result[self.seen.index(value)] = 1
        else:
            result[-1] = 1
        return result


def build_prediction():
    p_age = make_pipeline(
        make_union(
            OneHotTransformer(lambda x: x[1]['phone_brand'].lower()),
            OneHotTransformer(lambda x: x[1]['device_model'].lower())
        ),
        LogisticRegression()
    )

    x_train = [(x, y) for x, y in PERSONS.items()]
    x_test = [(x, y) for x, y in PERSONS_TESTS.items()]
    y_train_age = [y.get('age') for x, y in PERSONS.items()]

    print "fit age predictor"
    p_age.fit(x_train, y_train_age)
    print "predicting age"
    age_prediction = p_age.predict(x_test)

    # accuracy_age = cross_val_score(
    #     p_age,                   # The classifier
    #     x_train, y_train_age,    # Train data, used for cross validation
    #     scoring="accuracy",      # Evaluate the accuracy of the classifier
    #     cv=10,                   # 10-fold cross validation
    # ).mean()
    # print "Estimated accuracy age: %s" % (accuracy_age*100)

    p_gender = make_pipeline(
        make_union(
            OneHotTransformer(lambda x: x[1]['phone_brand'].lower()),
            OneHotTransformer(lambda x: x[1]['device_model'].lower()),
        ),
        LogisticRegression()
    )

    vectorizer = lambda x: 1 if x.lower() == 'm' else 0
    y_train_gender = [vectorizer(y.get('gender')) for x, y in PERSONS.items()]
    # accuracy_gender = cross_val_score(
    #     p_gender,                   # The classifier
    #     x_train, y_train_gender,    # Train data, used for cross validation
    #     scoring="accuracy",         # Evaluate the accuracy of the classifier
    #     cv=10,                      # 10-fold cross validation
    # ).mean()
    # print "Estimated accuracy gender: %s" % (accuracy_gender*100)


def load_gender_age_train():
    print "loading gender age train"
    with open('gender_age_train.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            PERSONS[row['device_id']] = {
                'gender': row['gender'],
                'age': row['age'],
                'group': row['group']
            }


def load_gender_age_test():
    print "loading gender age test"
    with open('gender_age_test.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            PERSONS_TESTS[row['device_id']] = {}


def load_phone_brand_device_model():
    print "loading phone brand device model"
    with open('phone_brand_device_model.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            person = PERSONS.get(row['device_id'])
            if person:
                person.update({
                    'phone_brand': row['phone_brand'],
                    'device_model': row['device_model'],
                })
            person_test = PERSONS_TESTS.get(row['device_id'])
            if person_test is not None:
                PERSONS_TESTS[row['device_id']] = {
                    'phone_brand': row['phone_brand'],
                    'device_model': row['device_model'],
                }


def load_app_events():
    print "loading app events"
    with open('app_events.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            app_id = row.get('app_id')
            exist = APP_IDS.get(app_id)
            if not exist:
                APP_IDS[app_id] = row.get('event_id')


if __name__ == "__main__":
    if len(sys.argv) != 2:
        exit('Incorrect parameters. Only outfile needs to be provided')
    load_gender_age_train()
    # load_app_events()
    load_gender_age_test()
    load_phone_brand_device_model()
    prediction = build_prediction()
