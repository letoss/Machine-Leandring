import json
import csv
import sys
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

PERSONS = {}

def build_prediction(train, test):
    p = make_pipeline(TfidfVectorizer(), LogisticRegression())

    X_train_id  = [x for x, y in PERSONS.items()]
    y_train = [x.get('age') for x in PERSONS]

    accuracy = cross_val_score(
        c,                   # The classifier
        X_train_id, y_train,    # Train data, used for cross validation
        scoring="accuracy",  # Evaluate the accuracy of the classifier
        cv=10,               # 10-fold cross validation
    ).mean()
    print "Estimated accuracy: %s" % (accuracy*100)

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


if __name__ == "__main__":
    if len(sys.argv) != 2:
        exit('Incorrect parameters. Only outfile needs to be provided')
    load_gender_age_train()
    load_phone_brand_device_model()
    import ipdb; ipdb.set_trace()
    
    prediction = build_prediction(train, test)
    with open(sys.argv[1], "wt") as out:
        for x, label in zip(test, prediction):
            d = {
                "id": x["id"],
                "prediction": label
            }
            out.write(json.dumps(d))
            out.write("\n")
