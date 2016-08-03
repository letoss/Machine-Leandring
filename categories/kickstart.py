import json
import sys


def build_prediction(train, test):
    """
    Edit here. Should return one prediction for each element in test.
    """
    knowlege = {}
    for x in train:
        category = x.get('top_level_category')
        price = x.get('price')
        
        if category not in knowlege:
            knowlege[category] = price
        else:
            knowlege[category] = (price + knowlege[category]) / 2

    for x in test:
        price = x.get('price')
        best = 999999999999
        for category, i in knowlege.iteritems():
            result = price % i

    import ipdb; ipdb.set_trace()    

    return ['MLM1384' for x in test]


def load_data(path):
    return [json.loads(line) for line in open(path)]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        exit('Incorrect parameters. Only outfile needs to be provided')
    train = load_data("mlm_items_train.jsonlines")
    test = load_data("mlm_items_test.jsonlines")
    prediction = build_prediction(train, test)
    with open(sys.argv[1], "wt") as out:
        for x, label in zip(test, prediction):
            d = {
                "id": x["id"],
                "prediction": label
            }
            out.write(json.dumps(d))
            out.write("\n")
