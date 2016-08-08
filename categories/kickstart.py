import json
import sys


def euclidean_distance(v1, v2):
    """
   Ex:
       euclidean_distance([0, 0], [0, 1]) == 1
       euclidean_distance([0, 0], [1, 1]) == 1.41...
   """
    if len(v1) != len(v2):
        raise ValueError("Vectors have different dimension")
    accum = 0.0
    for x1, x2 in zip(v1, v2):
        accum += (x1 - x2) ** 2
    return accum ** 0.5


def build_prediction(train, test):
    """
    Edit here. Should return one prediction for each element in test.
    """
    def descriptor(items):
        return [[item.get('seller_id'), item.get('price')] for item in items]

    def target(items):
        return [x.get('top_level_category') for x in items]

    items_description = descriptor(train)
    items_target = target(train)

    test_description = descriptor(test)

    output = []
    for td in test_description:
        best_distance = 9999999999
        best_index = 0
        for i, id in enumerate(items_description):
            result = euclidean_distance(td, id)
            if result < best_distance:
                best_index = i
                best_distance = result
        output.append(items_target[best_index])

    # return ['MLM1384' for x in test]
    return output


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
