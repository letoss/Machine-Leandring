import json
import sys


def exclude_counter(number_list):
    result = 0
    for num in number_list:
        if num > 160:
            result = result + 1
    return result


def build_prediction(train, test):
    """
    Edit here. Should return one prediction for each element in test.
    """
    know = {}
    for number in train:
        image = number.get('image')
        label = number.get('label')
        stored_count = know.get(label)
        if stored_count:
            know[label] = exclude_counter(image) + stored_count / 2
        else:
            know[label] = exclude_counter(image)

    output = []
    for number in test:
        count = exclude_counter(number.get('image'))
        best_count = 999999999999
        best_label = None
        for label, know_count in know.iteritems():
            result = abs(count - know_count)
            if result < best_count:
                best_count = result
                best_label = label
        output.append(best_label)

    return output


def load_data(path):
    return [json.loads(line) for line in open(path)]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        exit('Incorrect parameters. Only outfile needs to be provided')
    train = load_data("digits_train.jsonlines")
    test = load_data("digits_test.jsonlines")
    prediction = build_prediction(train, test)
    with open(sys.argv[1], "wt") as out:
        for x, label in zip(test, prediction):
            d = {
                "id": x["id"],
                "prediction": label
            }
            out.write(json.dumps(d))
            out.write("\n")
