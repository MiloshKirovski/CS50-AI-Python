import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4

column_types = {
    'Administrative': int,
    'Informational': int,
    'ProductRelated': int,
    'Month': int,
    'OperatingSystems': int,
    'Browser': int,
    'Region': int,
    'TrafficType': int,
    'VisitorType': int,
    'Weekend': int,
    'Administrative_Duration': float,
    'Informational_Duration': float,
    'ProductRelated_Duration': float,
    'BounceRates': float,
    'ExitRates': float,
    'PageValues': float,
    'SpecialDay': float,
}


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    with open(filename, mode='r') as f:
        evidence = []
        labels = []
        reader = csv.reader(f)

        header = next(reader)
        column_indices = {name: index for index, name in enumerate(header)}

        for row in reader:
            row_data = preprocess_row(row, column_indices)
            evidence.append(row_data[:-1])
            labels.append(1 if row_data[-1] == 'TRUE' else 0)

        return evidence, labels


def preprocess_row(row, column_indices):
    row_data = row.copy()
    for column, col_type in column_types.items():
        index = column_indices[column]
        if column == 'Month':
            row_data[index] = return_month(row_data[index])
        elif column == 'VisitorType':
            row_data[index] = 1 if row_data[index] == 'Returning_Visitor' else 0
        elif column == 'Weekend':
            row_data[index] = 1 if row_data[index] == 'TRUE' else 0
        elif col_type == float:
            row_data[index] = float(row_data[index])
        elif col_type == int:
            row_data[index] = int(row_data[index])
    return row_data


def return_month(name):
    month = (
        0 if name == 'Jan' else
        1 if name == 'Feb' else
        2 if name == 'Mar' else
        3 if name == 'Apr' else
        4 if name == 'May' else
        5 if name == 'June' else
        6 if name == 'Jul' else
        7 if name == 'Aug' else
        8 if name == 'Sep' else
        9 if name == 'Oct' else
        10 if name == 'Nov' else
        11
    )
    return month


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    positive_correct = 0
    negative_correct = 0

    total_positive = 0
    total_negative = 0

    for label, prediction in zip(labels, predictions):
        if prediction == 1 and label == 1:
            positive_correct += 1
        elif prediction == 0 and label == 0:
            negative_correct += 1

        if label == 1:
            total_positive += 1
        else:
            total_negative += 1

    sensitivity = positive_correct / total_positive
    specificity = negative_correct / total_negative

    return sensitivity, specificity


if __name__ == "__main__":
    main()
