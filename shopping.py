import csv
import sys
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


TEST_SIZE = 0.4

def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from the CSV file and split it into training and test sets
    evidence, labels = load_data(sys.argv[1])

    # print(f"This is evidence: {evidence}")
    # print(f"This is labels: {labels}")

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train the model and make predictions
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
    Load shopping data from a CSV file `filename` and convert it into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    Evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating-point number
        - Informational, an integer
        - Informational_Duration, a floating-point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating-point number
        - BounceRates, a floating-point number
        - ExitRates, a floating-point number
        - PageValues, a floating-point number
        - SpecialDay, a floating-point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    Labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    evidence = []
    labels = []

    with open(filename) as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Append all necessary fields from the row to the evidence list
            evidence.append(
                [
                    int(row["Administrative"]),
                    float(row["Administrative_Duration"]),
                    int(row["Informational"]),
                    float(row["Informational_Duration"]),
                    int(row["ProductRelated"]),
                    float(row["ProductRelated_Duration"]),
                    float(row["BounceRates"]),
                    float(row["ExitRates"]),
                    float(row["PageValues"]),
                    float(row["SpecialDay"]),
                    # Convert month to the corresponding index
                    check_month(row["Month"]),
                    int(row["OperatingSystems"]),
                    int(row["Browser"]),
                    int(row["Region"]),
                    int(row["TrafficType"]),
                    # Convert VisitorType to 0 or 1
                    int(1 if "Returning_Visitor" == row["VisitorType"] else 0),
                    # Convert Weekend to 0 or 1
                    int(1 if "TRUE" == row["Weekend"] else 0),
                ]
            )

            # Convert Revenue to 0 or 1 and append to the labels list
            labels.append(int(1 if "TRUE" == row["Revenue"] else 0))

        # [0, 0.0, 0, 0.0, 1, 0.0, 0.2, 0.2, 0.0, 0.0, 1, 1, 1, 1, 1, 1, 0] expected result
        # [0, 0.0, 0, 0.0, 1, 0.0, 0.2, 0.2, 0.0, 0.0, 1, 1, 1, 1, 1, 1, 0] sorted by me
        # print(evidence[0])
    return (evidence, labels)


def check_month(month):
    """
    Convert a month abbreviation to an integer index (0 for January, 11 for December).
    """
    months = {
        "Jan": 0,
        "Feb": 1,
        "Mar": 2,
        "Apr": 3,
        "May": 4,
        "June": 5,
        "Jul": 6,
        "Aug": 7,
        "Sep": 8,
        "Oct": 9,
        "Nov": 10,
        "Dec": 11,
    }

    for index in months:
        if month == index:
            # print(f"Month is: {month}, value is: {months[month]}")
            return int(months[index])


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    # Create a k-nearest neighbor classifier with k=1
    model = KNeighborsClassifier(n_neighbors=1)
    # Train the model on the evidence and labels
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
    sensitivity = float(0)
    specificity = float(0)

    # Counter for the total number of true positive labels
    total_true_positive = 0
    # Counter for the total number of true negative labels
    total_true_negative = 0
    # Counter for the total number of labels (total instances)
    total = 0

    for label, prediction in zip(labels, predictions):
        total += 1

        if label == 1:
            # Increment true positive count if both the actual and predicted labels are positive
            total_true_positive += 1
            # Increment sensitivity (true positive rate) if the actual positive label is accurately identified
            # by checking if the predicted label is also positive
            if label == prediction:
                sensitivity += 1

        if label == 0:
            # Increment true negative count if both the actual and predicted labels are negative
            total_true_negative += 1
            # Increment specificity (true negative rate) if the actual negative label is accurately identified
            # by checking if the predicted label is also negative
            if label == prediction:
                specificity += 1

    # print(f"Total = {total}")
    # print(f"Total positive = {total_true_positive}; Total negative = {total_true_negative}")

    # Calculate sensitivity and specificity by dividing the respective counts by the total instances
    sensitivity = sensitivity / total_true_positive
    specificity = specificity / total_true_negative
    # print(f"(sensitivity, specificity) = ({sensitivity}, {specificity})")

    return (sensitivity, specificity)


if __name__ == "__main__":
    main()

# USAGE: python shopping.py shopping.csv
