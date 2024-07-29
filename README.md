# Shopping Predictor

## Project Description

This project involves building a nearest-neighbor classifier to predict whether online shopping customers will complete a purchase based on their browsing behavior. The classifier uses a dataset of around 12,000 user sessions, analyzing features like pages visited, session duration, bounce rates, and more.

## Features

- **Data Preprocessing**: Load and process data from a CSV file.
- **Model Training**: Train a k-nearest-neighbor classifier using `scikit-learn`.
- **Evaluation Metrics**: Calculate sensitivity (true positive rate) and specificity (true negative rate).

## Usage

1. **Setup**:
   ```sh
   pip install scikit-learn
   ```

2. **Run the Program**:
   ```sh
   python shopping.py shopping.csv
   ```

## Functions to Implement

1. **`load_data(filename)`**: Load and preprocess data from CSV.
2. **`train_model(evidence, labels)`**: Train the k-nearest-neighbor model.
3. **`evaluate(labels, predictions)`**: Calculate sensitivity and specificity.

## Evaluation

- **Correct**: Number of correct predictions.
- **Incorrect**: Number of incorrect predictions.
- **True Positive Rate**: Proportion of actual positives correctly identified.
- **True Negative Rate**: Proportion of actual negatives correctly identified.

## Results Explanation

Run the program:

```bash
python shopping.py shopping.csv
```

After running the program, the output will include:

```bash
Correct: 4078
Incorrect: 854
True Positive Rate: 40.61%
True Negative Rate: 90.24%
```

### Interpretation of Results

- **Correct Predictions (4078)**: The number of user sessions where the model accurately predicted whether a purchase was completed.
- **Incorrect Predictions (854)**: The number of user sessions where the model's prediction was incorrect.
- **True Positive Rate (Sensitivity) (40.61%)**: This metric indicates that 40.61% of the actual purchases were correctly identified by the model. It reflects the model's ability to identify users who will make a purchase.
- **True Negative Rate (Specificity) (90.24%)**: This metric indicates that 90.24% of the non-purchases were correctly identified by the model. It reflects the model's ability to identify users who will not make a purchase.

These metrics help in understanding the performance of the classifier in distinguishing between customers who are likely to complete a purchase and those who are not. The higher the sensitivity and specificity, the better the model is at making accurate predictions.
