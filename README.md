# Penguins AdaBoost Project

## Overview

This project implements an AdaBoost classifier to predict penguin species from physical measurements. It utilizes a dataset containing details about different species of penguins, and applies machine learning techniques to predict species based on attributes like bill length, flipper length, body mass, and others.

## Dataset Description

The dataset used in this project is the `penguins` dataset, which includes various measurements such as bill length, flipper length, and body mass for different species of penguins. For the purpose of this demonstration, the dataset is preprocessed to exclude certain categories and handle missing values.

## Features

- **Data Preprocessing:** Removes specific categories ('Chinstrap' species) and handles missing values.
- **Feature Conversion:** Converts categorical features into numerical codes for processing.
- **AdaBoost Implementation:** Implements the AdaBoost algorithm to boost the classification accuracy.
- **Predictive Analysis:** Uses the trained AdaBoost model to predict penguin species.

## File Descriptions

- `penguins.csv`: The dataset file containing penguin observations.
- `adaboost_penguins.py`: The main Python script implementing the AdaBoost classifier.

## Dependencies

- Python 3.x
- Pandas
- NumPy
- scikit-learn

## Usage

1. **Loading the Dataset**: Ensure the dataset is located in a path accessible by the script or modify the path accordingly in the script.
2. **Running the Script**: Execute the script using Python to train the AdaBoost model and make predictions.

    ```bash
    python adaboost_penguins.py
    ```

3. **Model Evaluation**: The script will output the effectiveness of the model and can predict the species of a penguin based on the input features.

## Functions Overview

- `get_splits(data, attribute)`: Calculates possible split points for a given attribute.
- `calculate_split_probs(data, attribute, split, D)`: Calculates the probabilities of each class for a given split.
- `find_best_estimator(data, splits_list, D)`: Identifies the best threshold split and attribute based on the AdaBoost criteria.
- `update_D(data, estimator, error, D)`: Updates the distribution for the next iteration of AdaBoost.
- `adaboost(data, niter)`: Implements the AdaBoost algorithm over a specified number of iterations.
- `adaboost_predict(X, model)`: Predicts the class label for a given input using the AdaBoost model.

## Model Outputs

- **Probability Distribution Updates**: At each iteration, the model updates the probability distributions of the data points.
- **Best Estimator and Error**: Displays the best estimator (attribute and split) and the associated error at each iteration.
- **Final Prediction**: Outputs the predicted class for a test instance.

## Future Enhancements

- **Cross-validation Implementation**: Implement random sampling cross-validation to find the optimal number of boosting iterations for maximum accuracy.
- **Expand Dataset Utilization**: Include more categories and features for a more robust model.
- **GUI Integration**: Develop a graphical user interface for easier interaction with the model.

