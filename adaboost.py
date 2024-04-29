import pandas as pd
import numpy
from sklearn.model_selection import train_test_split

# Load the dataset

# url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
# data = pd.readcsv(url)
# I loaded it locally for convenience
path = "The path to penguins.csv" # make sure to replace this with the correct path
data = pd.read_csv(path)

# Display the number of rows and the first 10 rows
print('Dataset description:')
print(data.info())

# Removing rows for 'Chinstrap'
data = data[data['species'] != 'Chinstrap']
print("\nNumber of rows after removing 'Chinstrap':", len(data))
print(data.head(10))

# Removing rows with NaN
data = data.dropna()
print("\nNumber of rows after removing NaN:", len(data))
print(data.head(10))
print("\nNumber of NaN values for each attribute:")
print(data.isna().sum())

data.reset_index(drop=True, inplace=True)


# Calculating mean and variance
print("\nMean for each numeric attribute:")
print(data.mean(numeric_only=True))
print("\nVariance for each numeric attribute:")
print(data.var(numeric_only=True))


# Initializing the distribution D
D = pd.Series([1/len(data)] * len(data))
print("\nDistribution D:")
print(D.head())


# Converting attributes 'species', 'island', 'sex'
mapping = {
    'species': {'Adelie': 0, 'Gentoo': 1},
    'island': {'Biscoe': 0, 'Dream': 1, 'Torgersen': 2},
    'sex': {'Female': 0, 'Male': 1}
}
data['species'] = data['species'].map(mapping['species'])
data['island'] = data['island'].map(mapping['island'])
data['sex'] = data['sex'].map(mapping['sex'])
print(data.dtypes)


def get_splits(data, attribute):
    unique_values = sorted(data[attribute].unique())
    splits = [(unique_values[i] + unique_values[i + 1]) /
              2 for i in range(len(unique_values) - 1)]
    splits.insert(0, unique_values[0] - 0.5)  # Exterior split
    return splits


def get_splits_list(data):
    attributes = ['island', 'bill_length_mm',
                  'flipper_length_mm', 'body_mass_g', 'sex']
    splits_list = []
    for attr in attributes:
        splits_list.extend(get_splits(data, attr))
    return splits_list


# Displaying splits for 'bill_length_mm' and the list of splits
print("\nSplits for 'bill_length_mm':")
print(get_splits(data, 'bill_length_mm'))
splits_list = get_splits_list(data)
print("\nList of splits:")
print(splits_list)


print("\nTotal number of splits:", len(splits_list))


def calculate_split_probs(data, attribute, split, D):
    left = data[data[attribute] <= split]
    right = data[data[attribute] > split]
    result = {"left": {}, "right": {}}
    class_labels = data['species'].unique()

    for label in class_labels:
        result["left"][label] = D[left.index][left['species'] == label].sum()
        result["right"][label] = D[right.index][right['species'] == label].sum()

    return result


def calculate_split_prediction(split_probs):
    label_left = max(split_probs["left"], key=split_probs["left"].get)
    label_right = max(split_probs["right"], key=split_probs["right"].get)
    return [label_left, label_right]


def calculate_split_errors(split_probs):
    # Calculate the highest weight from left and right sides
    left_weights = max(split_probs["left"].values())
    right_weights = max(split_probs["right"].values())

    # Calculate error for the split
    error = 1 - left_weights - right_weights
    return error


def find_best_estimator(data, splits_list, D):
    best_error = float('inf')
    best_estimator = None
    for attribute in data.columns.drop('species'):
        for split in get_splits(data, attribute):
            split_probs = calculate_split_probs(data, attribute, split, D)
            error = calculate_split_errors(split_probs)
            if error < best_error:
                best_error = error
                split_prediction = calculate_split_prediction(split_probs)
                best_estimator = (attribute, split,
                                  split_prediction[0], split_prediction[1])
    return best_estimator, best_error


def get_estimator_weight(error):
    return 0.5 * numpy.log((1 - error + 1e-10) / (error + 1e-10))


def update_D(data, estimator, error, D):
    attribute, split, label_left, label_right = estimator
    new_D = D.copy()
    for i in data.index:
        row = data.loc[i]
        is_correct = (row[attribute] <= split and row['species'] == label_left) or \
                     (row[attribute] > split and row['species'] == label_right)
        if is_correct:
            new_D[i] *= 0.5 / (1 - error)
        else:
            new_D[i] *= 0.5 / error
    return new_D / new_D.sum()


def adaboost(data, niter):
    model = {"estimators": [], "estimators_weights": []}
    D = pd.Series([1/len(data)] * len(data))
    print('Initial distribution is: \n')
    print(D)
    for t in range(niter):
        print(f'Iteration: {t} \n')
        best_estimator, error = find_best_estimator(data, splits_list, D)
        print(f'The best estimator is: {best_estimator} \n')
        print(f'The lowest error is: {error} \n')
        model["estimators"].append(best_estimator)
        alpha = get_estimator_weight(error)
        model["estimators_weights"].append(alpha)
        D = update_D(data, best_estimator, error, D)
        print('Updated probability distribution is: \n')
        print(D)
    return model


def adaboost_predict(X, model):
    predictions = []
    for estimator in model["estimators"]:
        attribute, split, label_left, label_right = estimator
        prediction = 1 if X[attribute] <= split else -1
        predictions.append(prediction)

    total_weighted_vote = numpy.dot(predictions, model["estimators_weights"])

    return 0 if numpy.sign(total_weighted_vote) == 1 else 1


def random_sampling_cross_validation(data, max_niter=50, num_trials=5, test_size=0.2):
    best_niter = 0
    best_accuracy = 0

    for niter in range(1, max_niter + 1):
        trial_accuracies = []

        for trial in range(num_trials):
            # Create training and testing sets through random sampling
            train_data, test_data = train_test_split(data, test_size=test_size)

            # Reset indices for train_data and recalculate the distribution D
            train_data.reset_index(drop=True, inplace=True)
            D = pd.Series([1/len(train_data)] * len(train_data))

            # Train the AdaBoost model
            model = adaboost(train_data, niter)

            # Evaluate the model on the test set
            correct_predictions = sum(adaboost_predict(
                test_data.iloc[i], model) == test_data.iloc[i]['species'] for i in range(len(test_data)))
            accuracy = correct_predictions / len(test_data)
            trial_accuracies.append(accuracy)

        # Calculate the average accuracy for this number of iterations
        avg_accuracy = numpy.mean(trial_accuracies)

        # Update the best results, if applicable
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_niter = niter

    return best_niter, best_accuracy


def main():
    # # Load and preprocess data (make sure this step is completed)
    # data = pd.read_csv('path_to_penguins.csv') # Replace with the correct path
    # # ... (other preprocessing operations)

    # Call the adaboost function
    model = adaboost(data, 5)  # 5 iterations for AdaBoost
    print(model)

    # Define an instance X for prediction
    # Here we use the first instance as an example, but you can choose any other instance
    X = data.iloc[149]

    # Call the adaboost_predict function
    prediction = adaboost_predict(X, model)

    print("Prediction for the instance X is:", prediction)

    # best_niter, best_accuracy = random_sampling_cross_validation(data, max_niter=50)
    # print(f"Optimal number of iterations: {best_niter}, with an accuracy of: {best_accuracy}")


if __name__ == "__main__":
    main()
