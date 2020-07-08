import math
from heapq import nlargest
from random import randint

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold


def generate_real_index(temporary_index, not_included_indices):
    real_index = temporary_index
    for i in sorted(not_included_indices):
        if i <= real_index:
            real_index += 1
        else:
            break
    return real_index


def generate_possible_subsets(set):
    subsets = []
    for item in set:
        subset = set.copy()
        subset.remove(item)
        if len(subset) != 0:
            subsets.append(subset)

    return subsets


def entropy(collection):
    result = 0

    options = set(collection)

    for option in options:
        probability = collection.tolist().count(option) / len(collection)

        result += probability * math.log(probability, 2)

    return - result


def information_gain(feature_values, classes_values):
    overall_entropy = entropy(classes_values)

    feature_possible_values = set(feature_values)

    feature_conditional_entropy = 0
    for feature_possible_value in feature_possible_values:
        value_indices = [index for index, value in enumerate(feature_values) if
                         value == feature_possible_value]

        value_classes = np.array(
            [cl for i, cl in enumerate(classes_values) if i in value_indices])

        feature_conditional_entropy += (len(value_classes) / len(
            classes_values)) * entropy(value_classes)

    return overall_entropy - feature_conditional_entropy


def reorder_using_information_gain(X, y):
    feature_information_gains = [information_gain(X[:, feature], y) for feature
                                 in range(X.shape[1])]

    new_feature_order = [feature_information_gains.index(
        nlargest(n, feature_information_gains)[-1]) for n in
        range(1, len(feature_information_gains) + 1)]

    return new_feature_order


def generate_costs(X, min_cost=0, max_cost=1000):
    return np.array([randint(min_cost, max_cost) for _ in range(X.shape[1])])


def test_stop_criterion(X, y, classifier, feature_selection, feature_reorder,
                    criterion_optimizer, criterion_optimizer2, costs, save_data=None,
                    folds_number=10, debug=True):
    kfold = StratifiedKFold(n_splits=folds_number)
    folds = kfold.split(X, y)
    folds_scores = np.array([])
    folds_costs = np.array([])
    used_costs = costs.copy()
    stop_criterion_values = np.array([])
    stop_criterion_values2 = np.array([])

    for train, test in folds:
        X_train_selected = X[train]
        y_train = y[train]
        X_test_selected = X[test]
        y_test = y[test]

        if feature_reorder is not None:
            new_feature_order = feature_reorder(X_train_selected, y_train)
            X_train_selected = X_train_selected[:, new_feature_order]
            X_test_selected = X_test_selected[:, new_feature_order]
            used_costs = used_costs[new_feature_order]

        selected_features = list(range(X.shape[1]))

        if feature_selection is not None:
            selected_features = feature_selection(X_train_selected, y_train,
                                                  classifier)
            X_train_selected = X_train_selected[:, selected_features]
            X_test_selected = X_test_selected[:, selected_features]


        if criterion_optimizer2 is not None:
            classifier.stop_criterion2 = criterion_optimizer2(X_train_selected,
                                                            y_train)
            stop_criterion_values2 = np.append(stop_criterion_values2,
                                              classifier.stop_criterion2)


        if criterion_optimizer is not None:
            classifier.stop_criterion = criterion_optimizer(X_train_selected,
                                                            y_train)
            stop_criterion_values = np.append(stop_criterion_values,
                                              classifier.stop_criterion)


        if len(stop_criterion_values) > 0:
            print("Stop criterion values: {}".format(stop_criterion_values))

    if save_data is not None:
        save_data.add_data(stop_criterion_values2, 0, stop_criterion_values2, 0, stop_criterion_values)
    return 0, 0


def test_classifier(X, y, classifier, feature_selection, feature_reorder,
                    criterion_optimizer, costs, save_data=None,
                    folds_number=10, debug=True):
    kfold = StratifiedKFold(n_splits=folds_number)
    folds = kfold.split(X, y)
    folds_scores = np.array([])
    folds_costs = np.array([])
    used_costs = costs.copy()
    stop_criterion_values = np.array([])
    for train, test in folds:
        X_train_selected = X[train]
        y_train = y[train]
        X_test_selected = X[test]
        y_test = y[test]

        if feature_reorder is not None:
            new_feature_order = feature_reorder(X_train_selected, y_train)
            X_train_selected = X_train_selected[:, new_feature_order]
            X_test_selected = X_test_selected[:, new_feature_order]
            used_costs = used_costs[new_feature_order]

        selected_features = list(range(X.shape[1]))

        if feature_selection is not None:
            selected_features = feature_selection(X_train_selected, y_train,
                                                  classifier)
            X_train_selected = X_train_selected[:, selected_features]
            X_test_selected = X_test_selected[:, selected_features]

        if criterion_optimizer is not None:
            classifier.stop_criterion = criterion_optimizer(X_train_selected,
                                                            y_train)
            stop_criterion_values = np.append(stop_criterion_values,
                                              classifier.stop_criterion)

        classifier.fit(X_train_selected, y_train)

        results = classifier.predict(X_test_selected)

        accuracy = accuracy_score(y_test, results)

        folds_scores = np.append(folds_scores, accuracy)
        if not hasattr(classifier, 'used_features'):
            cost = sum(used_costs[selected_features])
        else:
            cost = sum(
                [sum(used_costs[:row_used_features]) for row_used_features in
                 classifier.used_features]) / len(X_test_selected)
            print("Used_feature: {}".format(classifier.used_features))
        folds_costs = np.append(folds_costs, cost)
    #
    mean_accuracy = folds_scores.mean()
    mean_cost = folds_costs.mean()

    if debug:
        print("Accuracies for folds: {}".format(folds_scores))
        print("Mean accuracy: {}".format(mean_accuracy))
        print("Costs for folds: {}".format(folds_costs))
        print("Mean cost: {}".format(mean_cost))
        if len(stop_criterion_values) > 0:
            print("Stop criterion values: {}".format(stop_criterion_values))

    if save_data is not None:
        save_data.add_data(folds_scores, mean_accuracy, folds_costs, mean_cost, stop_criterion_values)

    return mean_accuracy, mean_cost
    # return 0, 0


def optimize_sequential_classification_stop_criterion(X, y,
                                                      target_classifier,
                                                      classifier_class,
                                                      criterion_range,
                                                      feature_reorder,
                                                      folds_number=5):
    kfold = StratifiedKFold(n_splits=folds_number)
    folds = kfold.split(X, y)
    target_results = np.array([])
    for train, test in folds:
        X_train_selected = X[train]
        y_train = y[train]
        X_test_selected = X[test]
        y_test = y[test]

        target_classifier.fit(X_train_selected, y_train)
        results = target_classifier.predict(X_test_selected)

        accuracy = accuracy_score(y_test, results)
        target_results = np.append(target_results, accuracy)

    target_accuracy = target_results.mean()

    for stop_criterion in criterion_range:
        classifier = classifier_class(stop_criterion)
        kfold = StratifiedKFold(n_splits=folds_number)
        folds = kfold.split(X, y)
        folds_scores = np.array([])

        for train, test in folds:
            X_train_selected = X[train]
            y_train = y[train]
            X_test_selected = X[test]
            y_test = y[test]

            new_feature_order = feature_reorder(X_train_selected, y_train)
            X_train_selected = X_train_selected[:, new_feature_order]
            X_test_selected = X_test_selected[:, new_feature_order]

            classifier.fit(X_train_selected, y_train)

            results = classifier.predict(X_test_selected)

            accuracy = accuracy_score(y_test, results)

            folds_scores = np.append(folds_scores, accuracy)

        mean_accuracy = folds_scores.mean()
        print(("Decreasing: Stop: {} Mean: {}, Target: {}, Diff{} ").format(stop_criterion,mean_accuracy,target_accuracy,mean_accuracy-target_accuracy))
        offset = 0.015
        if mean_accuracy + offset < target_accuracy:
            return stop_criterion
    return stop_criterion