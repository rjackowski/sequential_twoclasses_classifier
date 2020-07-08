import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from utils import generate_real_index
from sklearn.naive_bayes import MultinomialNB

class NaiveBayes(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.classes = None
        self.model = dict()
        self.classes_prior_probabilities = []
        self.class_instances_counts = []

    def __reset_fields(self):
        self.classes = None
        self.model = dict()
        self.classes_prior_probabilities = []
        self.class_instances_counts = []

    def fit(self, X, y):
        self.__reset_fields()
        self.classes = list(set(y))
        self.classes.sort()

        for cl in self.classes:
            self.classes_prior_probabilities.append(
                y.tolist().count(cl) / len(y))
            if cl not in self.model.keys():
                self.model[cl] = []
            class_instances = [instance for instance_index, instance in
                               enumerate(X.tolist()) if
                               instance_index in [instance_index for
                                                  instance_index, instance_class
                                                  in enumerate(y) if
                                                  instance_class == cl]]
            self.class_instances_counts.append(len(class_instances))
            for feature_index in range(X.shape[1]):
                feature_dictionary = dict()
                feature_values = [class_instance[feature_index] for
                                  class_instance in class_instances]
                distinct_feature_values = list(set(feature_values))
                distinct_feature_values.sort()
                for feature_value in distinct_feature_values:
                    feature_value_probability_in_class = (feature_values.count(
                        feature_value) + 1) / (len(class_instances) + len(
                        distinct_feature_values))
                    feature_dictionary[
                        feature_value] = feature_value_probability_in_class
                self.model[cl].append(feature_dictionary)

    def predict_proba(self, X):
        instances_probabilities = []
        for instance in X:
            current_instance_class_probabilities = []
            for cl in self.classes:
                instance_class_probability = self.classes_prior_probabilities[
                    self.classes.index(cl)]
                for feature_index in range(len(instance)):
                    instance_feature_probability = \
                        self.model[cl][feature_index][
                            instance[feature_index]] if \
                            instance[feature_index] in self.model[cl][
                                feature_index].keys() else 1 / (
                                self.class_instances_counts[
                                    self.classes.index(cl)] + len(
                            self.model[cl][feature_index]))
                    instance_class_probability *= instance_feature_probability
                current_instance_class_probabilities.append(
                    instance_class_probability)
            instance_classes_probabilities = []
            for i in range(len(self.classes)):
                fin_prob = current_instance_class_probabilities[i] / sum(
                    current_instance_class_probabilities) if \
                    current_instance_class_probabilities[i] != 0 else 0
                instance_classes_probabilities.append(fin_prob)
            instances_probabilities.append(instance_classes_probabilities)
        return instances_probabilities

    def predict(self, X):
        return [self.classes[
                    instance_probabilities.index(max(instance_probabilities))]
                for instance_probabilities in self.predict_proba(X)]


class SequentialNaiveBayes(BaseEstimator, ClassifierMixin):
    def __init__(self, stop_criterion=0.95):
        self.naive_bayes_classifier = MultinomialNB()
        self.X = None
        self.y = None
        self.stop_criterion = stop_criterion
        self.used_features = None
        self.classes_ = None

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict_proba(self, X):
        already_predicted = []
        already_predicted_indices = []
        for number_of_features in range(1, self.X.shape[1] + 1):
            X_train = self.X[:, :number_of_features]
            not_yet_predicted = np.array([row for i, row in enumerate(X) if
                                          i not in already_predicted_indices])
            if len(not_yet_predicted) == 0:
                break
            X_predict = not_yet_predicted[:, :number_of_features]

            self.naive_bayes_classifier.fit(X_train, self.y)
            predictions = self.naive_bayes_classifier.predict_proba(X_predict)

            currently_correctly_predicted_indices = []
            for row_number, row in enumerate(predictions):
                if any(probability >= self.stop_criterion for probability in
                       row) or number_of_features == self.X.shape[1]:
                    already_predicted.append((generate_real_index(row_number,
                                                                  already_predicted_indices),
                                              row, number_of_features))
                    currently_correctly_predicted_indices.append(
                        generate_real_index(row_number,
                                            already_predicted_indices))
            already_predicted_indices += currently_correctly_predicted_indices
        self.classes_ = self.naive_bayes_classifier.classes_
        self.used_features = [tpl[2] for tpl in
                               sorted(already_predicted,
                                      key=lambda tpl: tpl[0])]
        return [tpl[1] for tpl in
                sorted(already_predicted, key=lambda tpl: tpl[0])]

    def predict(self, X):
        return [self.naive_bayes_classifier.classes_[
                    list(line_probabilities).index(max(line_probabilities))] for
                line_probabilities in self.predict_proba(X)]
