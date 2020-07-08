from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

class Sequential2ClassesNaiveBayes(BaseEstimator, ClassifierMixin):
    def __init__(self, stop_criterion=0.87):
        self.naive_bayes_classifier = MultinomialNB()
        self.X = None
        self.y = None
        self.stop_criterion = stop_criterion
        self.naive_bayes_classifier.stop_criterion = self.stop_criterion
        self.used_features = None
        self.classes_ = None
        self.time = 0

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict_proba(self, X):
        probe_amount = len(X)
        feature_amount = self.X.shape[1] + 1
        number = 1
        unpredicted = [a for a in range(probe_amount)]
        myRange = np.array([np.array([a, 0, []]) for a in unpredicted])

        while number <= feature_amount:
            self.naive_bayes_classifier = MultinomialNB()
            self.naive_bayes_classifier.stop_criterion = self.stop_criterion
            self.naive_bayes_classifier.fit(self.X[:, :number], self.y)
            if len(unpredicted) == 0:
                break;
            X_to_predict = X[unpredicted, :number]
            prediction = self.naive_bayes_classifier.predict_proba(X_to_predict)
            new_unpredicted = []
            for index, predict in enumerate(prediction):
                if(any(prop > self.stop_criterion for prop in predict) or number == feature_amount):
                    X_index = unpredicted[index]
                    myRange[X_index][1] = number
                    myRange[X_index][2] = predict
                else:
                    new_unpredicted.append(unpredicted[index])
            unpredicted = new_unpredicted
            number+=1

        self.used_features = myRange[:,1]
        self.classes_ = self.naive_bayes_classifier.classes_
        return myRange[:,2]

    def predict(self, X):
       return  [self.naive_bayes_classifier.classes_[
                    list(line_probabilities).index(max(line_probabilities))] for
                line_probabilities in self.predict_proba(X)]


class SequentialNaiveBayesOneVsRest(BaseEstimator, ClassifierMixin):
    def __init__(self, stop_criterion=0.95):
        self.sequentialNaiveBayes = Sequential2ClassesNaiveBayes()
        self.used_features = []
        self.stop_criterion = stop_criterion
        self.sequentialNaiveBayes.stop_criterion = stop_criterion

    def fit(self, X, y):
        self.sequentialNaiveBayes.stop_criterion = self.stop_criterion
        self.X = X
        self.y = y
        self.used_features = []

    def predict_proba(self,X_predict):
        self.classes = np.unique(self.y)
        result = []

        for x in X_predict:
            allClassesPred = []

            used_features = 0
            for z_class in self.classes:
                oneVsResty = np.where(self.y == z_class, self.y, "Other")
                self.sequentialNaiveBayes.fit(self.X, oneVsResty)
                c= np.array([x])
                temp_predictions = self.sequentialNaiveBayes.predict_proba(c)
                if self.sequentialNaiveBayes.used_features[0] > used_features:
                    used_features = self.sequentialNaiveBayes.used_features[0]
                index = np.where(self.sequentialNaiveBayes.classes_ == z_class)
                for pred in temp_predictions:
                    allClassesPred.append(pred[index[0][0]])
            self.used_features.append(used_features)
            result.append(allClassesPred)
        return(result)


    def predict(self, X):
        a=[self.classes[
                    instance_probabilities.index(max(instance_probabilities))]
                for instance_probabilities in self.predict_proba(X)]
        return a


def optimize_stop_criterion_with_divide(X, y,
                                                      target_classifier,
                                                      classifier_class,
                                                      start,
                                                      finish,
                                                      feature_reorder,
                                                      step=0.01,
                                                      folds_number=5):

    kf = StratifiedKFold(n_splits=folds_number)
    accuracyTable = np.array([])
    checked = {}
    finished = False

    for train,test in kf.split(X,y):
        target_classifier.fit(X[train], y[train])
        prediction = target_classifier.predict(X[test])
        accuracy = accuracy_score(y[test], prediction)
        accuracyTable = np.append(accuracyTable, accuracy)
    targetAccuracy = accuracyTable.mean()
    criterion_range = np.arange(start, finish + step, step)

    while not finished:
        if len(criterion_range) == 1:
            stop_criterion = criterion_range[0]
            break
        if len(criterion_range) == 2:
            if not criterion_range[0] in checked.keys():
                mid_index = 0
            elif not criterion_range[1] in checked.keys():
                mid_index = 1
            else:
                value_first = checked[criterion_range[0]]
                if value_first >= targetAccuracy:
                    stop_criterion = criterion_range[0]
                else:
                    stop_criterion = criterion_range[1]
                break
        else:
            mid_index = int(len(criterion_range) / 2)
        mid_value = criterion_range[mid_index]

        #Run classifier test
        classifier = classifier_class(stop_criterion=mid_value)
        accuracyTable = np.array([])
        kf = StratifiedKFold(n_splits=folds_number)

        for train,test in kf.split(X,y):
            X_train = X[train]
            X_test = X[test]
            new_feature_order = feature_reorder(X_train, y[train])
            X_train = X_train[:, new_feature_order]
            X_test = X_test[:, new_feature_order]
            classifier.fit(X_train, y[train])
            prediction = classifier.predict(X_test)
            accuracy = accuracy_score(y[test], prediction)
            accuracyTable = np.append(accuracyTable, accuracy)
        meanAccuraccy = accuracyTable.mean()

        print(("Dividing: Stop: {} Mean: {}, Target: {}, Diff{} ").format(mid_value, meanAccuraccy, targetAccuracy, meanAccuraccy - targetAccuracy))
        delta = 0.015
        if (meanAccuraccy + delta )>= targetAccuracy:
            criterion_range = criterion_range[:mid_index + 1]
        else:
            criterion_range = criterion_range[mid_index:]
        checked[mid_value] = meanAccuraccy

    return stop_criterion