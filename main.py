from functools import partial

import os
import numpy as np
import pandas as pd
from xlwt import Workbook
from sklearn import preprocessing
import time
import seq_classifier

from feature_selection import best_first_feature_selection, \
    greedy_cost_sensitive_feature_selection, \
    best_first_cost_sensitive_feature_selection
from naive_bayes import SequentialNaiveBayes, NaiveBayes
from utils import reorder_using_information_gain, generate_costs, \
    test_classifier, test_stop_criterion, \
    optimize_sequential_classification_stop_criterion
from sklearn.naive_bayes import MultinomialNB
from dataSaving import SaveToExcel
from sklearn.preprocessing import OrdinalEncoder



CROSS_VALIDATION_FOLDS_NUMBER = 10

DATA_FILES_DIRECTORY = "./data"

def main():
    # cProfile.run('execute()')
    execute()

def execute():
    files = os.listdir(DATA_FILES_DIRECTORY)

    for file in files:
        X, y = retrieve_data(file)

        costs = generate_costs(X, 0, 100)
        print("Costs used for this file: {}".format(costs))

        saveToExcel = SaveToExcel(name=file)

        execute_Naive_Bayes(X,y,costs,saveToExcel)
        # execute_feature_selection_comparation(X,y,costs,saveToExcel)
        execute_multiclasses_sequential(X,y,costs,saveToExcel)
        # execute_onevsrest_sequential_decrease(X,y,costs,saveToExcel)
        execute_onevsrest_sequential_divide(X,y,costs,saveToExcel)
        # test_stop(X,y,costs,saveToExcel)

        #########Save file in excel
        saveToExcel.close()


def retrieve_data(file):
    data = pd.read_csv(os.path.join(DATA_FILES_DIRECTORY, file)).to_numpy()

    print("Processing file: {}".format(file))
    X = data[:, :-1]
    y = data[:, -1]

    # change string values into integer
    encoder = OrdinalEncoder()
    encoder.fit(X)
    return X,y
    X = encoder.transform(X)



def test_stop(X,y,costs,saveToExcel):
        criterion_optimizer = partial(
            seq_classifier.optimize_stop_criterion_with_divide,
            target_classifier=MultinomialNB(),
            # classifier_class=seq_classifier.SequentialNaiveBayesOneVsRest,
            classifier_class=SequentialNaiveBayes,
            start=0.5,
            finish=1,
            step=0.01,
            feature_reorder=reorder_using_information_gain,
        )

        criterion_optimizer2 = partial(
            optimize_sequential_classification_stop_criterion,
            target_classifier=MultinomialNB(),
            # classifier_class=seq_classifier.SequentialNaiveBayesOneVsRest,
            classifier_class=SequentialNaiveBayes,
            criterion_range=np.arange(1, 0.5 - 0.01, -0.01),
            feature_reorder=reorder_using_information_gain)

        print(
            "Sequential One vs Rest classifier")
        # sequential_naive_bayes_classifier = seq_classifier.SequentialNaiveBayesOneVsRest()
        sequential_naive_bayes_classifier = SequentialNaiveBayes()

        saveToExcel.set_name("Sequential One vs Rest classifier")
        test_stop_criterion(X, y, sequential_naive_bayes_classifier,
                        None, reorder_using_information_gain,
                        criterion_optimizer, criterion_optimizer2,
                        costs,
                        folds_number=CROSS_VALIDATION_FOLDS_NUMBER,
                        save_data=saveToExcel)

def execute_multiclasses_sequential(X,y,costs,saveToExcel):
    print("Naive Bayes classifier scores (10-fold cross validation):")
    sequential_naive_bayes = SequentialNaiveBayes()

    criterion_optimizer = partial(
        seq_classifier.optimize_stop_criterion_with_divide,
        target_classifier=MultinomialNB(),
        classifier_class=SequentialNaiveBayes,
        start=0.5,
        finish=1,
        step=0.01,
        feature_reorder=reorder_using_information_gain,
        )

    print(
        "Sequential MultiClass classifier")

    saveToExcel.set_name("Sequential MultiClass")
    test_classifier(X, y, sequential_naive_bayes,
                    None, reorder_using_information_gain,
                    criterion_optimizer,
                    costs,
                    folds_number=CROSS_VALIDATION_FOLDS_NUMBER,
                    save_data=saveToExcel)




def execute_feature_selection_comparation(X,y,costs,saveToExcel):
    naive_bayes_classifier = MultinomialNB()

    saveToExcel.set_name("Best First Naive ")
    best_first_selection = partial(best_first_feature_selection)
    print(
        "Own Naive Bayes classifier with best-first search wrapper scores (10-fold cross validation):")
    test_classifier(X, y, naive_bayes_classifier,
                    best_first_selection, None, None, costs,
                    folds_number=CROSS_VALIDATION_FOLDS_NUMBER,
                    save_data=saveToExcel)

    saveToExcel.set_name("Greedy Cost Sensitive")
    greedy_cost_sensitive_selection = partial(
        greedy_cost_sensitive_feature_selection, costs=costs)
    print(
        "Own Naive Bayes classifier with greedy cost sensitive wrapper scores (10-fold cross validation):")
    test_classifier(X, y, naive_bayes_classifier,
                    greedy_cost_sensitive_selection, None,
                    None, costs,
                    folds_number=CROSS_VALIDATION_FOLDS_NUMBER,
                    save_data=saveToExcel)

    saveToExcel.set_name("Best First Cost Sensitive")
    best_first_cost_sensitive_selection = partial(
        best_first_cost_sensitive_feature_selection, costs=costs)
    print(
        "Own Naive Bayes classifier with best-first cost sensitive wrapper scores (10-fold cross validation):")
    test_classifier(X, y, naive_bayes_classifier,
                    best_first_cost_sensitive_selection, None,
                    None, costs,
                    folds_number=CROSS_VALIDATION_FOLDS_NUMBER,
                    save_data=saveToExcel)

    print(
        "Own Sequential Naive Bayes classifier scores (10-fold cross validation):")

def execute_onevsrest_sequential_decrease(X,y,costs,saveToExcel):
    criterion_optimizer = partial(
        optimize_sequential_classification_stop_criterion,
        target_classifier=MultinomialNB(),
        classifier_class=seq_classifier.SequentialNaiveBayesOneVsRest,
        criterion_range=np.arange(1.0, 0.5-0.01, -0.01),
        feature_reorder=reorder_using_information_gain)


    print(
        "Sequential One vs Rest classifier")
    sequential_naive_bayes_classifier = seq_classifier.SequentialNaiveBayesOneVsRest()

    saveToExcel.set_name("Sequential One vs Rest classifier - decreasing")
    test_classifier(X, y, sequential_naive_bayes_classifier,
                    None, reorder_using_information_gain,
                    criterion_optimizer,
                    costs,
                    folds_number=CROSS_VALIDATION_FOLDS_NUMBER,
                    save_data=saveToExcel)

def execute_onevsrest_sequential_divide(X,y,costs,saveToExcel):
    criterion_optimizer = partial(
        seq_classifier.optimize_stop_criterion_with_divide,
        target_classifier=MultinomialNB(),
        classifier_class=seq_classifier.SequentialNaiveBayesOneVsRest,
        start=0.5,
        finish=1,
        step=0.01,
        feature_reorder=reorder_using_information_gain,
        )

    print(
        "Sequential One vs Rest classifier")
    sequential_naive_bayes_classifier = seq_classifier.SequentialNaiveBayesOneVsRest()

    saveToExcel.set_name("Sequential One vs Rest classifier")
    test_classifier(X, y, sequential_naive_bayes_classifier,
                    None, reorder_using_information_gain,
                    criterion_optimizer,
                    costs,
                    folds_number=CROSS_VALIDATION_FOLDS_NUMBER,
                    save_data=saveToExcel)




def execute_Naive_Bayes(X,y,costs,saveToExcel):
    print("Naive Bayes classifier scores (10-fold cross validation):")
    saveToExcel.set_name("Naive Bayes")

    naive_bayes_classifier = MultinomialNB()
    test_classifier(X, y, naive_bayes_classifier, None, None,
                                  None,
                                  costs,
                                  folds_number=CROSS_VALIDATION_FOLDS_NUMBER,
                                  save_data=saveToExcel)


if __name__ == "__main__":
    main()