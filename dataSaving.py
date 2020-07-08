from xlwt import Workbook
import numpy
import os

class SaveToExcel:
    def __init__(self,name):
        self._name = name
        self._wb = Workbook()
        self.accuraciesSheet = self._wb.add_sheet('Accuracies')
        self.meanAccuracySheet = self._wb.add_sheet('Mean Accuracy')
        self.costSheet = self._wb.add_sheet('Costs')
        self.meanCostSheet = self._wb.add_sheet('Mean Cost')
        self.criterionValuesSheet = self._wb.add_sheet('Stop Criterion Values')
        self.row = 0

    def add_data(self,accuracies,mean_accuracy,costs,mean_cost,stop_criterion_values=numpy.array([])):
        self._add_table(self.accuraciesSheet, accuracies)
        self._add_value(self.meanAccuracySheet, mean_accuracy)
        self._add_table(self.costSheet, costs)
        self._add_value(self.meanCostSheet, mean_cost)
        self._add_table(self.criterionValuesSheet, stop_criterion_values)
        self.row += 1

    # save_data.add_data(folds_scores, mean_accuracy, folds_costs, mean_cost, stop_criterion_values)
    def _add_value(self, sheet, value):
        sheet.write(self.row,0,self.name)
        sheet.write(self.row,1,value)

    def _add_table(self,sheet,table):
        sheet.write(self.row, 0, self.name)
        it = 1
        for c in table:
            sheet.write(self.row, it, c)
            it += 1


    def set_name(self, name):
        self.name = name

    def close(self):
        filename = self._name + '.xls'
        self._wb.save(os.path.join('./results', filename))
