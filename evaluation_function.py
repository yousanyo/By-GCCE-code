from torcheval.metrics import MulticlassAccuracy
from torchmetrics.classification import MulticlassPrecision
from torchmetrics.classification import MulticlassRecall
from torchmetrics.classification import MulticlassF1Score

class Evaluation_Function:
    def __init__(self, ap_number, prediction, true):
        self.ap_number = ap_number
        self.prediction = prediction
        self.true = true
    
    def accuracy(self):
        accuracy_metric = MulticlassAccuracy()
        accuracy_metric.update(self.prediction, self.true)
        print(f'|---- Test Accuracy: {100 * accuracy_metric.compute():.2f}%')
    
    def precision(self):
        precision_metric = MulticlassPrecision(self.ap_number)
        precision = precision_metric(self.prediction, self.true)
        print(f'|---- Precision: {100 * precision:.2f}%')
    
    def recall(self):
        recall_metric = MulticlassRecall(self.ap_number)
        recall = recall_metric(self.prediction, self.true)
        print(f'|---- Recall: {100 * recall:.2f}%')
    
    def f1_score(self):
        f1_score_metric = MulticlassF1Score(self.ap_number)
        f1_score = f1_score_metric(self.prediction, self.true)
        print(f'|---- F1 Score: {100 * f1_score:.2f}%')