from dataclasses import dataclass, field
import pandas as pd
from abc import abstractmethod
from typing import List
from sklearn.metrics import roc_curve
import numpy as np

@dataclass
class ThresholdFinderFactory:
    accepted_method: List[str] = field(default_factory=lambda: ["youden_statistic"])
        
    def get_finder(self, method: str):
        if method not in self.accepted_method:
            raise ValueError(f"'method' should be either {self.accepted_method}, found {method} ")

        if method == "youden_statistic":
            return YoudenThresholdFinder()

@dataclass
class OptimalThresholdFinder:

    @abstractmethod
    def optimal_threshold(self, true_label: pd.Series, predicted_proba: pd.Series) -> float:
        pass
    
@dataclass
class YoudenThresholdFinder(OptimalThresholdFinder):

    def __youden_statistic(self, true_label: pd.Series, predicted_proba: pd.Series) -> float:
        fpr, tpr, thresholds = roc_curve(true_label, predicted_proba)
        optimal_proba_cutoff = sorted(list(zip(np.abs(true_pos_rate - false_pos_rate), thresholds)), 
                            key=lambda i: i[0], reverse=True)[0][1]
        print(optimal_proba_cutoff)
        return optimal_proba_cutoff

    def optimal_threshold(self, true_label: pd.Series, predicted_proba: pd.Series) -> float:
        return self.__youden_statistic(true_label, predicted_proba)