import pandas as pd
import pytest

from threshold_finder import __version__
from threshold_finder.finder import OptimalThresholdFinder, ThresholdFinder, YoudenThresholdFinder

class TestThresholdFinder:
    def test_get_finder(self):
        finder = ThresholdFinder().get_finder(method="youden_statistic")

        assert isinstance(finder, YoudenThresholdFinder)
        assert isinstance(finder, OptimalThresholdFinder)
    
    def test_wrong_method(self):
        with pytest.raises(ValueError):
            finder = ThresholdFinder().get_finder(method="wrong_method")

class TestYoudenThresholdFinder:
    def setup_method(self):
        self.true_label = pd.Series([1,1,1,0,0,0])
        self.predicted_proba = pd.Series([0.9, 0.8, 0.7, 0.72, 0.6, 0.5])
    
    def test_find_threshold(self):
        finder = YoudenThresholdFinder()

        threshold = finder.optimal_threshold(self.true_label, self.predicted_proba)

        assert isinstance(threshold, float)
        assert threshold == 0.70


