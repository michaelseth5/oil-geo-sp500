import unittest
import numpy as np
import pandas as pd
from src.research_validation import (selected_metrics,assert_research_frame,feature_drift,publication_audit,feature_columns,evaluate_walk_forward)
class ResearchChecks(unittest.TestCase):
 def test_threshold_coverage(self):
  r=selected_metrics([1,0,1,0],[0.9,0.1,0.51,0.49],0.66)
  self.assertEqual(r["selected"],2)
  self.assertEqual(r["coverage"],0.5)
 def test_threshold_accuracy(self):
  self.assertEqual(selected_metrics([1,0],[.99,.01])["accuracy_selected"],1.0)
 def test_empty_coverage(self):
  self.assertIsNone(selected_metrics([0,1],[.51,.49],.90)["accuracy_selected"])
 def test_threshold_bounds(self):
  with self.assertRaises(ValueError):selected_metrics([0],[.5],.49)
 def test_reject_invalid_probability(self):
  with self.assertRaises(ValueError):selected_metrics([0],[1.5])
 def test_feature_targets_excluded(self):
  df=pd.DataFrame({"x":[2],"target":[1],"next_monday_return":[0.1]})
  self.assertEqual(feature_columns(df),["x"])
 def test_unsorted_dates(self):
  df=pd.DataFrame({"target":[0]*100,"x":[1]*100},index=pd.date_range("2025-01-01",periods=100)[::-1])
  with self.assertRaises(ValueError):assert_research_frame(df)
 def test_missing_feature_values(self):
  df=pd.DataFrame({"target":[0]*100,"x":[1.]*99+[float("nan")]},index=pd.date_range("2025-01-01",periods=100))
  with self.assertRaises(ValueError):assert_research_frame(df)
 def test_news_cutoff(self):
  df=pd.DataFrame({"published_at":["2025-01-01T10:00:00Z","2025-01-03T10:00:00Z"],"prediction_cutoff":["2025-01-02T00:00:00Z"]*2})
  self.assertEqual(publication_audit(df)["blocked"],1)
 def test_missing_news_timestamps(self):
  with self.assertRaises(ValueError):publication_audit(pd.DataFrame({"date":["2025-01-01"]}))
 def test_drift_identical(self):
  x=pd.DataFrame({"x":np.arange(100,dtype=float)})
  self.assertLess(feature_drift(x,x,["x"])[0]["psi"],0.001)
 def test_drift_shift(self):
  x=pd.DataFrame({"x":np.arange(100,dtype=float)})
  y=pd.DataFrame({"x":np.arange(100,200,dtype=float)})
  self.assertTrue(feature_drift(x,y,["x"])[0]["flag"])
 def test_walk_forward_non_overlap(self):
  idx=pd.date_range("2010-01-04",periods=125,freq="W-MON")
  df=pd.DataFrame({"signal":np.sin(np.arange(125)),"other":np.cos(np.arange(125)),
                   "target":(np.arange(125)%2).astype(int)},index=idx)
  r=evaluate_walk_forward(df,initial_train=80,test_window=15)
  self.assertEqual(len(r["predictions"]),44)
  self.assertLess(r["folds"][0]["train_end"],r["folds"][0]["test_start"])
if __name__=="__main__":unittest.main()
