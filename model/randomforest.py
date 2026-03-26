import numpy as np
import pandas as pd
import warnings
from model.base import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from Config import Config
from numpy import *
import random

# Suppress sklearn single-label confusion matrix shape warning (expected in narrow branches)
warnings.filterwarnings(
    "ignore",
    message="A single label was found in 'y_true' and 'y_pred'",
    category=UserWarning,
    module="sklearn"
)
num_folds = 0
seed =0
# Data
np.random.seed(seed)
random.seed(seed)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 200)


class RandomForest(BaseModel):
    def __init__(self,
                 model_name: str,
                 embeddings: np.ndarray,
                 y: np.ndarray) -> None:
        super(RandomForest, self).__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        self.mdl = RandomForestClassifier(
            n_estimators=100,          # Reduced from 1000 to prevent overfitting
            max_depth=10,              # Limit tree depth to prevent overfitting
            min_samples_leaf=2,        # Require >=2 samples at each leaf
            min_samples_split=4,       # Require >=4 samples to split a node
            random_state=seed,
            class_weight='balanced'    # Use 'balanced' (not balanced_subsample) for consistent weighting
        )
        self.predictions = None
        self.data_transform()

    def train(self, data) -> None:
        self.mdl = self.mdl.fit(data.X_train, data.y_train)

    def predict(self, X_test):
        predictions = self.mdl.predict(X_test)
        self.predictions = predictions
        return predictions

    def print_results(self, data):
        n_test = len(data.y_test)
        # previous behavior; print all metrics regardless of test count
        acc = accuracy_score(data.y_test, self.predictions)
        print(f"  Accuracy: {acc:.4f}")
        print(classification_report(data.y_test, self.predictions, zero_division=0))
        try:
            labels = sorted(set(data.y_test) | set(self.predictions))
            cm = confusion_matrix(data.y_test, self.predictions, labels=labels)
            print(f"  Confusion Matrix (rows=true, cols=pred):")
            print(f"  Labels: {labels}")
            print(f"  {cm}")
        except Exception:
            pass


    def data_transform(self) -> None:
        ...

