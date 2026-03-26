import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Config import *
import random
seed =0
random.seed(seed)
np.random.seed(seed)

class Data():
    def __init__(self,
                 X: np.ndarray,
                 df: pd.DataFrame,
                 target_col: str = Config.TYPE2) -> None:

        y = df[target_col].to_numpy()
        y_series = pd.Series(y)

        good_y_value = y_series.value_counts()[y_series.value_counts() >= Config.MIN_CLASS_COUNT].index

        if len(good_y_value)<1:
            print(f"None of the class have more than {Config.MIN_CLASS_COUNT} records for {target_col}: Skipping ...")
            self.X_train = None
            return

        good_mask = y_series.isin(good_y_value).values
        y_good = y[good_mask]
        X_good = X[good_mask]
        # Use boolean array mask on df directly to keep original labels
        df_good = df[good_mask]

        if X_good.shape[0] == 0:
            self.X_train = None
            return

        new_test_size = X.shape[0] * 0.2 / X_good.shape[0]
        if new_test_size >= 1.0 or new_test_size <= 0.0:
            new_test_size = 0.2

        self.X_train, self.X_test, self.y_train, self.y_test, self.train_df, self.test_df = train_test_split(
            X_good, y_good, df_good, test_size=new_test_size, random_state=Config.RANDOM_STATE, stratify=y_good)

        # do not skip on small test sets (previous behavior)
        self.y = y_good
        self.classes = good_y_value
        self.embeddings = X


    def get_type(self):
        return  self.y
    def get_X_train(self):
        return  self.X_train
    def get_X_test(self):
        return  self.X_test
    def get_type_y_train(self):
        return  self.y_train
    def get_type_y_test(self):
        return  self.y_test
    def get_train_df(self):
        return  self.train_df
    def get_embeddings(self):
        return  self.embeddings
    def get_type_test_df(self):
        return  self.test_df
    def get_X_DL_test(self):
        return self.X_DL_test
    def get_X_DL_train(self):
        return self.X_DL_train

