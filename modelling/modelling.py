from model.randomforest import RandomForest
from modelling.data_model import Data
from Config import Config
import pandas as pd
import numpy as np

def run_chained_multi_outputs(X, df, name):
    print(f"\n{'='*60}")
    print(f"DESIGN CHOICE 1: Chained Multi-outputs — {name}")
    print(f"{'='*60}")
    chain_targets = [
        ('y_chain_1', 'Type 2 only'),
        ('y_chain_2', 'Type 2 + Type 3'),
        ('y_chain_3', 'Type 2 + Type 3 + Type 4'),
    ]
    for level, (target, desc) in enumerate(chain_targets, 1):
        print(f"\n  -- Chain Level {level}: {desc} --")
        data = Data(X, df, target_col=target)
        if data.X_train is None:
            print(f"  [SKIPPED] {target} — insufficient class sizes (< {Config.MIN_CLASS_COUNT}).")
            continue

        model = RandomForest(f"RF_Chain_{level}", data.get_embeddings(), data.get_type())
        model.train(data)
        predictions = model.predict(data.X_test)
        model.print_results(data)

        # Write predictions back to the original df using the test set index
        df.loc[data.test_df.index, f'pred_{target}'] = predictions


def run_hierarchical_modeling(X, df, name):
    print(f"\n{'='*60}")
    print(f"DESIGN CHOICE 2: Hierarchical Modelling — {name}")
    print(f"{'='*60}")

    # Reset index so df positional index matches X numpy rows
    df = df.reset_index(drop=True)

    # ── Level 1: classify Type 2 ──────────────────────────────────
    print(f"\n  -- Level 1: predicting {Config.TYPE2} --")
    data_l1 = Data(X, df, target_col=Config.TYPE2)
    if data_l1.X_train is None:
        print("  [SKIPPED] Level 1 — insufficient data.")
        return

    model_l1 = RandomForest("RF_Hier_L1", data_l1.get_embeddings(), data_l1.get_type())
    model_l1.train(data_l1)
    preds_l1 = model_l1.predict(data_l1.X_test)
    print("  Level 1 results:")
    model_l1.print_results(data_l1)
    df.loc[data_l1.test_df.index, f'pred_hier_{Config.TYPE2}'] = preds_l1

    # ── Level 2: for each Type 2 class, classify Type 3 ──────────
    for c2 in data_l1.classes:
        mask2 = (df[Config.TYPE2] == c2).values          # numpy bool array
        if mask2.sum() < Config.MIN_CLASS_COUNT:
            continue

        df_sub2 = df[mask2].reset_index(drop=True)
        X_sub2  = X[mask2]

        print(f"\n  -- Level 2 Branch [{Config.TYPE2}={c2}]: predicting {Config.TYPE3} --")
        data_l2 = Data(X_sub2, df_sub2, target_col=Config.TYPE3)
        if data_l2.X_train is None:
            print(f"  [SKIPPED] — insufficient {Config.TYPE3} classes in branch '{c2}'.")
            continue

        model_l2 = RandomForest(f"RF_Hier_L2_{c2}", data_l2.get_embeddings(), data_l2.get_type())
        model_l2.train(data_l2)
        preds_l2 = model_l2.predict(data_l2.X_test)
        print(f"  Level 2 results for '{c2}':")
        model_l2.print_results(data_l2)

        # write back using original df index via test_df
        df.loc[data_l2.test_df.index, f'pred_hier_{Config.TYPE3}'] = preds_l2

        # ── Level 3: for each Type 3 class, classify Type 4 ──────
        for c3 in data_l2.classes:
            mask3 = (df_sub2[Config.TYPE3] == c3).values   # numpy bool matching X_sub2
            if mask3.sum() < Config.MIN_CLASS_COUNT:
                continue

            df_sub3 = df_sub2[mask3].reset_index(drop=True)
            X_sub3  = X_sub2[mask3]

            print(f"\n  -- Level 3 Branch [{Config.TYPE3}={c3}]: predicting {Config.TYPE4} --")
            data_l3 = Data(X_sub3, df_sub3, target_col=Config.TYPE4)
            if data_l3.X_train is None:
                print(f"  [SKIPPED] — insufficient {Config.TYPE4} classes in branch '{c3}'.")
                continue

            model_l3 = RandomForest(f"RF_Hier_L3_{c3}", data_l3.get_embeddings(), data_l3.get_type())
            model_l3.train(data_l3)
            preds_l3 = model_l3.predict(data_l3.X_test)
            print(f"  Level 3 results for '{c3}':")
            model_l3.print_results(data_l3)
            df.loc[data_l3.test_df.index, f'pred_hier_{Config.TYPE4}'] = preds_l3