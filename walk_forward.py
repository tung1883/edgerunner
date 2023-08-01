
import pandas as pd
import numpy as np

def walk_forward(df, model_fn, n_splits=5):
    n = len(df)
    fold = n // n_splits
    results = []
    for i in range(1, n_splits):
        train = df.iloc[:i * fold]
        test  = df.iloc[i * fold:(i+1) * fold]
        model = model_fn(train)
        results.append(model(test))
    return results

