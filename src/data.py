from typing import Tuple
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

COLUMNS = {
    "age": int,
    "workclass": "category",
    "fnlwgt": int,
    "education": "category",
    "education_num": int,
    "marital-status": "category",
    "occupation": "category",
    "relationship": "category",
    "race": "category",
    "sex": "category",
    "capital-gain": int,
    "capital-loss": int,
    "hours-per-week": int,
    "native-country": "category",
    "target": "category",
}


def load_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    # ' ?'/'?' is a missing value marker
    # We drop education because it is represented by education_num
    # We drop fnlwgt as it is an indicator which should not be used while fitting the model
    df = pd.read_csv(
        "../data/raw/adult/adult.data",
        header=None,
        names=COLUMNS.keys(),
        index_col=False,
        na_values=[" ?", "?"],
        dtype=COLUMNS,
    ).drop(columns=["education", "fnlwgt"])

    X, y = df.loc[:, df.columns != "target"], df["target"]
    X = impute_missing_data(X)

    return X, y


def impute_missing_data(X: pd.DataFrame):
    cols_with_na = X.columns[X.isna().any()].tolist()
    filler = DecisionTreeClassifier(max_depth=5)
    unknown = X.loc[:, X.columns.isin(cols_with_na)].copy()

    # One hot encoded
    known = pd.get_dummies(X.loc[:, ~X.columns.isin(cols_with_na)], drop_first=True)

    for col in cols_with_na:
        y_known = unknown[col][~unknown[col].isnull()]
        X_known, X_nan = (
            known.loc[~unknown[col].isnull()],
            known.loc[unknown[col].isnull()],
        )
        filler.fit(X_known, y_known)
        unknown.loc[unknown[col].isnull(), col] = filler.predict(X_nan)

    X_filled = pd.concat([known, unknown], axis=1)
    X_filled = pd.get_dummies(X_filled)

    return X_filled
