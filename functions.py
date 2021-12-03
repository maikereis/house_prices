import re
import numpy as np
import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt

from pandas.api.types import CategoricalDtype
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from xgboost import XGBRegressor
from transformers import BuildCluster


def plot_relationship(data, title):
    width = 8
    if len(data) > 70:
        width = 12
    fig, ax = plt.subplots(figsize=(width, 4))
    g = sns.barplot(x=data.index, y=data, ax=ax)
    ax.tick_params(axis="x", rotation=-90)
    ax.set_ylim(1.2 * data.min(), 1.2 * data.max())
    g.set_title(title)
    plt.tight_layout()


def plot_variance(pca, width=8, dpi=100):
    # Create figure
    fig, axs = plt.subplots(1, 2)
    n = pca.n_components_
    grid = np.arange(1, n + 1)
    # Explained variance
    evr = pca.explained_variance_ratio_
    axs[0].bar(grid, evr)
    axs[0].set(xlabel="Component", title="% Explained Variance", ylim=(0.0, 1.0))
    # Cumulative Variance
    cv = np.cumsum(evr)
    axs[1].plot(np.r_[0, grid], np.r_[0, cv], "o-")
    axs[1].set(xlabel="Component", title="% Cumulative Variance", ylim=(0.0, 1.0))
    # Set up figure
    fig.set(figwidth=8, dpi=100)
    return axs


def corrplot(df, method="pearson", annot=True, **kwargs):
    sns.clustermap(
        df.corr(method),
        vmin=-1.0,
        vmax=1.0,
        cmap="icefire",
        method="complete",
        annot=annot,
        **kwargs,
    )


def get_missing(df, percent=False):
    if percent:
        missing = (df.isnull().sum() * 100 / df.isnull().count()).sort_values(
            ascending=False
        )
    else:
        missing = df.isnull().sum()

    missing = missing[missing > 0].sort_values(ascending=False)
    return missing


def get_rsquares(X, y):

    X = X.copy()

    model = LinearRegression()

    for col in X.select_dtypes("object"):
        X[col], _ = X[col].factorize()

    rsquares = {}
    for col in X.columns:
        X_, y_ = pd.get_dummies(X[col]), y
        model.fit(X_, y_)
        r2 = model.score(X_, y_)
        rsquares[col] = r2
    rsquares = pd.Series(rsquares)
    return rsquares


def make_mi_scores(X, y):
    X = X.copy()

    if isinstance(X, pd.DataFrame):
        for colname in X.select_dtypes("object"):
            X[colname], _ = X[colname].factorize()
        for colname in X.select_dtypes("category"):
            X[colname] = X[colname].cat.codes

    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(
        X, y, discrete_features=discrete_features, random_state=0
    )
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def custom_boxplot(X, y, **kwargs):
    sns.boxplot(x=X, y=y)
    x = plt.xticks(rotation=90)


def get_low_cardinality(df, threshold):
    low_card = {
        col: len(df[col].unique()) for col in df if (len(df[col].unique())) <= threshold
    }
    return low_card


def get_high_cardinality(df, threshold):
    low_card = {
        col: len(df[col].unique()) for col in df if (len(df[col].unique())) > threshold
    }
    return low_card


def score_dataset(X, y, model=XGBRegressor()):
    X = X.copy()
    if isinstance(X, pd.DataFrame):
        for colname in X.select_dtypes(["category"]):
            X[colname] = X[colname].cat.codes
    # Metric for Housing competition is RMSLE (Root Mean Squared Log Error)
    log_y = np.log(y)
    score = cross_val_score(model, X, log_y, cv=5, scoring="neg_mean_squared_error")
    score = -1 * score.mean()
    score = np.sqrt(score)
    return score


def drop_uninformative(df, mi_scores):
    return df.loc[:, mi_scores > 0.0]


def generate_submission_file(X_test, y_pred, score):
    # Concatenate the predictions in a dataset
    output = pd.DataFrame(y_pred, index=X_test.index, columns=["SalePrice"])

    output["Id"] = output.index

    # Generate a file name with datetime info
    file_name = datetime.datetime.today().strftime(
        f"%Y_%m_%d__%H_%M_%S_with_score__{np.round(score,6)}"
    )
    # Generate the submission file
    output.to_csv(f"output/{file_name}__.csv", index=False)


def plot_interactions(data, X, hue, y="SalePrice"):
    sns.lmplot(
        data=data,
        x=X,
        y=y,
        hue=hue,
        col=hue,
        scatter_kws={"edgecolor": "w"},
        col_wrap=3,
        height=2,
    )


def find_best_k(df, targets, cols, trials):
    best_score = 100
    for k in range(1, trials):
        add_cluster = BuildCluster(cols, k, 50, name="Cluster1")
        df_with_cluster = add_cluster.transform(df)
        score = score_dataset(df_with_cluster, targets)
        if score < best_score:
            best_score = score
            print(">", end="")
        print(f"{k}, {score}")
