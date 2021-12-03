import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder


class FillWith:

    """
    Fill a list of cols with a specified value

    >>> df = {'col1': [1, 2, NaN], 'col2': [4, 5, 6], 'col3': [7, NaN, 10]}
    >>> filler = FillWith(['col1', 'col2'], -1)
    >>> df_filled = filler.transform(df)

    >>> df_filled
       col1  col2 col3
    0     1     4    7
    1     2     5   -1
    2    -1     6   10

    """

    def __init__(self, cols=[], value="None"):
        self.cols = cols
        self.value = value

    def __repr__(self):
        return "FillWith(cols=%s, value=%s)" % (self.cols, self.value)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.X = X.copy()
        self.X[self.cols] = self.X[self.cols].fillna(self.value)
        return self.X


class FillByKNN:
    def __init__(self, n_neighbors=5, targets=None, keyword=None):
        self.n_neighbors = n_neighbors
        self.targets = targets
        self.keyword = keyword
        self.cols = None

    def fit(self, X, y=None):
        return self

    def __repr__(self):
        return "FillByKNN(n_neighbors=%i, targets=%s, keyword=%s,cols=%s)" % (
            self.n_neighbors,
            self.targets,
            self.keyword,
            self.cols,
        )

    def transform(self, X, y=None):

        self.X = X.copy()

        # If 'keyword' is none, use all the columns to estimate the KNN
        # else, use just columns that match the 'keyword'
        if self.keyword is None:
            self.cols_ = [col for col in self.X.columns]
        else:
            self.cols_ = [col for col in self.X.columns if self.keyword in col]

        # If 'targets' is none, use select all columns with missing value to fill
        if self.targets is None:
            self.targets_ = list(self.X.columns[self.X.isnull().any()])
            if self.keyword is not None:
                self.cols_ = list(set(self.targets + self.cols))
        else:
            self.targets_ = self.targets

        self.mapper = {}
        # Find each categorical column passed as arguments 'cols' and convert to numeric
        for col in self.cols_:
            if X[col].dtypes == "object" or X[col].dtypes == "category":
                # replace 'object' columns to 'numeric' versions and store nominal value in mapper
                # to reverse the transformation.
                self.X.loc[:, col], self.mapper[col] = self.X.loc[:, col].factorize()

        # Input missing values with KNNImputer
        self.imputer = KNNImputer(n_neighbors=self.n_neighbors)
        self.result = pd.DataFrame(
            self.imputer.fit_transform(self.X[self.cols_]),
            columns=self.cols_,
            index=self.X.index,
        )

        # Update targets with imputed data
        self.X.loc[:, self.targets_] = self.result[self.targets_]

        # Reverse to categorical values
        for col, unique in self.mapper.items():
            try:
                self.X.loc[:, col] = unique[self.X[col].astype(int)]
            except Exception as e:
                print(e)
                pass

        return self.X


class FillByGroup:

    """
    Group columns and fill based on a reference col.

    >>> df = {'HasCar': [NaN, 1, 1, NaN], 'CarMotor': [NaN, 2, 3, 4], 'CarWheelSize':[NaN, NaN, Large, NaN] 'Bike': [5, 6, 7, 8]}
    >>> filler = FillByGroup(group_ref = 'HasCar', group_keyword = 'Car', num_value= 0, cat_values = 'None')
    >>> df_filled = filler.transform(df)

    >>> df_filled
       HasCar  CarMotor CarWheelSize Bike
    0       0         0         None    5
    1       1         2          NaN    6
    2       1         3        Large    7
    3       0         4         None    8

    """

    def __init__(
        self, group_ref=None, group_keyword=None, num_value=0, cat_value="None"
    ):

        self.group_ref = group_ref
        self.group_keyword = group_keyword
        self.num_value = num_value
        self.cat_value = cat_value

    def __repr__(self):
        return (
            "FillByGroup(group_ref=%s, group_keyword=%s, num_value=%i, cat_value=%s)"
            % (self.group_ref, self.group_keyword, self.num_value, self.cat_value)
        )

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        self.X = X.copy()
        self.cols = [col for col in self.X.columns if self.group_keyword in col]
        self.null_mask = pd.isnull(self.X[self.group_ref])

        # split 'object' (str) and 'numeric' (int, float) columns
        self.num_cols = list(X[self.cols].select_dtypes("number").columns)
        self.object_cols = list(X[self.cols].select_dtypes("category").columns) + list(
            X[self.cols].select_dtypes("object").columns
        )

        self.X.loc[self.null_mask, self.num_cols] = self.X.loc[
            self.null_mask, self.num_cols
        ].fillna(self.num_value)
        self.X.loc[self.null_mask, self.object_cols] = self.X.loc[
            self.null_mask, self.object_cols
        ].fillna(self.cat_value)

        return self.X


class BasicFill:

    """

    Just fill all data with 'None' to categorical, and 0 to numeric

    >>> df = {'col1': [1, 2, NaN], 'col2': ['A', 'B', 'NaN'], 'col3': [NaN, 5, 6]}
    >>> filler = BasicFill(object_cols = ['col2'], numeric_cols = ['col1', 'col3'], num_value = -99, obj_value = 'None')
    >>> df_filled = filler.transform(df)

    >>> df_filled
       col1    col2  col3
    0     1     'A'   -99
    1     2     'B'     5
    2   -99  'None'     6

    """

    def __init__(self, object_cols, numeric_cols, obj_value="None", num_value=0):
        self.object_cols = object_cols
        self.numeric_cols = numeric_cols
        self.obj_value = obj_value
        self.num_value = num_value

    def __repr__(self):
        return "BasicFill()"

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        self._X = X.copy()
        self._X[self.object_cols] = self._X[self.object_cols].fillna(self.obj_value)
        self._X[self.numeric_cols] = self._X[self.numeric_cols].fillna(self.num_value)
        return self._X


class FixTypos:
    def __init__(self):
        return None

    def __repr__(self):
        return "FixTypos()"

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        self.X = X.copy()
        self.X.Exterior2nd = self.X.Exterior2nd.replace("Brk Cmn", "BrkComm")
        self.X.Exterior2nd = self.X.Exterior2nd.replace("CmentBd", "CemntBd")
        self.X.Exterior2nd = self.X.Exterior2nd.replace("Wd Shng", "Wd Sdng")
        self.X.BldgType = self.X.BldgType.replace("Twnhs", "TwnhsI")
        self.X.BldgType = self.X.BldgType.replace("2fmCon", "2FmCon")
        self.X.BldgType = self.X.BldgType.replace("Duplex", "Duplx")
        self.X.rename(
            columns={
                "1stFlrSF": "FirstFlrSF",
                "2ndFlrSF": "SecondFlrSF",
                "3SsnPorch": "Threeseasonporch",
            },
            inplace=True,
        )

        return self.X


class CategorizeNominal:
    def __init__(self, ordered_dict, unordered_dict):
        self.ordered_dict = ordered_dict
        self.unordered_dict = unordered_dict

    def __repr__(self):
        return "CategorizeNominal()"

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        self.X = X.copy()

        for col, levels in self.unordered_dict.items():
            self.X[col] = self.X[col].astype(CategoricalDtype(levels))
        for col, levels in self.ordered_dict.items():
            self.X[col] = self.X[col].astype(CategoricalDtype(levels, ordered=True))

        return self.X


class DropCols:
    def __init__(self, cols):
        self._cols = list(cols)

    def __repr__(self):
        return "BuildPCA(cols=%s)" % (self._cols)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        self.X = X.copy()

        if all(col in self.X.columns for col in self._cols):
            self.X = self.X.drop(self._cols, axis=1)

        return self.X


class FloatToInt:
    def __init__(self, cols_to_drop):
        self.cols_to_drop = cols_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        self.X = X.copy()
        for col in self.X.select_dtypes("category"):
            self.X[col] = self.X[col].astype(int)
        return self.X


class EncodeCategories:
    def __init__(self):
        return None

    def __repr__(self):
        return "EncodeCategories()"

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        self.X = X.copy()
        for colname in self.X.select_dtypes(["category"]):
            self.X[colname] = self.X[colname].cat.codes
        return self.X


class ConvertToInt:
    def __init__(self):
        return None

    def __repr__(self):
        return "ConvertToInt()"

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        self.X = X.copy()
        for col in self.X.select_dtypes(["float"]):
            self.X[col] = self.X[col].astype(int)
        return self.X


class BuildMathFeatures:
    def __init__(self):
        return None

    def __repr__(self):
        return "BuildMathFeatures()"

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        self.X = X.copy()
        self.X["TotalFlrSF"] = self.X.FirstFlrSF + self.X.SecondFlrSF
        return self.X


class BuildInteractionsFeatures:
    def __init__(self):
        self.enc = OneHotEncoder(sparse=False, handle_unknown="ignore")
        return None

    def __repr__(self):
        return "BuildInteractionsFeatures()"

    def fit(self, X, y=None):
        self.enc.fit(X[["BedroomAbvGr"]])
        return self

    def transform(self, X, y=None):
        self.X = X.copy()

        X1 = self.enc.transform(self.X[["BedroomAbvGr"]])
        X1 = pd.DataFrame(
            X1, columns=list(self.enc.get_feature_names_out()), index=self.X.index
        )
        X1 = X1.mul(self.X.GrLivArea, axis=0)
        self.X = pd.concat([self.X, X1], axis=1)
        return self.X


class BuildCountsFeatures:
    def __init__(self):
        return None

    def __repr__(self):
        return "BuildCountsFeatures()"

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        self.X = X.copy()
        self.X["Rms_Ktchn_Bath"] = (
            self.X["TotRmsAbvGrd"] + self.X["KitchenAbvGr"] + self.X["FullBath"]
        )
        return self.X


class BuildGroupFeatures:
    def __init__(self):
        return None

    def __repr__(self):
        return "BuildGroupFeatures()"

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        self.X = X.copy()
        self.X["Neigh_Area_mean"] = self.X.groupby("Neighborhood")["LotArea"].transform(
            "mean"
        )
        self.X["BedRm_LvArea_mean"] = self.X.groupby("BedroomAbvGr")[
            "GrLivArea"
        ].transform("mean")
        return self.X


class BuildCluster:
    def __init__(self, cols, n_clusters, n_init=50, name="Cluster"):
        self.cols = cols
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.name = name

    def __repr__(self):
        return "BuildGroupFeatures(cols=%s, n_clusters=%s, n_init=%i, name=%s)" % (
            self.cols,
            self.n_clusters,
            self.n_init,
            self.name,
        )

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        self.X = X.copy()

        self.X_cols = self.X[self.cols]
        self.X_scaled = (self.X_cols - self.X_cols.mean(axis=0)) / self.X_cols.std(
            axis=0
        )
        self.kmeans = KMeans(
            n_clusters=self.n_clusters, n_init=self.n_init, random_state=0
        )
        self.X[self.name] = self.kmeans.fit_predict(self.X_scaled)
        return self.X


class BuildPCA:
    def __init__(self, cols, drop_originals=False, use_n_components=None):
        self._cols = cols
        self._pca = None
        self._X_pca = None
        self._loadings = None
        self._drop_originals = drop_originals
        self._use_n_components = use_n_components

    def __repr__(self):
        return "BuildPCA(cols=%s)" % (self._cols)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        self.X = X.copy()
        self._pca, self._X_pca, self._loadings = self._apply_pca(self.X, self._cols)

        if self._use_n_components:
            self._X_pca = self._X_pca.iloc[:, : self._use_n_components]

        self.X = pd.concat([self.X, self._X_pca], axis=1)

        if self._drop_originals:
            self.X = self.X.drop(self._cols, axis=1)
        return self.X

    def _apply_pca(self, X, features):

        X = X.loc[:, features]
        X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)

        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)

        component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]

        X_pca = pd.DataFrame(X_pca, columns=component_names, index=X_scaled.index)

        loadings = pd.DataFrame(
            pca.components_.T,  # transpose the matrix of loadings
            columns=component_names,  # so the columns are the principal components
            index=X.columns,  # and the rows are the original features
        )

        return pca, X_pca, loadings

    def get_variables(self):
        return self._cols

    def get_x_pca(self):
        return self._X_pca

    def get_pca(self):
        return self._pca

    def get_loadings(self):
        return self._loadings
