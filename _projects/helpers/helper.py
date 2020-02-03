from termcolor import cprint

def get_splits(dataframe, valid_fraction=0.1):
    """
    Splits a dataframe into train, validation, and test sets. 
    Set the size of the validation and test sets with valid_fraction.
        dataframe: Pandas dataframe to be split
        valid_fraction: % to split data by (default 0.1)
    """
    valid_rows = int(len(dataframe) * valid_fraction)
    train = dataframe[ : -valid_rows * 2 ]
    valid = dataframe[ - valid_rows * 2 : -valid_rows]
    test = dataframe[ -valid_rows : ]
    return train, valid, test

from itertools import combinations
def process_titanic(df, cat_col, num_col):
  """
    Process the Titanic dataset
      df: Pandas dataframe
      cat_col: categorical features
      num_col: numerical features
  """
  # interactions
  for comb in combinations(cat_col, 2):
    new_feat = comb[0] + "_" + comb[1]
    df[new_feat] = df[comb[0]].astype(str) + "_" + df[comb[1]].astype(str)
  return(df)



from sklearn.pipeline import Pipeline
# https://ramhiser.com/post/2018-03-25-feature-selection-with-scikit-learn-pipeline/
class PipelineFS(Pipeline):
  def fit(self, X, y=None, **fit_params):
    """
    Extension of Pipeline to support feature selection inside the pipeline.
    """
    print(type(self))
    super().fit(X, y, **fit_params)
    self.feature_importances_ = self.steps[-1][-1].feature_importances_
    return self
  
  
# https://www.kaggle.com/baghern/a-deep-dive-into-sklearn-pipelines
from sklearn.base import BaseEstimator, TransformerMixin

class TextSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on text columns in the data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]
    
class NumberSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on numeric columns in the data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.key]]
