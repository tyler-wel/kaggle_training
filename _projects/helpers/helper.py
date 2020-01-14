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
  
from sklearn.pipeline import Pipeline
# https://ramhiser.com/post/2018-03-25-feature-selection-with-scikit-learn-pipeline/
class PipelineFS(Pipeline):
  def fit(self, X, y=None, **fit_params):
    """
    Extension of Pipeline to support feature selection inside the pipeline.
    """
    super(PipelineFS, self).fit(X, y, **fit_params)
    self.feature_importances_ = self.steps[-1][-1].feature_importances_
    return self