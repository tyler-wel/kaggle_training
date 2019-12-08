from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
def score_pipeline_validation(my_pipeline, X_train, y_train):
    """Return the average MAE over 5 CV folds of the given pipeline.
    
    Keyword argument:
    my_pipeline -- the given pipeline to test
    X_train -- X training set
    y_train -- y training set
    """
    scores = -1 * cross_val_score(my_pipeline, X_train, y_train,
                              cv=5,
                              scoring='neg_mean_absolute_error')
    return scores.mean()

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
def get_pipeline_score(n_estimators, X, y):
    """Return the avg MAE over 5 CV folds of a random forest model.
    
    Keyword argument:
    n_estimators -- the number of trees in the forest
    X -- given training predictors
    y -- given training target
    """
    
    my_pipeline = Pipeline(steps=[
        # Assuming our data is numerical only
        ('preprocessor', SimpleImputer()),
        ('model', RandomForestRegressor(n_estimators=n_estimators, random_state=0))
    ])
    
    scores = -1 * cross_val_score(my_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')
    return scores.mean()