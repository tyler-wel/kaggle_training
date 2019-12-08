from sklearn.metrics import mean_absolute_error
def score_basic_model(model, X_train, X_valid, y_train, y_valid):
    """Uses mean_absolute_error to score a given model and data set
    
    """
    model.fit(X_train, y_train)
    predict = model.predict(X_valid)
    return mean_absolute_error(y_valid, predict)
