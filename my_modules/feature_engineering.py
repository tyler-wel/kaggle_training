# helper functions for feature engineering
import lightgbm as lgb
from sklearn import metrics
from termcolor import colored, cprint # colored prints


def get_kickstarter_splits(dataframe, valid_fraction=0.1):
    """
    Splits a dataframe into train, validation, and test sets. 
    Set the size of the validation and test sets with valid_fraction.
        dataframe: Pandas dataframe to be split
        valid_fraction: % to split data by (default 0.1)
    """
    valid_size = int(len(dataframe) * valid_fraction)
    train = dataframe[ : -valid_size * 2]
    # valid_size = test_size, last two sections of data (80, 10, 10)
    valid = dataframe[-valid_size * 2 : -valid_size]
    test = dataframe[-valid_size : ]

    return train, valid, test

def get_ad_splits(dataframe, valid_fraction=0.1):
    """
    Splits a dataframe into train, validation, and test sets. First
    orders by the column 'click_time'. Set the size of the validation
    and test sets with valid_fraction.
        dataframe: Pandas dataframe to be split
        valid_fraction: % to split data by (default 0.1)
    """
    dataframe = dataframe.sort_values('click_time')
    valid_rows = int(len(dataframe) * valid_fraction)
    train = dataframe[ : -valid_rows * 2 ]
    valid = dataframe[ - valid_rows * 2 : -valid_rows]
    test = dataframe[ -valid_rows : ]
    return train, valid, test

def train_kickstarter_model(train, valid):
    """
    Trains kickstarter model based on provided train and validation datasets.
        train: Training set
        valid: Validation set
    """
    feature_cols = train.columns.drop('outcome')

    dtrain = lgb.Dataset(train[feature_cols], label=train['outcome'])
    dvalid = lgb.Dataset(valid[feature_cols], label=valid['outcome'])

    params = {'num_leaves':64, 'objective':'binary', 'metric':'auc', 'seed':7}
    print(colored('Training model...', 'cyan'))
    bst = lgb.train(params, dtrain, num_boost_round=1000, valid_sets=[dvalid], early_stopping_rounds=10, verbose_eval=False)
    valid_pred = bst.predict(valid[feature_cols])
    valid_score = metrics.roc_auc_score(valid['outcome'], valid_pred)
    print(f"Validation AUC score: {valid_score:.4f}")
    return bst

def train_ad_model(train, valid, test=None, feature_cols=None):
    """
    Trains ad clicks model based on provided train and validation datasets.
        train: Training set
        valid: Validation set
        test: Test set (optional)
        feature_cols: Feature columns (optional)
    """
    if feature_cols is None:
        feature_cols = train.columns.drop(['click_time', 'attributed_time', 'is_attributed'])
    dtrain = lgb.Dataset(train[feature_cols], label=train['is_attributed'])
    dvalid = lgb.Dataset(valid[feature_cols], label=valid['is_attributed'])
    
    params = { 'num_leaves':64, 'objective':'binary', 'metric':'auc', 'seed':7 }
    num_round = 1000
    print(colored("Training model...", "cyan"))
    bst = lgb.train(params, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=20, verbose_eval=False)
    valid_pred = bst.predict(valid[feature_cols])
    valid_score = metrics.roc_auc_score(valid['is_attributed'], valid_pred)
    print(f"Validation AUC score: {valid_score:.4f}")
    
    if test is not None:
        test_pred = bst.predict(test[feature_cols])
        test_score = metrics.roc_auc_score(test['is_attributed'], test_pred)
        return bst, valid_score, test_score
    else:
        return bst, valid_score
    
def print_scores(scores):
    cprint('Scores so far...', 'green')
    for key in scores:
        print(f"{key}: {scores[key]:.4f}")
    m = max(scores, key=scores.get)
    
    cprint(f'Best score so far: {m} : {scores[m]:.4f}', 'cyan')