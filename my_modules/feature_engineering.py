# helper functions for feature engineering
import lightgbm as lgb
from sklearn import metrics
from termcolor import colored # colored prints


def get_data_splits(dataframe, valid_fraction=0.1):
    """
    Returns a valid, test, and train dataframe.
        :param dataframe: Pandas dataframe to be split
        :param valid_fraction=0.1: % to split data by (default 0.1)
    """
    valid_size = int(len(dataframe) * valid_fraction)
    train = dataframe[ : -valid_size * 2]
    # valid_size = test_size, last two sections of data (80, 10, 10)
    valid = dataframe[-valid_size * 2 : -valid_size]
    test = dataframe[-valid_size : ]

    return train, valid, test

def train_kickstarter_model(train, valid):
    """
    Trains kickstarter model based on provided train and validation datasets.
        :param train: Training set
        :param valid: Validation set
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

