from sklearn.model_selection import StratifiedKFold, train_test_split
import joblib
from configs.config import RANDOM_SEED, CV_SPLITS, TEST_SPLITS

def split(data, test_size=0.15, cv_splits=5):
    '''
    Split data on train/test/val sets
    
    Return:
        train_ids: list, SK_ID_CURR ids of train data
        test_ids: list, SK_ID_CURR ids of test data
        splits: tuple of lists, cv splits on SK_ID_CURR
    '''

    ids = data['SK_ID_CURR']
    y = data['TARGET']

    # Split data on train / test by SK_ID_CURR
    train_ids, test_ids = train_test_split(ids, test_size=test_size, stratify=y, random_state=RANDOM_SEED)

    # Get training data X and y to put it in cv.split
    train_df = data[data['SK_ID_CURR'].isin(train_ids)]
    X_train = train_df.drop(columns='TARGET')
    y_train = train_df['TARGET']

    # Split training data on train / validation subsets, using SK_ID_CURR
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_SEED)
    splits = [
        (
            X_train.iloc[train_idx]['SK_ID_CURR'].values,
            X_train.iloc[val_idx]['SK_ID_CURR'].values
        )
        for train_idx, val_idx in cv.split(X_train, y_train)
    ]

    joblib.dump(splits, CV_SPLITS)
    joblib.dump((train_ids, test_ids), TEST_SPLITS)

    return train_ids, test_ids, splits