from pathlib import Path

PROJECT_ROOT =  Path(__file__).resolve().parents[1]

# raw data paths
APPLIC_PATH = PROJECT_ROOT / 'data/raw/application_train.csv'
BUREAU_PATH = PROJECT_ROOT / 'data/raw/bureau.csv'
BUREAU_BALANCE_PATH = PROJECT_ROOT / 'data/raw/bureau_balance.csv'
PREVIOUS_APPLIC_PATH = PROJECT_ROOT / 'data/raw/previous_application.csv'
POS_CASH_BALANCE_PATH = PROJECT_ROOT / 'data/raw/POS_CASH_balance.csv'
INSTAL_PAYMENTS_PATH = PROJECT_ROOT / 'data/raw/installments_payments.csv'
CREDIT_CARD_BALANCE_PATH = PROJECT_ROOT / 'data/raw/credit_card_balance.csv'

# interim data paths, uses as temporary storage for faster evaluation
INTERIM_PATH = PROJECT_ROOT / 'data/interim'
INTERIM_BUREAU = INTERIM_PATH / 'bureau_agg.csv'
INTERIM_INSTAL = INTERIM_PATH / 'installment_agg.csv'
INTERIM_CREDIT_CARD = INTERIM_PATH / 'credit_card.csv'
INTERIM_POS_CASH_BALANCE = INTERIM_PATH / 'pos_cash_balance.csv'
INTERIM_PREVIOUS_APPLIC = INTERIM_PATH / 'previous_application.csv'

# experiment results
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
HYPERPARAM_DIR = ARTIFACTS_DIR / "hyperparam_selection"
FEATURE_SELECTION_DIR = ARTIFACTS_DIR / "feature_selection"
ABLATION_TEST = FEATURE_SELECTION_DIR / 'lr_albation_test.csv'
MISSING_THRESHOLD_TEST = FEATURE_SELECTION_DIR / 'lr_missing_threshold_test.csv'

# processed data and computed splits
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data/processed'
CV_SPLITS = PROCESSED_DATA_DIR / 'cv_splits.pkl'
TEST_SPLITS = PROCESSED_DATA_DIR / 'test_splits.pkl'
PROCESSED_PATH = PROCESSED_DATA_DIR / 'application_features_baseline.csv'

# global constants
LAST_K = 3
YEAR = 365.25
RANDOM_SEED = 42