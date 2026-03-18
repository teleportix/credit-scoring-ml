import pandas as pd
import numpy as np
from configs.config import *
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

def build_features(save=True):
    '''
    Perform feature engineering with raw data.

    Returns:
        pd.DataFrame: Save engineered DataFrame to data/processed folder and return it. 

    Raises:
        Exception: for unexpected errors during aggregation.
    '''

    raw_data = load_data()
    logger.info('Raw data read successfully')

    applic_df = raw_data['applic']
    bureau_df = raw_data['bureau']
    bureau_balance_df = raw_data['bureau_balance']
    previous_applic_df = raw_data['previous_applic']
    pos_cash_df = raw_data['pos_cash']
    instal_df = raw_data['instal']
    credit_card_df = raw_data['credit_card']

    try:
        start = time.time()

        step_start = time.time()
        bureau_balance_agg_df = bureau_balance_agg(bureau_balance_df)
        logger.info(f'bureau_balance aggregation done in {time.time() - step_start:.2f}s, shape: {bureau_balance_agg_df.shape}')

        step_start = time.time()
        bureau_agg_df = bureau_agg(bureau_balance_agg_df, bureau_df)
        logger.info(f'bureau aggregation done in {time.time() - step_start:.2f}s, shape: {bureau_agg_df.shape}')

        step_start = time.time()
        pos_cash_agg_df = pos_cash_agg(pos_cash_df)
        logger.info(f'pos_cash aggregation done in {time.time() - step_start:.2f}s, shape: {pos_cash_agg_df.shape}')

        step_start = time.time()
        installment_agg_df = installment_agg(instal_df)
        logger.info(f'installment aggregation done in {time.time() - step_start:.2f}s, shape: {installment_agg_df.shape}')

        step_start = time.time()
        credit_card_agg_df = credit_card_agg(credit_card_df)
        logger.info(f'credit_card aggregation done in {time.time() - step_start:.2f}s, shape: {credit_card_agg_df.shape}')

        step_start = time.time()
        previous_agg_df = previous_agg(previous_applic_df, credit_card_agg_df, pos_cash_agg_df, installment_agg_df)
        logger.info(f'previous_application aggregation done in {time.time() - step_start:.2f}s, shape: {previous_agg_df.shape}')

        applic_agg_df = applic_agg(applic_df, bureau_agg_df, previous_agg_df)
        logger.info(f'Full aggregation done in {time.time() - start:.2f}s successfully. Final dataset shape: {applic_agg_df.shape}')
        logger.info(f'Total features: {applic_agg_df.shape[1] - 2}')
    except Exception as e:
        logger.error(f"Unexpected error in aggregation: {e}")
        raise

    if save:
        applic_agg_df.to_csv(PROCESSED_PATH, index=False)
        logger.info(f'Processed dataset saved to: {PROCESSED_PATH}')

    return applic_agg_df



def load_data():
    '''
    Load raw data for feature engineering.

    Returns:
        Dictionary of pd.DataFrame: Dictionary of raw data stored in pd.DataFrame, keys: ['applic', 'bureau', 'bureau_balance', 'previous_applic', 'pos_cash', 'instal', 'credit_card'].

    Raises:
        FileNotFoundError: If no path to the raw files is found. 
    '''
    raw_paths = [APPLIC_PATH, BUREAU_PATH, BUREAU_BALANCE_PATH, PREVIOUS_APPLIC_PATH, POS_CASH_BALANCE_PATH, INSTAL_PAYMENTS_PATH, CREDIT_CARD_BALANCE_PATH]

    if not all(p.exists() for p in raw_paths):
        raise FileNotFoundError(
            "Raw data not found. See README.md for download insturctions."
        )

    df = pd.read_csv(APPLIC_PATH)
    bureau_df = pd.read_csv(BUREAU_PATH)
    bureau_balance_df = pd.read_csv(BUREAU_BALANCE_PATH)
    previous_applic_df = pd.read_csv(PREVIOUS_APPLIC_PATH)
    pos_cash_df = pd.read_csv(POS_CASH_BALANCE_PATH)
    instal_df = pd.read_csv(INSTAL_PAYMENTS_PATH)
    credit_card_df = pd.read_csv(CREDIT_CARD_BALANCE_PATH)

    return {'applic': df, 'bureau': bureau_df, 'bureau_balance': bureau_balance_df, 
            'previous_applic': previous_applic_df, 'pos_cash': pos_cash_df, 
            'instal': instal_df, 'credit_card': credit_card_df}

def bureau_balance_agg(bureau_balance_df):

    bureau_balance_agg = bureau_balance_df.groupby('SK_ID_BUREAU').agg(
    loan_duration=('MONTHS_BALANCE', 'count'),
    prepaid_ratio=('STATUS', lambda x: (x == 'C').mean()),
    default_ever=('STATUS', lambda x: int((x == '5').any())),
    bad_dpd_ratio=('STATUS', lambda x: (x.isin(['2', '3', '4', '5']).mean())),
    bad_dpd_count=('STATUS', lambda x: (x.isin(['2', '3', '4', '5']).sum())),
    small_dpd_ratio=('STATUS', lambda x: (x == '1').mean()),
    small_dpd_count=('STATUS', lambda x: (x == '1').sum()),
    paid_in_time_ratio=('STATUS', lambda x: (x == '0').mean()),
    paid_in_time_count=('STATUS', lambda x: (x == '0').sum()),
    unknown_ratio=('STATUS', lambda x: (x == 'X').mean())
    ).reset_index()
    
    return bureau_balance_agg
    
def bureau_agg(bureau_balance_agg_df, bureau_df):

    bureau_df = bureau_df.merge(
    bureau_balance_agg_df, 
    how='left',
    on='SK_ID_BUREAU',
    )
    
    bureau_df = bureau_df.assign(
        early_closure_days= bureau_df['DAYS_CREDIT_ENDDATE'] - bureau_df['DAYS_ENDDATE_FACT'],
        credit_duration= np.abs(bureau_df['DAYS_CREDIT'] - bureau_df['DAYS_ENDDATE_FACT']),
        ever_overdue_flag= (bureau_df['AMT_CREDIT_MAX_OVERDUE'] > 0).astype(int),
        has_bureau_balance_history=bureau_df['loan_duration'].notna().astype(int),

        cnt_curent_overdue= (
            (bureau_df['CREDIT_ACTIVE'] == 'Active') & 
            (bureau_df['CREDIT_DAY_OVERDUE'] > 0)
        ).astype(int),
        overdue_days_active= np.where(
            bureau_df['CREDIT_ACTIVE'] == 'Active', 
            bureau_df['CREDIT_DAY_OVERDUE'],
            0
        ),
        overdue_ratio=safe_devision(bureau_df, 'AMT_CREDIT_SUM_OVERDUE', 'AMT_CREDIT_SUM'),

        credit_sum_active= np.where(
            bureau_df['CREDIT_ACTIVE'] == 'Active',
            bureau_df['AMT_CREDIT_SUM'],
            0
        ),
        annuity_active= np.where(
            bureau_df['CREDIT_ACTIVE'] == 'Active',
            bureau_df['AMT_ANNUITY'],
            0
        )
    )

    bureau_agg = bureau_df.groupby('SK_ID_CURR').agg(
    # credit status
    sold_times=('CREDIT_ACTIVE', lambda x: x.isin(['Sold', 'Bad debt']).sum()),
    closed_ratio=('CREDIT_ACTIVE', lambda x: (x == 'Closed').mean()),
    active_credits=('CREDIT_ACTIVE', lambda x: (x == 'Active').sum()),
    #time related features
    first_credit_time=('DAYS_CREDIT', 'min'),
    overdue_days_mean=('CREDIT_DAY_OVERDUE', 'mean'),
    overdue_days_active_mean=('overdue_days_active', 'mean'),
    overdue_active_max=('CREDIT_DAY_OVERDUE', 'max'),
    early_closure_days_ratio=('early_closure_days', 'mean'),
    credit_duration_mean=('credit_duration', 'mean'),
    prolonged_max=('CNT_CREDIT_PROLONG', 'max'),
    prolonged_times=('CNT_CREDIT_PROLONG', 'sum'),
    last_credit_update=('DAYS_CREDIT_UPDATE', 'max'),
    first_credit_update=('DAYS_CREDIT_UPDATE', 'min'),
    loan_duration_avg=('loan_duration', 'mean'),
    loan_duration_max=('loan_duration', 'max'),
    # ammount related features
    overdue_historical_max=('AMT_CREDIT_MAX_OVERDUE', 'max'),
    overdue_credits_active=('cnt_curent_overdue', 'sum'),
    overdue_ammount_active=('AMT_CREDIT_SUM_OVERDUE', 'sum'),
    overdue_ratio_max=('overdue_ratio', 'max'),
    credit_sum_mean=('AMT_CREDIT_SUM', 'mean'),
    active_credit_sum=('credit_sum_active', 'sum'),
    debt_max=('AMT_CREDIT_SUM_DEBT', 'max'),
    debt_mean=('AMT_CREDIT_SUM_DEBT', 'mean'),
    credit_limit_max=('AMT_CREDIT_SUM_LIMIT', 'max'),
    consumer_credit_sum=('CREDIT_TYPE', lambda x: (x == 'Consumer credit').sum()),
    current_annuity=('annuity_active', 'sum'),
    annuity_mean=('AMT_ANNUITY', 'mean'),
    # dpd severity / frequency
    worst_dpd=('bad_dpd_ratio', 'max'),
    bad_dpd_avg=('bad_dpd_ratio', 'mean'),
    bad_dpd_cnt=('bad_dpd_count', 'sum'),
    bad_dpd_times_max=('bad_dpd_count', 'max'),
    worst_small_dpd=('small_dpd_ratio', 'max'),
    small_dpd_avg=('small_dpd_ratio', 'mean'),
    small_dpd_cnt=('small_dpd_count', 'sum'),
    # flag features
    has_credit_card_bureau=('CREDIT_TYPE', lambda x: int((x == 'Credit card').any())),
    has_microloan=('CREDIT_TYPE', lambda x: int((x == 'Microloan').any())),
    has_bureau_balance_history=('has_bureau_balance_history', 'max'),
    annuity_known=('AMT_ANNUITY', lambda x: int(not x.isna().all())),
    # ratios and other features
    credit_card_cnt=('CREDIT_TYPE', lambda x: (x == 'Credit card').sum()),
    low_risk_loans=('CREDIT_TYPE', lambda x: x.isin(['Car loan', 'Mortgage', 'Loan for business development']).sum()),
    prepaid_ratio_avg=('prepaid_ratio', 'mean'),
    defaults=('default_ever', 'sum'),
    loans_ever_overdue=('ever_overdue_flag', 'sum'),
    loans_overdue_ratio=('ever_overdue_flag', 'mean'),
    paid_in_time_avg=('paid_in_time_ratio', 'mean'),
    paid_in_time_cnt=('paid_in_time_count', 'sum'),
    unknown_ratio_avg=('unknown_ratio', 'mean'),
    ).reset_index()

    bureau_agg['debt_ratio_mean'] = np.where(
        bureau_agg['credit_sum_mean'] != 0,
        bureau_agg['debt_mean'] / bureau_agg['credit_sum_mean'],
        np.nan
    )
    assert bureau_agg['SK_ID_CURR'].is_unique, \
    "Duplicate SK_ID_CURR in bureau_agg"

    return bureau_agg


def pos_cash_agg(pos_cash_df):

    pos_cash_agg = pos_cash_df.groupby('SK_ID_PREV').agg(
    max_cnt_installment=('CNT_INSTALMENT', 'max'),
    is_completed=('NAME_CONTRACT_STATUS', lambda x: int((x == 'Completed').any())),
    sk_dpd_days=('SK_DPD', 'mean'),
    sk_dpd_max=('SK_DPD', 'max'),
    sk_dpd_def_days=('SK_DPD_DEF', 'mean'),
    sk_dpd_def_max=('SK_DPD_DEF', 'max'),
    ).reset_index()

    assert pos_cash_agg['SK_ID_PREV'].is_unique, \
    "Duplicate SK_ID_PREV in pos_cash_agg"

    return pos_cash_agg

def installment_agg(installment_df):
    installment_df = installment_df.sort_values(['SK_ID_PREV', 'DAYS_INSTALMENT'], ascending=[True, False])

    installment_df = installment_df.assign(
    delay_days=installment_df['DAYS_ENTRY_PAYMENT'] - installment_df['DAYS_INSTALMENT'],
    underpay_amt=installment_df['AMT_INSTALMENT'] - installment_df['AMT_PAYMENT'],
    inst_rank_recency=installment_df.groupby('SK_ID_PREV').cumcount(),
    payment_ratio=safe_devision(installment_df, 'AMT_PAYMENT', 'AMT_INSTALMENT')
    )

    installment_df = installment_df.assign(
        delay_days_recent=np.where(
            installment_df['inst_rank_recency'] < LAST_K,
            installment_df['delay_days'],
            np.nan
        ),
        underpay_recent=np.where(
            installment_df['inst_rank_recency'] < LAST_K,
            installment_df['underpay_amt'],
            np.nan
        )
    )

    installment_agg = installment_df.groupby('SK_ID_PREV').agg(
    inst_cnt=('AMT_INSTALMENT', 'count'),
    # timing sevirity / frequency
    payments_duration=('DAYS_INSTALMENT', lambda x: x.max() - x.min()),
    delay_days_mean=('delay_days', 'mean'),
    delay_days_max=('delay_days', 'max'),
    delay_freq_mean=('delay_days', lambda x: (x > 0).mean()),
    # amount severity / frequency
    underpay_amt_max=('underpay_amt', 'max'),
    underpay_freq_mean=('underpay_amt', lambda x: (x > 0).mean()),
    overpay_freq_mean=('underpay_amt', lambda x: (x <= 0).mean()),
    payment_ratio_min=('payment_ratio', 'min'),
    # timing / amount consistency
    delay_days_std=('delay_days', 'std'),
    payment_ratio_std=('payment_ratio', 'std'),
    # recency behaviour
    delay_recent_max=('delay_days_recent', 'max'),
    delay_recent_mean=('delay_days_recent', 'mean'),
    underpay_recent_freq=('underpay_recent', lambda x: (x > 0).mean()),
    # Missing data handling
    unpaid_inst_cnt=('DAYS_ENTRY_PAYMENT', lambda x: x.isna().sum())
    ).reset_index()

    assert installment_agg['SK_ID_PREV'].is_unique, \
    "Duplicate SK_ID_PREV in installment_agg"

    return installment_agg

def credit_card_agg(credit_card_df):

    credit_card_df = credit_card_df.sort_values(['SK_ID_PREV', 'MONTHS_BALANCE'], ascending=False)

    credit_card_df = credit_card_df.assign(
        rank_recency=credit_card_df.groupby('SK_ID_PREV').cumcount(),
        balance_rate=safe_devision(credit_card_df, 'AMT_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL'),
        spend_rate=safe_devision(credit_card_df, 'AMT_DRAWINGS_CURRENT', 'AMT_BALANCE'),
        interest_rate=np.where(
            credit_card_df['AMT_RECEIVABLE_PRINCIPAL'] != 0,
            (credit_card_df['AMT_RECIVABLE'] - credit_card_df['AMT_RECEIVABLE_PRINCIPAL']) / credit_card_df['AMT_RECEIVABLE_PRINCIPAL'],
            np.nan
        ),
        cost_ratio=safe_devision(credit_card_df, 'AMT_TOTAL_RECEIVABLE', 'AMT_RECEIVABLE_PRINCIPAL'),
        loan_rate=safe_devision(credit_card_df, 'AMT_BALANCE', 'AMT_RECEIVABLE_PRINCIPAL'),
        has_credit_limit = (credit_card_df['AMT_CREDIT_LIMIT_ACTUAL'] > 0 ).astype(int),
        zero_balance=(credit_card_df['AMT_BALANCE'] == 0).astype(int),
        zero_principal=(credit_card_df['AMT_RECEIVABLE_PRINCIPAL'] == 0).astype(int),
    )

    # recent behaviour features
    credit_card_df = credit_card_df.assign(
        credit_limit_recent=np.where(
            credit_card_df['rank_recency'] < LAST_K,
            credit_card_df['AMT_CREDIT_LIMIT_ACTUAL'],
            np.nan
        ),
        dpd_recent=np.where(
            credit_card_df['rank_recency'] < LAST_K,
            credit_card_df['SK_DPD'],
            np.nan
        ),
        balance_rate_recent=np.where(
            credit_card_df['rank_recency'] < LAST_K,
            credit_card_df['balance_rate'],
            np.nan
        ),
        loan_rate_recent=np.where(
            credit_card_df['rank_recency'] < LAST_K,
            credit_card_df['loan_rate'],
            np.nan
        ),
        current_debt=np.where(
        (credit_card_df['NAME_CONTRACT_STATUS'] == 'Active') & (credit_card_df['MONTHS_BALANCE'] == -1),
        credit_card_df['AMT_TOTAL_RECEIVABLE'],
        0
        )
    )

    credit_card_agg = credit_card_df.groupby('SK_ID_PREV').agg(
    has_credit_card_internal=('MONTHS_BALANCE', lambda x: int(x.size > 0)),
    # Ammount related aggregation
    cc_limit_max=('AMT_CREDIT_LIMIT_ACTUAL', 'max'),
    cc_spend_mean=('spend_rate', 'mean'),
    cc_balance_rate=('balance_rate', 'mean'),
    cc_payment_mean=('AMT_PAYMENT_TOTAL_CURRENT', 'mean'),
    # credit principal (interest / cost / loan rate)
    cc_interest_mean=('interest_rate', 'mean'),
    cc_cost_mean=('cost_ratio', 'mean'),
    cc_cost_worst=('cost_ratio', 'max'),
    cc_loan_rate_mean=('loan_rate', 'mean'),
    cc_loan_rate_worst=('loan_rate', 'min'),
    # Days past due aggregation
    cc_dpd_max=('SK_DPD', 'max'),
    cc_dpd=('SK_DPD', 'sum'),
    cc_dpd_freq=('SK_DPD', lambda x: (x > 0).mean()),
    cc_dpd_def_max=('SK_DPD_DEF', 'max'),
    cc_dpd_def=('SK_DPD_DEF', 'sum'),
    cc_dpd_def_freq=('SK_DPD_DEF', lambda x: (x > 0).mean()),
    # draw / intallment times 
    cc_draw_mean=('CNT_DRAWINGS_CURRENT', 'mean'),
    cc_inst_mean=('CNT_INSTALMENT_MATURE_CUM', 'mean'),
    cc_has_limit_mean=('has_credit_limit', 'mean'),
    cc_zero_balance_mean=('zero_balance', 'mean'),
    cc_zero_principal_mean=('zero_principal', 'mean'),
    #recency behaviour
    cc_recent_limit_max=('credit_limit_recent', 'max'),
    cc_recent_dpd_freq=('dpd_recent', lambda x: (x > 0).mean()),
    cc_recent_dpd_max=('dpd_recent', 'max'),
    cc_recent_balance_rate=('balance_rate_recent', 'mean'),
    cc_recent_loan_rate=('loan_rate_recent', 'mean'),
    cc_current_debt=('current_debt', 'sum')
    ).reset_index()

    assert credit_card_agg['SK_ID_PREV'].is_unique, \
    "Duplicate SK_ID_PREV in credit_card_agg"

    return credit_card_agg

def previous_agg(previous_df, credit_card_agg_df, pos_cash_agg_df, installment_agg_df):

    previous_df = (
    previous_df
    .merge(credit_card_agg_df, how='left', on='SK_ID_PREV')
    .merge(pos_cash_agg_df, how='left', on='SK_ID_PREV')
    .merge(installment_agg_df, how='left', on='SK_ID_PREV')
    )

    # sort so that the most recent applications are at the top
    previous_df.sort_values(['SK_ID_CURR', 'DAYS_DECISION'], inplace=True, ascending=False)

    DATE_COLS = [
    'DAYS_FIRST_DRAWING',
    'DAYS_FIRST_DUE',
    'DAYS_LAST_DUE_1ST_VERSION',
    'DAYS_LAST_DUE',
    'DAYS_TERMINATION'
    ]

    previous_df[DATE_COLS] = previous_df[DATE_COLS].replace(365243, np.nan)

    previous_df['is_approved'] = previous_df['NAME_CONTRACT_STATUS'].isin(['Approved', 'Unused offer']).astype(int)
    previous_df['is_refused'] = (previous_df['NAME_CONTRACT_STATUS'] == 'Refused').astype(int)

    previous_df = previous_df.assign(
        rank_recency=previous_df.groupby('SK_ID_CURR').cumcount(),
        approved_annuity=previous_df['AMT_ANNUITY'].where(previous_df['is_approved'] == 1),
        approved_credit=previous_df['AMT_CREDIT'].where(previous_df['is_approved'] == 1),
        refused_credit=previous_df['AMT_CREDIT'].where(previous_df['is_refused'] == 1),
        request_diff_ratio=np.where(
            previous_df['AMT_APPLICATION'] != 0,
            (previous_df['AMT_APPLICATION'] - previous_df['AMT_CREDIT']) / previous_df['AMT_APPLICATION'],
            np.nan
        ),
        was_disbursed=previous_df['DAYS_FIRST_DRAWING'].notna().astype(int),
        grace_period=previous_df['DAYS_FIRST_DUE'] - previous_df['DAYS_FIRST_DRAWING'],
        delay=previous_df['DAYS_TERMINATION'] - previous_df['DAYS_LAST_DUE_1ST_VERSION'],
        prolongation=previous_df['DAYS_LAST_DUE'] - previous_df['DAYS_LAST_DUE_1ST_VERSION'],
        loan_duration=previous_df['DAYS_LAST_DUE_1ST_VERSION'] - previous_df['DAYS_FIRST_DUE'],
        unpaid_ratio=safe_devision(previous_df, 'unpaid_inst_cnt', 'inst_cnt')
    )

    previous_df['recent_approved'] = np.where(
        previous_df['rank_recency'] < LAST_K,
        previous_df['is_approved'],
        np.nan
    )
    
    reject_mapping = {
    'SCO': 'scoring',
    'SCOFR': 'scoring',
    'HC': 'internal_reject',
    'SYSTEM': 'internal_reject',
    'LIMIT': 'affordability',
    'CLIENT': 'client_reason'
}
    
    previous_df['reject_code'] = previous_df['CODE_REJECT_REASON'].map(reject_mapping).fillna('other')

    previous_agg = previous_df.groupby('SK_ID_CURR').agg(
    # POS_CASH_balance related aggregation
    prev_PC_installment_mean=('max_cnt_installment', 'mean'),
    prev_dpd_max=('sk_dpd_max', 'max'),
    prev_PC_dpd_mean=('sk_dpd_days', 'mean'),
    prev_PC_dpd_def_max=('sk_dpd_def_max', 'max'),
    prev_PC_dpd_def_mean=('sk_dpd_def_days', 'mean'),
    prev_PC_completed_mean=('is_completed', 'mean'),
    #Installments payments related aggregation
        # timing sevirity / frequency
    prev_inst_mean=('inst_cnt', 'mean'),
    prev_inst_duration_max=('payments_duration', 'max'),
    prev_inst_duration_mean=('payments_duration', 'mean'),
    prev_inst_delay_max=('delay_days_max', 'max'),
    prev_inst_delay_mean=('delay_days_mean', 'mean'),
    prev_inst_delay_freq_mean=('delay_freq_mean', 'mean'),
        # amount sevirity / frequency
    prev_inst_underpay_amt_max=('underpay_amt_max', 'max'),
    prev_inst_underpay_freq_mean=('underpay_freq_mean', 'mean'),
    prev_inst_overpay_freq_mean=('overpay_freq_mean', 'mean'),
    prev_inst_payment_ratio_mean=('payment_ratio_min', 'mean'),
    prev_inst_payment_ratio_worst=('payment_ratio_min', 'min'),
        # timing / amount consistency
    prev_inst_delay_std_max=('delay_days_std', 'max'),
    prev_inst_delay_std_mean=('delay_days_std', 'mean'),
    prev_inst_delay_std_std=('delay_days_std', 'std'),
    prev_inst_pay_ratio_std_max=('payment_ratio_std', 'max'),
    prev_inst_pay_ratio_std_mean=('payment_ratio_std', 'mean'),
    prev_inst_pay_ratio_std_std=('payment_ratio_std', 'std'),
        # recency behaviour
    prev_inst_delay_recent_max=('delay_recent_max', 'max'),
    prev_inst_delay_recent_mean=('delay_recent_mean', 'mean'),
    prev_inst_underpay_recent_freq=('underpay_recent_freq', 'mean'),
        # Missing data handling
    prev_inst_unpaid_ratio_max=('unpaid_ratio', 'max'),
    prev_inst_unpaid_ratio_mean=('unpaid_ratio', 'mean'),
    #Credit card related aggregation
    has_credit_card_internal=('has_credit_card_internal', 'max'),
        # ammount related aggregation
    prev_cc_limit_max=('cc_limit_max', 'max'),
    prev_cc_spend_mean=('cc_spend_mean', 'mean'),
    prev_cc_balance_rate_mean=('cc_balance_rate', 'mean'),
    prev_cc_balance_rate_max=('cc_balance_rate', 'max'),
    prev_cc_payment_mean=('cc_payment_mean', 'mean'),
        # credit principal (interest / cost / loan rate)
    prev_cc_interest_mean=('cc_interest_mean', 'mean'),
    prev_cc_cost_mean=('cc_cost_mean', 'mean'),
    prev_cc_cost_worst=('cc_cost_worst', 'max'),
    prev_cc_loan_rate_mean=('cc_loan_rate_mean', 'mean'),
    prev_cc_loan_rate_worst=('cc_loan_rate_worst', 'min'),
        # days past due aggregation
    prev_cc_dpd_max=('cc_dpd_max', 'max'),
    prev_cc_dpd_mean=('cc_dpd', 'mean'),
    prev_cc_dpd_freq=('cc_dpd_freq', 'mean'),
    prev_cc_dpd_def_max=('cc_dpd_def_max', 'max'),
    prev_cc_dpd_def_mean=('cc_dpd_def', 'mean'),
    prev_cc_dpd_def_freq=('cc_dpd_def_freq', 'mean'),
        # draw / intallment times 
    prev_cc_draw_mean=('cc_draw_mean', 'mean'),
    prev_cc_inst_mean=('cc_inst_mean', 'mean'),
    prev_cc_has_limit_mean=('cc_has_limit_mean', 'mean'),
    prev_cc_zero_balance_mean=('cc_zero_balance_mean', 'mean'),
    prev_cc_zero_principal_mean=('cc_zero_principal_mean', 'mean'),
        #recency behaviour
    prev_cc_recent_limit_max=('cc_recent_limit_max', 'max'),
    prev_cc_recent_dpd_freq=('cc_recent_dpd_freq', 'mean'),
    prev_cc_recent_dpd_max=('cc_recent_dpd_max', 'max'),
    prev_cc_recent_balance_rate=('cc_recent_balance_rate', 'mean'),
    prev_cc_recent_loan_rate=('cc_recent_loan_rate', 'mean'),
    cc_current_debt=('cc_current_debt', 'sum'),
    # Annuity / credit amt 
    prev_annuity_mean=('approved_annuity', 'mean'),
    prev_annuity_max=('approved_annuity', 'max'),
    prev_credit_mean=('approved_credit', 'mean'),
    prev_credit_max=('approved_credit', 'max'),
    prev_refused_credit_mean=('refused_credit', 'mean'),
    prev_refused_credit_max=('refused_credit', 'max'),
    prev_request_diff_mean=('request_diff_ratio', 'mean'),
    # Approved / refused times
    prev_applic_cnt=('SK_ID_PREV', 'count'),
    prev_approved_cnt=('is_approved', 'sum'),
    prev_refused_cnt=('is_refused', 'sum'),
    # Downpayment aggregation
    prev_downpayment_mean=('AMT_DOWN_PAYMENT', 'mean'),
    prev_downpayment_max=('AMT_DOWN_PAYMENT', 'max'),
    prev_no_downpayment_mean=('AMT_DOWN_PAYMENT', lambda x: x.isna().mean()),
    prev_downpayment_rate_mean=('RATE_DOWN_PAYMENT', 'mean'),
    prev_downpayment_rate_max=('RATE_DOWN_PAYMENT', 'max'),
    # Reject reason binning
    prev_refused_sco_cnt=('reject_code', lambda x: (x == 'scoring').sum()),
    prev_refused_limit_cnt=('reject_code', lambda x: (x == 'affordability').sum()),
    prev_refused_hc_cnt=('reject_code', lambda x: (x == 'internal_reject').sum()),
    prev_refused_client_cnt=('reject_code', lambda x: (x == 'client_reason').sum()),
    prev_refused_other_cnt=('reject_code', lambda x: (x == 'other').sum()),
    # Recent behaviour
    prev_last_application=('DAYS_DECISION', 'min'),
    prev_recent_applications_cnt=('DAYS_DECISION', lambda x: (x >= -YEAR).sum()),
    prev_recent_approved=('recent_approved', 'sum'),
    # Yield group binning
    prev_high_yield_cnt=('NAME_YIELD_GROUP', lambda x: (x == 'high').sum()),
    prev_middle_yield_cnt=('NAME_YIELD_GROUP', lambda x: (x == 'middle').sum()),
    prev_low_yield_cnt=('NAME_YIELD_GROUP', lambda x: x.isin(['low_normal', 'low_action']).sum()),
    # Time related aggregation
    prev_was_disbursed=('was_disbursed', 'mean'),
    prev_grace_period_mean=('grace_period', 'mean'),
    prev_delay_mean=('delay', 'mean'), 
    prev_delay_max=('delay', 'max'),
    prev_prolong_mean=('prolongation', 'mean'),
    prev_prolong_max=('prolongation', 'max'),
    prev_loan_duration_mean=('loan_duration', 'mean'),
    prev_loan_duration_max=('loan_duration', 'max'),
    # Other aggregation
    prev_insured_mean=('NFLAG_INSURED_ON_APPROVAL', 'mean'),
    prev_type_pos=('NAME_PORTFOLIO', lambda x: (x == 'POS').sum()),
    prev_type_cash=('NAME_PORTFOLIO', lambda x: (x == 'Cash').sum()),
    prev_cnt_payment_max=('CNT_PAYMENT', 'max'),
    prev_cnt_payment_mean=('CNT_PAYMENT', 'mean'),
    ).reset_index()

    assert previous_agg['SK_ID_CURR'].is_unique, \
    "Duplicate SK_ID_CURR in previous_agg"

    previous_agg['approved_ratio'] = safe_devision(previous_agg, 'prev_approved_cnt', 'prev_applic_cnt')
    previous_agg['has_prev_application'] = 1

    return previous_agg


def applic_agg(applic_df, bureau_agg_df, previous_agg_df):
    df = (
        applic_df
        .merge(bureau_agg_df, how='left', on='SK_ID_CURR')
        .merge(previous_agg_df, how='left', on='SK_ID_CURR')
    )
    assert df['SK_ID_CURR'].is_unique, \
    "Duplicate SK_ID_CURR after merge"


    df = df.assign(
    credit_income_ratio=safe_devision(df, 'AMT_CREDIT', 'AMT_INCOME_TOTAL'),
    annuity_income_ratio=safe_devision(df, 'AMT_ANNUITY', 'AMT_INCOME_TOTAL'),
    log_income=np.log1p(df['AMT_INCOME_TOTAL']),
    log_credit=np.log1p(df['AMT_CREDIT']),
    has_many_children=(df['CNT_CHILDREN'] >= 3).astype(int),
    cnt_children_capped=df['CNT_CHILDREN'].clip(upper=3),
    ext_source_1_missing=df['EXT_SOURCE_1'].isna().astype(int),
    ext_source_3_missing=df['EXT_SOURCE_3'].isna().astype(int),
    credit_annuity_ratio=safe_devision(df, 'AMT_ANNUITY', 'AMT_CREDIT'),
    credit_over_goods_price= df['AMT_CREDIT'] - df['AMT_GOODS_PRICE'],
    days_without_work= df['DAYS_BIRTH'] - df['DAYS_EMPLOYED'],
    credit_per_family_member=safe_devision(df, 'AMT_CREDIT', 'CNT_FAM_MEMBERS'),
    annuity_per_family_member=safe_devision(df, 'AMT_ANNUITY', 'CNT_FAM_MEMBERS'),
    income_per_family_member=safe_devision(df, 'AMT_INCOME_TOTAL', 'CNT_FAM_MEMBERS')
    )

    df = df.assign(
    has_realty_info=df['APARTMENTS_AVG'].notna().astype(int),
    has_variability=df['prev_inst_pay_ratio_std_mean'].notna().astype(int),
    has_credit_history=df['first_credit_time'].notna().astype(int),
    )

    #365243 means NaN in dataset, we add DAYS_EMPLOYED_MISSING as borrowers whith NaN values have lower default rate
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
    df['days_employed_missing'] = df['DAYS_EMPLOYED'].isna().astype(int)

    # Replace inf and -inf in all rows where needed
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df

# Helper functions

def reduce_memory(df, verbose=True):
    """Reduce memory usage of a dataframe by setting data types. """
    start_mem = df.memory_usage().sum() / 1024 ** 2

    for col in df.columns:
        col_type = df[col].dtypes

        if pd.api.types.is_object_dtype(col_type):
            n_unique = df[col].nunique(dropna=False)
            n_total = len(df[col])
            if (n_unique / n_total) < 0.5:
                df[col] = df[col].astype('category')
            continue

        if pd.api.types.is_integer_dtype(col_type):
            cmin = df[col].min()
            cmax = df[col].max()

            if df[col].isnull().any():
                if cmin > -128 and cmax < 127:
                    df[col] = df[col].astype('Int8')
                elif cmin > -32768 and cmax < 32767:
                    df[col] = df[col].astype('Int16')
                else:
                    df[col] = df[col].astype('Int32')
            elif cmin > np.iinfo(np.int8).min and cmax < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif cmin > np.iinfo(np.int16).min and cmax < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif cmin > np.iinfo(np.int32).min and cmax < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)

        elif pd.api.types.is_float_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast='float')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    memory_reduction = 100 * (start_mem - end_mem) / start_mem

    if verbose:
        print(f'Memory: {start_mem:.2f} MB → {end_mem:.2f} MB '
              f'({memory_reduction:.1f}% reduction)')
        
    return df

def safe_devision(df, num, den):
    '''
    Helper function for safe devision of numerator: num on denominator: den in DataFrame: df.

    Returns: pd.Series
    '''
    devided = np.where(
        df[den] != 0,
        df[num] / df[den],
        np.nan
    )
    
    return devided