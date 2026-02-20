The Logistic Regression baseline achieves a ROC-AUC of approximately 0.74, indicating that the selected and engineered features provide meaningful discriminatory power between defaulters and non-defaulters.

Threshold analysis demonstrates the expected recall–precision trade-off and allows decision thresholds to be selected according to business risk tolerance rather than relying on a fixed default cutoff.

Feature ablation experiments show that even features with small marginal coefficients contribute to overall model stability and ranking performance, and are therefore retained to improve robustness rather than being removed aggressively

baseline 20 features Logistic regression results:
ROC-AUC: 0.7363202540176794

threshold	recall	precision
0	0.1	0.997382	0.081565
1	0.2	0.974421	0.088717
2	0.3	0.914804	0.103335
3	0.4	0.819134	0.125614
4	0.5	0.674522	0.155327
5	0.6	0.500705	0.195748
6	0.7	0.308157	0.253521
7	0.8	0.130312	0.329598
8	0.9	0.016918	0.340081

Including all features with 3 constructed, only using application dataset Roc_auc:
Train ROC-AUC: 0.7513062645430759, validation ROC-AUC: 0.753292064612759

threshold	recall	precision
0	0.1	0.996931	0.082879
1	0.2	0.970464	0.092584
2	0.3	0.914269	0.108808
3	0.4	0.819908	0.131385
4	0.5	0.693134	0.162114
5	0.6	0.524166	0.199679
6	0.7	0.347526	0.255319
7	0.8	0.169544	0.336890
8	0.9	0.028002	0.408964

Index(['DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_ID_PUBLISH',
       'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY',
       'REG_CITY_NOT_WORK_CITY', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
       'DAYS_LAST_PHONE_CHANGE', 'CODE_GENDER_F', 'CODE_GENDER_M',
       'NAME_INCOME_TYPE_Pensioner', 'NAME_INCOME_TYPE_Working',
       'NAME_EDUCATION_TYPE_Higher education',
       'NAME_EDUCATION_TYPE_Secondary / secondary special'],
      dtype='object')

      Index(['REG_CITY_NOT_WORK_CITY', 'FLAG_DOCUMENT_3', 'ext_source_1_missing',
       'CODE_GENDER_F', 'CODE_GENDER_M',
       'NAME_INCOME_TYPE_Commercial associate', 'NAME_INCOME_TYPE_Working',
       'NAME_EDUCATION_TYPE_Higher education',
       'NAME_EDUCATION_TYPE_Secondary / secondary special',
       'NAME_FAMILY_STATUS_Married',
       'ORGANIZATION_TYPE_Business Entity Type 3',
       'FONDKAPREMONT_MODE_reg oper account', 'HOUSETYPE_MODE_block of flats',
       'WALLSMATERIAL_MODE_Panel', 'WALLSMATERIAL_MODE_Stone, brick'],
      dtype='object')

      ROC-AUC for bureau add features:
      Train: 0.7592454270794223
      Test: 0.7554369598027679