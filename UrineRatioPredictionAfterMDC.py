# importing functions
import sys
sys.dont_write_bytecode = True  # to Avoid .pyc Files
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize, minmax_scale, MaxAbsScaler, RobustScaler, Normalizer, QuantileTransformer, PowerTransformer
from sklearn.inspection import permutation_importance
import shap

# In[]: functions to separate features and targets; evaluate the reults; plot; 
def FeatureTarget (Data):
    X = Data[:,:-1]
    X = pd.DataFrame(X, columns=featureColumn)
    y = Data[:,-1].ravel()
    return X, y


def EvalResult (yTest, yPred):
    print('coefficient of determination r2: %.2f' % metrics.r2_score(yTest, yPred))
    print('Mean absolute error: %.2f' % metrics.mean_absolute_error(yTest, yPred))
    print('Root mean squared error: %.2f' % np.sqrt(metrics.mean_squared_error(yTest, yPred)))
    return


def PlotData (ytest, ypred, title_, ylabel):
    fig, ax = plt.subplots()
    ax.plot(ytest, 'b', label='Actual')
    ax.plot(ypred, 'r', label='Prediction')
    ax.set(xlabel='Samples', ylabel= ylabel)
    ax.set_title(title_)
    ax.legend()
    plt.show()
    return
    

# In[]: Import data
UrineMedAgeSexLab_AllGroups = []
for Group in range(4):
    UrineMedAgeSexLab_PerGroup = pd.read_csv('C:/Users/SaraA/Data/' + str(Group+1) + '.csv')
    UrineMedAgeSexLab_PerGroup['Group' + str(Group+1)] = 1
    UrineMedAgeSexLab_AllGroups.append(UrineMedAgeSexLab_PerGroup)
Prepared_UrineMedAgeSexLab = pd.concat(UrineMedAgeSexLab_AllGroups)
Prepared_UrineMedAgeSexLab['Group1'] = Prepared_UrineMedAgeSexLab['Group1'].fillna(0)
Prepared_UrineMedAgeSexLab['Group2'] = Prepared_UrineMedAgeSexLab['Group2'].fillna(0)
Prepared_UrineMedAgeSexLab['Group3'] = Prepared_UrineMedAgeSexLab['Group3'].fillna(0)
Prepared_UrineMedAgeSexLab['Group4'] = Prepared_UrineMedAgeSexLab['Group4'].fillna(0)
print(len(Prepared_UrineMedAgeSexLab))

# In[]: BMI
BMITempBP = pd.read_csv('C:/Users/SaraA/Data/PatientAdmitWeightBMITempBP.csv')
BMITempBP = BMITempBP.dropna(subset=['HeightTXT'])
BMITempBP = BMITempBP.reset_index(drop=True)
## Convert HeightTXT in feet and inches to centimeters
HeightNMB = []
for i in range(len(BMITempBP)):
    Height = ''.join((ch if ch in '0123456789.-e' else ' ') for ch in BMITempBP.loc[i]['HeightTXT'])
    Height = [float(i) for i in Height.split()]
    h_inch = Height[1]
    h_inch += Height[0] * 12
    h_cm = round(h_inch * 2.54, 1)
    HeightNMB.append(h_cm)
BMITempBP['HeightNMB'] = HeightNMB
BMITempBP = BMITempBP[['PatientEncounterID','BodyMassIndexNBR','HeightNMB']]
BMITempBP = BMITempBP.rename(columns={'PatientEncounterID': 'EncounterID'})
Prepared_UrineMedAgeSexLab = pd.merge(Prepared_UrineMedAgeSexLab, BMITempBP, on = 'EncounterID', how='left')
## calculate most recent BMI
Prepared_UrineMedAgeSexLab['BMI'] = Prepared_UrineMedAgeSexLab['Weight'] / (Prepared_UrineMedAgeSexLab['HeightNMB'] /100)**2

# In[]: Prepare data; drop missing values, reset index, add new columns
# rename Sex column
Prepared_UrineMedAgeSexLab = Prepared_UrineMedAgeSexLab.rename(columns={"Sex-Female": "Female"})
# # drop missing values, reset index
Prepared_UrineMedAgeSexLab = Prepared_UrineMedAgeSexLab.drop(['NTPROBNP'], axis=1)
Prepared_UrineMedAgeSexLab = Prepared_UrineMedAgeSexLab.dropna()
Prepared_UrineMedAgeSexLab = Prepared_UrineMedAgeSexLab.reset_index(drop=True)
# remove the weight numbers that are below 40 and above kg
Prepared_UrineMedAgeSexLab = Prepared_UrineMedAgeSexLab[Prepared_UrineMedAgeSexLab.Weight > 40]
Prepared_UrineMedAgeSexLab = Prepared_UrineMedAgeSexLab[Prepared_UrineMedAgeSexLab.Weight < 200]
Prepared_UrineMedAgeSexLab = Prepared_UrineMedAgeSexLab.reset_index(drop=True)

# In[]: Add a new column for Urine_0 / Urine_1b
Prepared_UrineMedAgeSexLab['UrineRatio_1'] = Prepared_UrineMedAgeSexLab['Urine_0'] / Prepared_UrineMedAgeSexLab['Urine_1b']
Prepared_UrineMedAgeSexLab['UrineRatio'] = Prepared_UrineMedAgeSexLab['Urine_1a'] / Prepared_UrineMedAgeSexLab['Urine_0']
Prepared_UrineMedAgeSexLab = Prepared_UrineMedAgeSexLab[Prepared_UrineMedAgeSexLab['UrineRatio_1'] != np.inf]
Prepared_UrineMedAgeSexLab = Prepared_UrineMedAgeSexLab[Prepared_UrineMedAgeSexLab['UrineRatio'] != np.inf]
Prepared_UrineMedAgeSexLab = Prepared_UrineMedAgeSexLab.reset_index(drop=True)
## Add new columns for Urine / BMI --------------------------------------------------------
Prepared_UrineMedAgeSexLab['Urine_2bR'] = Prepared_UrineMedAgeSexLab['Urine_2b'] / Prepared_UrineMedAgeSexLab['BMI']
Prepared_UrineMedAgeSexLab['Urine_1bR'] = Prepared_UrineMedAgeSexLab['Urine_1b'] / Prepared_UrineMedAgeSexLab['BMI']
Prepared_UrineMedAgeSexLab['Urine_0R'] = Prepared_UrineMedAgeSexLab['Urine_0'] / Prepared_UrineMedAgeSexLab['BMI']
Prepared_UrineMedAgeSexLab['Urine_1aR'] = Prepared_UrineMedAgeSexLab['Urine_1a'] / Prepared_UrineMedAgeSexLab['BMI']

# In[]: Adding new columns to create 6-hour time bins
Prepared_UrineMedAgeSexLab['time_12'] =((Prepared_UrineMedAgeSexLab['time_1'] + Prepared_UrineMedAgeSexLab['time_2']) >=1).astype(int)
Prepared_UrineMedAgeSexLab['time_34'] =((Prepared_UrineMedAgeSexLab['time_3'] + Prepared_UrineMedAgeSexLab['time_4']) >=1).astype(int)
Prepared_UrineMedAgeSexLab['time_56'] =((Prepared_UrineMedAgeSexLab['time_5'] + Prepared_UrineMedAgeSexLab['time_6']) >=1).astype(int)
Prepared_UrineMedAgeSexLab['time_78'] =((Prepared_UrineMedAgeSexLab['time_7'] + Prepared_UrineMedAgeSexLab['time_8']) >=1).astype(int)

# In[]: Encounter DiagnosisNMs data
EncounterDiagnosisICD10_ = pd.read_csv('C:/Users/SaraA/Data/ADTDiagnosisNM.csv')
EncounterDiagnosisICD10_ = EncounterDiagnosisICD10_.rename(columns={'PatientEncounterID': 'EncounterID'})
ICD10_HeartFailure = ['I501','I5020','I5021','I5022','I5023','I5030','I5031','I5032','I5033','I5040','I5041','I5042','I5043','I509','B572','I110','T17318A','E851']
EncounterDiagnosisICD10_['HeartFailure'] = EncounterDiagnosisICD10_.ICD10.isin(ICD10_HeartFailure)
EncounterDiagnosisICD10_['HeartFailure'][EncounterDiagnosisICD10_.HeartFailure == True] = 1
EncounterDiagnosisICD10_['HeartFailure'][EncounterDiagnosisICD10_.HeartFailure == False] = 0

ICD10_AcuteKidney  = ['N170','N179']
EncounterDiagnosisICD10_['AcuteKidney'] = EncounterDiagnosisICD10_.ICD10.isin(ICD10_AcuteKidney)
EncounterDiagnosisICD10_['AcuteKidney'][EncounterDiagnosisICD10_.AcuteKidney == True] = 1
EncounterDiagnosisICD10_['AcuteKidney'][EncounterDiagnosisICD10_.AcuteKidney == False] = 0

ICD10_ChronicKidney = ['I129','N182','N183','N184','N185','N186','N189','N19','T17318A','T17328A','Z9911','Z992']
EncounterDiagnosisICD10_['ChronicKidney'] = EncounterDiagnosisICD10_.ICD10.isin(ICD10_ChronicKidney)
EncounterDiagnosisICD10_['ChronicKidney'][EncounterDiagnosisICD10_.ChronicKidney == True] = 1
EncounterDiagnosisICD10_['ChronicKidney'][EncounterDiagnosisICD10_.ChronicKidney == False] = 0

ICD10_Cardiomyopathy = ['Q245','I428','I425','I255','I422','I421','I5181','I509','NoDx','I426']
EncounterDiagnosisICD10_['Cardiomyopathy'] = EncounterDiagnosisICD10_.ICD10.isin(ICD10_Cardiomyopathy)
EncounterDiagnosisICD10_['Cardiomyopathy'][EncounterDiagnosisICD10_.Cardiomyopathy == True] = 1
EncounterDiagnosisICD10_['Cardiomyopathy'][EncounterDiagnosisICD10_.Cardiomyopathy == False] = 0

# In[]: merge multiple records of individials into one record
EncounterDiagnosisNMs_1 = EncounterDiagnosisICD10_[['EncounterID','HeartFailure','AcuteKidney', 'ChronicKidney','Cardiomyopathy']]
EncounterDiagnosisNMs_2 = EncounterDiagnosisNMs_1.groupby('EncounterID')['HeartFailure','AcuteKidney', 'ChronicKidney','Cardiomyopathy'].sum().astype(int)
EncounterDiagnosisNMs_2[EncounterDiagnosisNMs_2 != 0] = 1 # if more than one record, set it to 1
EncounterDiagnosisNMs_2 = EncounterDiagnosisNMs_2.reset_index()
# # merge urine/medication data and diagnoses data
Prepared_UrineMedAgeSexLab = pd.merge(Prepared_UrineMedAgeSexLab, EncounterDiagnosisNMs_2, on = 'EncounterID', how='left')
Prepared_UrineMedAgeSexLab = Prepared_UrineMedAgeSexLab.fillna(0)

# In[]: removing duplicated
Data_EncounterID = Prepared_UrineMedAgeSexLab[['EncounterID','Age','HeartFailure','AcuteKidney', 'ChronicKidney','Cardiomyopathy']].drop_duplicates()

# In[]: renaming column names
Prepared_UrineMedAgeSexLab = Prepared_UrineMedAgeSexLab.rename(
    columns={"MedDose": "Medication Dose",
             'Urine_1b': 'Urine 2 hours before medication',
             'Urine_0': 'Urine 1 hour before medication',
             'Female': 'Sex',
             'HeartFailure': 'Heart Failure',
             'AcuteKidney':  'Acute Kidney Disease',
             'ChronicKidney': 'Chronic Kidney Disease',
             'Group1': 'Acute Care Cardiac Unit',
             'Group2': 'Cardiac ICU',
             'Group3': 'Cardiac Surgery',
             'Group4': 'Non-cardiac ICU',
             'time_12': '00-06H',
             'time_34': '06-12H',
             'time_56': '12-18H',
             'time_78': '18-00H'
             })

# In[]: split test and train randomly in a way that no same subject is in both testing and training sets
unique_users = Prepared_UrineMedAgeSexLab['MRN'].unique()
target_n = int(.7 * len(Prepared_UrineMedAgeSexLab))
shuffled_users = np.random.RandomState(seed=42).permutation(unique_users)
user_count = Prepared_UrineMedAgeSexLab['MRN'].value_counts()
mapping = user_count.reindex(shuffled_users).cumsum() <= target_n
mask = Prepared_UrineMedAgeSexLab['MRN'].map(mapping)
TrainingSet = Prepared_UrineMedAgeSexLab[mask]
TrainingSet = TrainingSet.reset_index(drop=True)
TestingSet = Prepared_UrineMedAgeSexLab[~mask]
TestingSet = TestingSet.reset_index(drop=True)

# In[]: split data into features and target
featureColumn = ['Age',
                 'Weight',
                 'Medication Dose',
                 'Creatinine',
                 'UrineRatio_1',
                 'Sex',
                 'Heart Failure','Acute Kidney Disease','Chronic Kidney Disease','Cardiomyopathy',
                 'Acute Care Cardiac Unit','Cardiac ICU','Cardiac Surgery','Non-cardiac ICU',
                 '00-06H','06-12H','12-18H','18-00H']
TargetColumn = ['UrineRatio'] 
FinalColumn = featureColumn + TargetColumn

# In[]: scaling data
X = Prepared_UrineMedAgeSexLab[featureColumn]
y = Prepared_UrineMedAgeSexLab[TargetColumn].values.ravel()

### If not notmalizing the data at all, use the 4 lins below-----------------
X_train = TrainingSet[featureColumn]
y_train = TrainingSet[TargetColumn].values.ravel()
X_test = TestingSet[featureColumn]
y_test = TestingSet[TargetColumn].values.ravel()

featureColumn = np.reshape(featureColumn, (-1, 1))

### if normalizing the data use the lines below------------------------------
# scaler = Normalizer()  # Normalization  MinMaxScaler
# scaler = scaler.fit(TrainingSet[FinalColumn])
# TrainingSet_Scaled = scaler.transform(TrainingSet[FinalColumn])
# TestingSet_Scaled = scaler.transform(TestingSet[FinalColumn])
# ### split features and target------------------------------------------------
# X_train, y_train = FeatureTarget (TrainingSet_Scaled)
# X_test, y_test = FeatureTarget (TestingSet_Scaled)

# In[]: assigning weights to the samples since some individuals are represented by more samples than others
Count_IDs = TrainingSet['EncounterID'].value_counts().reset_index(drop=False).rename(columns={"index": "EncounterID", "EncounterID": "Counts"})
sample_weight = pd.DataFrame(np.ones((len(TrainingSet), 1)),columns=['Weights'])
sample_weight['EncounterID'] =  TrainingSet['EncounterID']
for i in range(len(Count_IDs)):
    sample_weight['Weights'][sample_weight.EncounterID == Count_IDs['EncounterID'][i]] = 1/Count_IDs['Counts'][i]
sample_weight = sample_weight[['Weights']].values.ravel()

# In[]: xgboost (extreme gradient boosting)
from xgboost import XGBRegressor
model_xgb = XGBRegressor(max_depth=4,
                         n_estimators=100,
                         eta=0.1, 
                         # subsample=0.7, 
                          colsample_bytree=0.85,
                          colsample_bylevel=0.85,
                          colsample_bynode=0.85)
model_xgb_fitted = model_xgb.fit(X_train, y_train, sample_weight=sample_weight)
y_Trpred_xgb = model_xgb_fitted.predict(X_train)
y_pred_xgb = model_xgb_fitted.predict(X_test)
# get Feature Importances
FeatureImportances_xgb = np.reshape(model_xgb_fitted.feature_importances_, (-1, 1))
importance_xgb = permutation_importance(model_xgb_fitted, X_train, y_train, scoring='neg_mean_squared_error').importances_mean
importance_xgb = np.reshape(importance_xgb, (-1, 1))
FeatureImportances_xgb = np.concatenate((featureColumn, FeatureImportances_xgb,importance_xgb), axis=1)
# Validating the model and ploting the outpot
EvalResult(y_train, y_Trpred_xgb)
EvalResult(y_test, y_pred_xgb)
PlotData (y_test[:300], y_pred_xgb[:300], 'extreme gradient boosting', TargetColumn[0])

# In[]: Linear Regression model from scikit-learn library
# print('----------------------------------------Linear Regression')
# from sklearn.linear_model import LinearRegression
# model_LR = LinearRegression().fit(X_train, y_train, sample_weight=sample_weight)
# y_pred_LR = model_LR.predict(X_test) # Make predictions using the testing set
# coefficient = model_LR.coef_
# print('Regression coefficient:', model_LR.coef_)
# print('Regression intercept:', model_LR.intercept_)
# print('')
# # f test to find significant variables
# from sklearn.feature_selection import f_regression
# FeatureImportances_LR = f_regression(X_train, y_train, center=True)
# FeatureImportances_LR = pd.DataFrame(FeatureImportances_LR).T
# FeatureImportances_LR.columns = ['F-Value', 'P-Value']
# FeatureImportances_LR['coefficient'] = coefficient.T
# FeatureImportances_LR['Variables'] = featureColumn
# FeatureImportances_LR = FeatureImportances_LR[['Variables', 'coefficient', 'F-Value', 'P-Value']]
# EvalResult (y_test, y_pred_LR) 

# In[]: Decision Tree Regressorn
# from sklearn.tree import DecisionTreeRegressor
# print('----------------------------------------Decision Tree')
# model_DT = DecisionTreeRegressor(max_depth=2).fit(X_train, y_train, sample_weight=sample_weight)
# y_pred_DT = model_DT.predict(X_test)
# EvalResult (y_test, y_pred_DT)

# In[]: Random Forest Regressor
# print('---------------------------------------Random Forest')
# from sklearn.ensemble import RandomForestRegressor
# model_RF = RandomForestRegressor(max_depth=2, random_state=0).fit(X_train, y_train, sample_weight=sample_weight)
# y_pred_RF = model_RF.predict(X_test)
# EvalResult (y_test, y_pred_RF)

# In[]: SHAP TreeExplainer
print('---------------------------------------xgboost-whole dataset for training')
model_xgb = model_xgb.fit(X, y, sample_weight=sample_weight)
y_pred = model_xgb.predict(X)
EvalResult(y, y_pred)

explainer = shap.TreeExplainer(model_xgb)
shap_values = explainer.shap_values(X)
shap_values_ = explainer(X)

# In[]: SHAP plots
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X, show=False)  # SHAP Summary Plot
ax.set_title("XGB to predict urine output at the time of medication administration in ")

fig, ax = plt.subplots()
shap.summary_plot(shap_values, X, plot_type="bar", color='green', show=False)  # SHAP Feature importance plot

fig, ax = plt.subplots()
plt.scatter(X['Age'], shap_values_[:,'Age'].values, color='blue',  s = 10)
plt.axhline(y=0, color="red", linestyle=":")
# plt.ylim(-50, 100)
plt.xlabel("Age (years)")
plt.ylabel("SHAP Value for Age")
plt.show()

plt.scatter(X['Weight'], shap_values_[:,'Weight'].values, color='blue',  s = 10)
plt.axhline(y=0, color="red", linestyle=":")
# plt.ylim(-75, 150)
plt.xlabel("Weight (kg)")
plt.ylabel("SHAP Value for Weight")
plt.show()

plt.scatter(X['Medication Dose'], shap_values_[:,'Medication Dose'].values, color='blue',  s = 10)
plt.axhline(y=0, color="red", linestyle=":")
# plt.ylim(-10, 10)
plt.xlabel("Medication Dose (mg)")
plt.ylabel("SHAP Value for Medication Dose")
plt.show()

plt.scatter(X['Creatinine'], shap_values_[:,'Creatinine'].values, color='blue',  s = 10)
plt.axhline(y=0, color="red", linestyle=":")
# plt.ylim(-10, 10)
plt.xlabel("Creatinine (mg/dL)")
plt.ylabel("SHAP Value for Creatinine")
plt.show()

# shap.plots.scatter(shap_values_[:,'Age'])
# shap.plots.scatter(shap_values_[:,'Weight'])
# shap.plots.scatter(shap_values_[:,'Medication Dose'])
# shap.plots.scatter(shap_values_[:,'Creatinine'])
# shap.plots.scatter(shap_values_[:,'BloodPressureSystolicNBR'])
# shap.plots.scatter(shap_values_[:,'BloodPressureDiastolicNBR'])
# shap.plots.scatter(shap_values_[:,'TemperatureFahrenheitNBR'])
# shap.plots.scatter(shap_values_[:,'HeartRateNBR'])
# shap.plots.scatter(shap_values_[:,'time_cos'])

# shap.dependence_plot('Creatinine', shap_values, X, interaction_index='Medication Dose', show=False)

# shap.dependence_plot('00-06H', shap_values, X, interaction_index='Heart Failure', show=False)
# labels = ['', 0, '', 1, '']
# plt.xticks(np.arange(-0.5, 2, step=0.5),labels)
# shap.dependence_plot('00-06H', shap_values, X, interaction_index='Acute Kidney Disease', show=False)
# labels = ['', 0, '', 1, '']
# plt.xticks(np.arange(-0.5, 2, step=0.5),labels)
# shap.dependence_plot('00-06H', shap_values, X, interaction_index='Chronic Kidney Disease', show=False)
# labels = ['', 0, '', 1, '']
# plt.xticks(np.arange(-0.5, 2, step=0.5),labels)
# shap.dependence_plot('00-06H', shap_values, X, interaction_index='Cardiomyopathy', show=False)
# labels = ['', 0, '', 1, '']
# plt.xticks(np.arange(-0.5, 2, step=0.5),labels)

# ## SHAP interaction plot
# shap_interaction = explainer.shap_interaction_values(X)
# shap.summary_plot(shap_interaction, X, max_display=20, show=False)

# In[]: post processing on shap values    
import seaborn as sns
data_boxplot = []
lenBoxplotData = shap_values.shape[1]
for w in range(lenBoxplotData-4):  # 6 with BNP
    data_boxplot.append([])
Count = 0
for SHAPCol in range(shap_values.shape[1]):
    if SHAPCol != 0:
        if SHAPCol in range(5,6,1): # 7,8 with BNP
            print(SHAPCol)
            shapValues_Data = np.concatenate((np.reshape(shap_values[:,SHAPCol], (-1, 1)), np.reshape(np.array(X[featureColumn[SHAPCol,0]]), (-1, 1))), axis=1)
            shapValues_Data_red = shapValues_Data[shapValues_Data[:,1] == 1]
            shapValues_Data_blue = shapValues_Data[shapValues_Data[:,1] == 0]
            data_boxplot[Count].append(shapValues_Data_red[:,0])
            Count = Count + 1
            data_boxplot[Count].append(shapValues_Data_blue[:,0])
            Count = Count + 1
        if SHAPCol in range(6,lenBoxplotData,1): # 8 with BNP
            print('second')
            print(SHAPCol)
            shapValues_Data = np.concatenate((np.reshape(shap_values[:,SHAPCol], (-1, 1)), np.reshape(np.array(X[featureColumn[SHAPCol,0]]), (-1, 1))), axis=1)
            shapValues_Data_red = shapValues_Data[shapValues_Data[:,1] == 1]
            data_boxplot[Count].append(shapValues_Data_red[:,0])
            Count = Count + 1
fig, ax = plt.subplots(figsize=(25,25))
ax = sns.boxplot(data=data_boxplot, orient='h', width=1, fliersize=4, linewidth = 4, showfliers = False)
plt.axvline(x=0, color="black", linestyle=":")
ax.set_yticklabels(['Female','Male',
                    'Heart Failure',
                    'Acute Kidney Disease',
                    'Chronic Kidney Disease',
                    'Cardiomyopathy',
                    'Acute Care Cardiac Unit',
                    'Cardiac ICU',
                    'Cardiac Surgery',
                    'Non-cardiac ICU',
                    '00-06H',
                    '06-12H',
                    '12-18H',
                    '18-00H'
                    ])
plt.xlabel("SHAP Values",fontsize = 35)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
# plt.xlim(-25, 30)
# plt.xlim(-5, 5)
