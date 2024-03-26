import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
#from sklearn import...

#import dataset
#clean dataset - renaming, duplicates, dropping columns, dropping NaN values, converting column entries to groups / binary, rounding values, converting value ranges to string groups,
#data viz - find insightful column relationships, key predictors of defaults
#feature engineering - correlation test to see if columns can be removed, distribution analysis of each columns values (checking if logarithmic scaling is necessary)
#some columns can be removed if they show default stability across all loan groups
#data preprocessing - converting categorical values into numerical values, further column removal for simplicity
#cross-validation
#fit model
#test model
#report analytics

# 1 = default, 0 = non-default
#try various models: RF, DT, LGR, LGR built from scratch

df = pd.read_csv("train.csv")

df.drop(['ID','Batch Enrolled','Revolving Utilities','Application Type'],axis=1,inplace=True) # removing unnecessary columns
df.rename({"Employment Duration":"HomeOwnership", "Total Current Balance":"Balance", "Loan Status":"Defaulted",'Loan Amount':"Amount"},axis=1, inplace=True)
df.rename({"Home Ownership":"Salary"},axis=1, inplace=True) # renaming columns for readability/clarity
#would drop NaNs but all data is valid
df.loc[df['Verification Status']=='Source Verified','Verification Status']= 'Verified' # make this column binary
df['Interest Rate']=np.ceil(df['Interest Rate']) # rounding values for convenience
def amount_map(amount):
    if amount<15000:
        return 'Small'
    if amount<25000:
        return 'Medium'
    else: return 'Big'
df['Loan Category']= df.Amount.map(amount_map) # converting value ranges to string groups

#sns.histplot(x='Grade', hue="Grade",data=df)
#plt.title("Number of loans based on the Grade")
#plt.ylabel('Frequency')
#plt.show()
#could add more viz here to see how each loan grade and interest rate affects ratio of defaults in dataset group
#also could home ownership status, size of loan etc. affect default rate

#Now we run a correlation test to see if any columns can sensibly be removed
corr = df[['Amount','Funded Amount',"Funded Amount Investor",'Term','Interest Rate','Salary','Debit to Income'
          ,'Delinquency - two years',"Inquires - six months","Open Account","Public Record",'Revolving Balance'
          ,'Total Accounts','Total Received Interest',"Total Received Late Fee",'Recoveries',
          'Collection Recovery Fee','Last week Pay',"Total Collection Amount","Balance",
          "Total Revolving Credit Limit"]].corr()
plt.figure(figsize=(12,12))
sns.heatmap(corr, cmap='Blues', annot=True, annot_kws={"size": 5}) # some fields without numerical data omitted
#plt.show() # we see no correlations - all fields hold value
#plt.title('Feature Correlation Heatmap', fontsize=16)
plt.savefig('correlation_heatmap.png', bbox_inches='tight', dpi=300, orientation='portrait')
plt.close()

# some columns naturally have normal distributions - ideal as this fits to base assumption of GLMs
# for other columns we must apply log scaling and make them more normal


df_integ=df.drop(['Grade','Sub Grade','HomeOwnership','Verification Status','Payment Plan','Loan Title','Initial List Status',
                 'Loan Category'],axis=1)
columns=['Amount','Funded Amount',"Funded Amount Investor",'Term','Interest Rate','Salary','Debit to Income'
          ,'Delinquency - two years',"Inquires - six months","Open Account","Public Record",'Revolving Balance'
          ,'Total Accounts','Total Received Interest',"Total Received Late Fee",'Recoveries',
          'Collection Recovery Fee','Last week Pay','Accounts Delinquent',"Total Collection Amount","Balance",
          "Total Revolving Credit Limit"]

normal_columns = ['Amount', 'Debit to Income', 'Last week Pay']
log_columns= ['Funded Amount Investor', 'Interest Rate', 'Salary', "Inquires - six months", 'Open Account', 
              'Revolving Balance','Total Accounts', 'Total Received Interest', 'Total Received Late Fee', 
              'Recoveries', 'Collection Recovery Fee', 'Total Collection Amount']

for i in log_columns:
    df_integ[i]=np.log10(df_integ[i])

drop_columns=['Funded Amount','Term','Delinquency - two years',"Public Record",'Accounts Delinquent',"Total Revolving Credit Limit"]
df.drop(drop_columns, axis=1,inplace=True)

numeric_columns=['Amount', 'Debit to Income', 'Last week Pay','Funded Amount Investor', 'Interest Rate', 'Salary', "Inquires - six months", 'Open Account', 
              'Revolving Balance','Total Accounts', 'Total Received Interest', 'Total Received Late Fee', 
              'Recoveries', 'Collection Recovery Fee', 'Total Collection Amount']

for i in numeric_columns:
    df[i]=df_integ[i]  # basically changing the original df so the non-normal columns are now log-scaled

str_columns=['Grade','Sub Grade','HomeOwnership','Verification Status','Payment Plan','Loan Title','Initial List Status'] # we need everything to be numeric so we must deal with this

df.drop(['Loan Title','Payment Plan','Loan Category'],axis=1, inplace=True) # unnecessary columns - loan category similar to amount, 

#Data Preprocessing  - convert categorical --> numerical
df.drop('Collection 12 months Medical',axis=1,inplace=True) # not necessary either

categorical_variables=['Grade','Sub Grade',"HomeOwnership",'Initial List Status','Verification Status']
df_encode=pd.get_dummies(data=df, columns=categorical_variables,prefix='Col',drop_first=True,prefix_sep="_", dtype='int8')

df_encode

#df_encode.info() # we have now reshaped the df with encoded categories

from scipy.stats import boxcox  # Balance column is still not normal - using classic Boxcox technique to normalise
y_bc,lam, ci= boxcox(df_encode['Balance'],alpha=0.05)

#ci,lam

df_encode['Balance']= np.log(df_encode['Balance']) 

df_encode.drop(['Inquires - six months','Revolving Balance'],axis=1,inplace=True)

#df_encode.info()

# Now we can move on to final steps - default status is binary so Logistic Regression is valid - Decision Tree / Random Forest are also applicable

#-------------------------------------------------------------------------------------------------------------------

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


X= df_encode.drop('Defaulted', axis=1)
y=df_encode['Defaulted']

from imblearn.over_sampling import SMOTE

smote=SMOTE()    # could use Smote to address imbalancing as there are very few defaults:non-defaults
smote.fit(X,y)   # functions to generate representative new samples of default/underrepresented cases
X,y=smote.fit_resample(X,y)

# ^ This is VITAL! Logistic regression simply predicts all non-defaults otherwise
# even after smote LR only gives 50% accuracy and offers no positive predictions... we need CV

# increase normality vs normalising?? is this a technical difference between making something gaussian and scaling to 0 to 1?

scaler = MinMaxScaler()  # scales/normalises each column to 0 to 1 - gives varying absolute value differences in each column equal weighting
x_scaled = scaler.fit_transform(X)


from LogReg import LogisticReg
from sklearn.ensemble import RandomForestClassifier
custom_lgr = LogisticReg()
lgr = LogisticRegression(max_iter=500) # select built-in logistic regression model
tr = tree.DecisionTreeClassifier() # select built-in decision tree model
rfc = RandomForestClassifier( random_state=42) # built in random forest


"""
from sklearn.metrics import roc_auc_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=40)

lst_accu_stratified_lgr = []
lst_accu_stratified_tr = []
lst_aucroc_lgr = []

for train_index, test_index in skf.split(x_scaled, y):
    # Splitting Data
    x_train_fold, x_test_fold = x_scaled[train_index], x_scaled[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]

    # Fitting Models
    lgr.fit(x_train_fold, y_train_fold)
    tr.fit(x_train_fold, y_train_fold)

    # Making Predictions & Evaluating
    y_pred_fold_lgr = lgr.predict_proba(x_test_fold)[:, 1]  # For ROC-AUC Score
    
    # Calculate AUC-ROC Score for Logistic Regression
    aucroc_lgr = roc_auc_score(y_test_fold, y_pred_fold_lgr)
    lst_aucroc_lgr.append(aucroc_lgr)
    
    # Predicting and calculating accuracy for each fold for both models
    y_pred_lgr = lgr.predict(x_test_fold)
    y_pred_tr = tr.predict(x_test_fold)

    # Append accuracies
    lst_accu_stratified_lgr.append(accuracy_score(y_test_fold, y_pred_lgr))
    lst_accu_stratified_tr.append(accuracy_score(y_test_fold, y_pred_tr))

print(f"Average AUC-ROC Score - Logistic Regression: {np.mean(lst_aucroc_lgr)}")
# Average accuracies
print(f"Average accuracy - Logistic Regression: {np.mean(lst_accu_stratified_lgr)}")
print(f"Average accuracy - Decision Tree: {np.mean(lst_accu_stratified_tr)}")

"""

#use scaled X below
X_train, X_test, Y_train, Y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

lgr.fit(X_train, Y_train) # SKLearn Logistic Regression
tr.fit(X_train, Y_train) # SKLearn Decision Tree
custom_lgr.fit(X_train, Y_train) # Custom Logistic Regression
rfc.fit(X_train, Y_train)

# now add random forest

y_pred_lgr = lgr.predict(X_test)
y_pred_tr = tr.predict(X_test)
y_prob_lgr = lgr.predict_proba(X_test)[:, 1]
y_prob_tr = tr.predict_proba(X_test)[:, 1]
y_pred_rf = rfc.predict(X_test)
y_prob_rf = rfc.predict_proba(X_test)[:, 1] 

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
# I want Accuracy, Precision, Recall, F1, AUC, for LR, Custom LR, Tree, Forest
Models = ['Decision Tree (SKLearn)', 'Random Forest (SKLearn)', 'Logistic Regression (SKLearn)', 'Custom Logistic Regression']
Metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
Stats = np.zeros((len(Models),5))
Stats[0][0] = round(accuracy_score(Y_test, y_pred_tr),3)
Stats[0][1]  = round(precision_score(Y_test, y_pred_tr),3)
Stats[0][2] = round(recall_score(Y_test, y_pred_tr),3)
Stats[0][3] = round(f1_score(Y_test, y_pred_tr),3)
Stats[0][4] = round(roc_auc_score(Y_test, y_prob_tr),3)
Stats[1][0] = round(accuracy_score(Y_test, y_pred_rf),3)
Stats[1][1] = round(precision_score(Y_test, y_pred_rf),3)
Stats[1][2] = round(recall_score(Y_test, y_pred_rf),3)
Stats[1][3] = round(f1_score(Y_test, y_pred_rf),3)
Stats[1][4] = round(roc_auc_score(Y_test, y_prob_rf),3)
Stats[2][0] = round(accuracy_score(Y_test, y_pred_lgr),3)
Stats[2][1]  = round(precision_score(Y_test, y_pred_lgr),3)
Stats[2][2] = round(recall_score(Y_test, y_pred_lgr),3)
Stats[2][3] = round(f1_score(Y_test, y_pred_lgr),3)
Stats[2][4] = round(roc_auc_score(Y_test, y_prob_lgr),3)
Stats[3][0] = round(custom_lgr.stats(X_test,Y_test)[0],3)
Stats[3][1] = round(custom_lgr.stats(X_test,Y_test)[1],3)
Stats[3][2] = round(custom_lgr.stats(X_test,Y_test)[2],3)
Stats[3][3] = round(custom_lgr.stats(X_test,Y_test)[3],3)
Stats[3][4] = round(custom_lgr.stats(X_test,Y_test)[4],3)

#print(Stats)

df_stats = pd.DataFrame(Stats, index=Models, columns=Metrics)
print(df_stats)

fig, ax = plt.subplots(figsize=(8,8))  # Adjust size as needed
ax.axis('tight')
ax.axis('off')

# Create the table and add it to the plot
the_table = ax.table(cellText=df_stats.values,
                     colLabels=df_stats.columns,
                     rowLabels=df_stats.index,
                     cellLoc='center', 
                     loc='center',
                     colColours=["navy"]*df_stats.shape[1], 
                     rowColours=["navy"]*df_stats.shape[0])

the_table.auto_set_font_size(False)  # Disable automatic font size setting
the_table.set_fontsize(10)  # Set a smaller font size
the_table.scale(1.5, 1.5)  # You can scale the table for better fit
for key, cell in the_table.get_celld().items():
    if key[0] == 0 or key[1] < 0:  # Column labels and row labels
        cell.get_text().set_color('white')

#plt.title('Model Performance Metrics', fontsize=16, pad=20)
plt.savefig('model_performance_table.png', bbox_inches='tight', dpi=300, orientation='portrait')
plt.close()

#generate metrics table
#generate image of interest rate vs default rate
#generate image of loan handover

# Neglect cross-validation
# I want Accuracy, Precision, Recall, F1, AUC, for LR, Custom LR, Tree, Forest
# Display this in a table
# Image of a decision tree picking default or non-default would also be nice and a list of factors included
# maybe image of features or 1 key EDA graph?

# Why does custom LR underperform? It uses fundamental gradient descent, sklearn uses more complex, quicker, optimal convergence methods - also uses regularization
# sklearn also uses anti-numerical instability methods (arising in sigmoid), and includes hyperparameter tuning

# Positive = Default, Negative = Non-Default (Far more common)
# We are most interested in recall (true positive rate)
# For original unbalanced dataset 90% accuracy can be achieved by simply predicting non-default on every sample

# In future we will add cross-validation