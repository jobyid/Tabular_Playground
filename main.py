import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import compose
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import set_config
from sklearn import metrics
from sklearn import neighbors
from sklearn import tree
from sklearn import ensemble
from sklearn.svm import SVC, SVR
import xgboost as xgb
set_config(display='diagram') # Useful for display the pipeline

df = pd.read_csv('./tabular-playground-series-feb-2021/train.csv')
test_df = pd.read_csv('./tabular-playground-series-feb-2021/test.csv')
#print(df.head())
#print(df.info())
#print(df.columns)
# check for missing
#print("missing values :", df.isna().sum())
# Split target out from training data
y = df.target
x = df.iloc[:,:-1]
# identify the categorical variables and numerical features
cat_vars = ['cat0', 'cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7','cat8', 'cat9']
num_vars = ['cont0', 'cont1', 'cont2', 'cont3', 'cont4', 'cont5','cont6', 'cont7', 'cont8', 'cont9', 'cont10', 'cont11', 'cont12','cont13']
# make pipelines for both cat features and num features
cat_4_regres = pipeline.Pipeline(steps=[
    ('label', OrdinalEncoder())
])
num_4_regres = pipeline.Pipeline(steps=[
    ('scale', preprocessing.StandardScaler())
])

prep = compose.ColumnTransformer(transformers=[
    ('num', num_4_regres, num_vars),
    ('cat', cat_4_regres, cat_vars)
], remainder='drop')

X_train, X_val, y_train, y_val = train_test_split(x,y, test_size=0.2)

prep.fit(X_train)
X_train = prep.transform(X_train)
X_val   = prep.transform(X_val)

classifiers = {
    #"SVR":SVR(),
    "SGD reg": linear_model.SGDRegressor(),
    "Bayes Ridge": linear_model.BayesianRidge(),
    "lasso reg ":linear_model.LassoLars(),
    "ARD reg": linear_model.ARDRegression(),
    "Passive Aggressive ": linear_model.PassiveAggressiveRegressor(),
    "Theil Sen reg":linear_model.TheilSenRegressor(),
    "Lin reg": linear_model.LinearRegression(),
    "Decision Tree": tree.DecisionTreeRegressor(),
    #"Knn Reg": neighbors.KNeighborsRegressor(),
    "Random Forrest": ensemble.RandomForestRegressor(),
    #"XGBoost": xgb.XGBModel()
    }

results = pd.DataFrame({"Name":[], "Accuracy":[], "Balanced Accuracy":[]})

for name, model in classifiers.items():
    print("doing model ", name)
    model.fit(X_train, y_train)
    pred  = model.predict(X_val)
    results = results.append({"Name":name, "Accuracy":model.score(X_val, y_val)*100, "Balanced Accuracy":0},ignore_index=True)
    print(model.score(X_val, y_val))
print(results)
#pred_proba = model.predict_proba(X_val)
#print("Accuracy:          ", metrics.accuracy_score(y_val, pred)*100)
#print("Balanced accuracy: ", metrics.balanced_accuracy_score(y_val, pred)*100)
#print("Log loss:          ", metrics.log_loss(y_val, pred_proba))
#print("AUC:               ", metrics.roc_auc_score(y_val, pred_proba)*100) # Area Under ROC Curve
