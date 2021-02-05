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
import catboost as cb
import category_encoders as ce
from sklearn import decomposition
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn import neural_network
plt.rcParams['figure.dpi'] = 125
plt.rcParams['figure.figsize'] = (30,30)

df = pd.read_csv('./tabular-playground-series-feb-2021/train.csv')
x_df  = df.drop(['target'], axis=1)
y = df.target
print(df.head())
cat_vars = ['cat0', 'cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7','cat8', 'cat9']
num_vars = ['cont0', 'cont1', 'cont2', 'cont3', 'cont4', 'cont5','cont6', 'cont7', 'cont8', 'cont9', 'cont10', 'cont11', 'cont12','cont13']
enc = ce.OrdinalEncoder()
x_df = enc.fit_transform(x_df)
#plt.matshow(df.corr())
#plt.show()
x_cols = x_df.columns

def pca_work():
    pca = decomposition.PCA()
    comp = pca.fit_transform(x)
    sns.barplot(y=comp)
    plt.show()


def pair_plot():
    sns.pairplot(x.iloc[:, 11:].sample(10000), kind="hist")
    plt.savefig('pairs.png')
    plt.show()


def rfe_feature_extraction():
    model = linear_model.BayesianRidge()
    rfe = RFE(model, n_features_to_select=12)
    fit = rfe.fit(x_df, y)
    print("Num Features: %s" % (fit.n_features_))
    print("Selected Features: %s" % (fit.support_))
    print("Feature Ranking: %s" % (fit.ranking_))
    return fit.support_


# A helper method for pretty-printing the coefficients
def pretty_print_coefs(coefs, names = None, sort = False):
    if names == None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst,  key = lambda x:-np.abs(x[0]))
    return " + ".join("%s * %s" % (round(coef, 3), name)
                                   for coef, name in lst)


def ridge_feature_select():
    ridge = linear_model.Ridge(alpha=1.0)
    ridge.fit(x, y)
    print("Ridge model:", pretty_print_coefs(ridge.coef_))

top_eight = rfe_feature_extraction()
drop_feats = []
i=0
for x in top_eight:
   if x == False:
       drop_feats.append(x_cols[i])
   i+=1
print("Dropped Features: ", drop_feats)

drop_feats = np.array(drop_feats)
print(type(drop_feats))
x_f = x_df.drop(drop_feats, axis=1)

x_train, x_test, y_train, y_test = train_test_split(x_f,y)
x_train = x_train
x_test = x_test
nn = neural_network.MLPRegressor()
nn.fit(x_train,y_train)
score = nn.score(x_test,y_test)
print("NN score is: ", score)
