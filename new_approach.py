import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings("ignore")
# https://www.kaggle.com/jonas0/beginner-friendly-february-tabular-tutorial/
train_data = pd.read_csv('tabular-playground-series-feb-2021/train.csv')
test_data  = pd.read_csv('tabular-playground-series-feb-2021/test.csv')
cat_features = ['cat0', 'cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 'cat8', 'cat9']
numerical_features = ['cont0', 'cont1', 'cont2', 'cont3', 'cont4', 'cont5','cont6', 'cont7', 'cont8', 'cont9', 'cont10', 'cont11', 'cont12', 'cont13']


def num_plots():
    global i, fig, ax
    for i in numerical_features:
        fig, ax = plt.subplots(1, 2, figsize=(18, 6))

        ax[0].plot(train_data["id"], train_data[i])
        ax[1].plot(test_data["id"], test_data[i])

        ax[0].set(xlabel="id", ylabel=i)
        ax[0].set_title('train_data')

        ax[1].set(xlabel="id", ylabel=i)
        ax[1].set_title("test_data")

        plt.show()


def num_hist_plot():
    global i, fig, ax
    for i in numerical_features:
        fig, ax = plt.subplots(1, 2, figsize=(18, 6))

        sns.distplot(a=train_data[i], ax=ax[0])
        ax[0].set(xlabel='id', ylabel=i)
        ax[0].set_title('train_data')

        ax[1].set(xlabel='id', ylabel=i)
        ax[1].set_title("test_data")
        sns.distplot(a=test_data[i], ax=ax[1])
        plt.show()


def feature_vs_target_plots():
    global i, fig
    for i in numerical_features:
        fig = plt.figure(figsize=(10, 6))
        plt.plot(train_data[i], train_data["target"], linestyle='', marker='x')
        plt.title(i)
        plt.show()


def outliers():
    outlier = train_data.loc[train_data.target < 1.0]
    print(outlier, "\n")
    print(outlier.index)
    # remove the outlier from the train_data set
    train_data.drop([99682], inplace=True)


outliers()


def cat_bar_plots():
    global i, fig, ax
    for i in cat_features:
        fig, ax = plt.subplots(1, 2, figsize=(18, 6))

        train_data[i].value_counts().plot(kind='bar', ax=ax[0])
        ax[0].set(xlabel='id', ylabel=i)
        ax[0].set_title('train_data')

        ax[1].set(xlabel='id', ylabel=i)
        ax[1].set_title("test_data")
        test_data[i].value_counts().plot(kind='bar', ax=ax[1])

        plt.show()


def cat_catplots():
    global i
    for i in cat_features:
        sns.catplot(x=i, y="target", data=train_data)
        plt.show()

y_train = train_data["target"]
train_data.drop(columns=['target'], inplace=True)
test_data_backup = test_data.copy()

def save_solution(y_pred):
    solution = pd.DataFrame({"id": test_data_backup.id, "target": y_pred})
    solution.to_csv("solution.csv", index=False)
    print("saved successful!")


def train_catboost():
    categorical_features = cat_features
    # dropping the id column slightly improves the score
    train_data.drop(columns=["id"], inplace=True)
    test_data.drop(columns=["id"], inplace=True)
    model_ctb = CatBoostRegressor(iterations=3000, learning_rate=0.02, od_type='Iter', loss_function='RMSE',  # eval_metric='AUC',
                                  grow_policy='SymmetricTree',  # auto_class_weights = 'Balanced',
                                  # max_depth = 8,
                                  subsample=0.8,  # colsample_bylevel = 0.9,
                                  # l2_leaf_reg = 0.80,
                                  # one_hot_max_size = 4,
                                  verbose=30, random_seed=17)

    model_ctb.fit(train_data, y_train, cat_features=categorical_features)

    y_pred = model_ctb.predict(test_data)
    save_solution(y_pred)
    print(y_pred)

train_catboost()




