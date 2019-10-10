import numpy as np;
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

# =====================
def try_different_method(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    result = model.predict(x_test)
    return r2_score(y_test, result)


###########3.具体方法选择##########
####3.1决策树回归####
from sklearn import tree
model_DecisionTreeRegressor = tree.DecisionTreeRegressor()
####3.2线性回归####
from sklearn import linear_model
model_LinearRegression = linear_model.LinearRegression()
####3.3SVM回归####
from sklearn import svm
model_SVR = svm.SVR()
####3.4KNN回归####
from sklearn import neighbors
model_KNeighborsRegressor = neighbors.KNeighborsRegressor()
####3.5随机森林回归####
from sklearn import ensemble
model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=20)#这里使用20个决策树
####3.6Adaboost回归####
from sklearn import ensemble
model_AdaBoostRegressor = ensemble.AdaBoostRegressor(n_estimators=50)#这里使用50个决策树
####3.7GBRT回归####
from sklearn import ensemble
model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators=100)#这里使用100个决策树
####3.8Bagging回归####
from sklearn.ensemble import BaggingRegressor
model_BaggingRegressor = BaggingRegressor()
####3.9ExtraTree极端随机树回归####
from sklearn.tree import ExtraTreeRegressor
model_ExtraTreeRegressor = ExtraTreeRegressor()

models = [model_AdaBoostRegressor, model_BaggingRegressor, model_DecisionTreeRegressor, model_ExtraTreeRegressor,
         model_ExtraTreeRegressor, model_GradientBoostingRegressor, model_KNeighborsRegressor, model_LinearRegression,
         model_RandomForestRegressor, model_SVR]

# =====================

# ======================
# Stage 1: Load data
X = pd.read_csv('./X_train.csv')
Y = pd.read_csv('./y_train.csv')
print('Data_set Information:')
print(X.info())
# print(X.describe())
# ======================

# ======================
# Stage 2: Fill missing data
X = X.fillna(X.mean())    # Fill missing data with mean
# ======================

X.head()
Y.head()
X = X.values
Y = Y.values
X = X[:, 1:]
Y = Y[:, 1]
std = StandardScaler()
X = std.fit_transform(X)

# ======================
# Stage 3: Feature selection
print("Before feature selection: ", X.shape)
clf_selec = linear_model.Lasso(alpha=0.1)
model_selec = SelectFromModel(clf_selec.fit(X, Y), prefit=True)
X = model_selec.transform(X)
print("After feature selection: ", X.shape)
# ======================

# ======================
# Stage 4: Outlier detection

# ======================

# ======================
# Stage 5: Train and predict
kf = KFold(n_splits=5, shuffle=True)

for model in models:
    r2_mean = 0
    for train_idx, test_idx in kf.split(X):
        # train
        x_train = X[train_idx]
        y_train = Y[train_idx]
        x_test = X[test_idx]
        y_test = Y[test_idx]
#        r2_mean += try_different_method(model, x_train, y_train, x_test, y_test)
    print(r2_mean/5)

model = ensemble.GradientBoostingRegressor(n_estimators=100)
model.fit(X, Y)
x_new = pd.read_csv('./X_test.csv')
x_new = x_new.fillna(x_new.mean())
x_new.head()
x_new = x_new.values
x_new = x_new[:, 1:]
x_new = std.fit_transform(x_new)
x_new = model_selec.transform(x_new)
y_pred = model.predict(x_new)
pd.DataFrame(y_pred, columns=['y']).to_csv('y_test.csv')
# ======================
