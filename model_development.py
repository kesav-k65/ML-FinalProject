#%%
#IMPORTS AND PREPROCESSING (I)

#imports
import numpy as np
from scipy.io import arff
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, SGDRegressor
from sklearn.feature_selection import VarianceThreshold, SelectKBest, r_regression, f_regression
from sklearn.cross_decomposition import PLSRegression
import torch
import torch.nn as nn
import torch.optim as optim
import mpmath
import sympy
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from joblib import dump
from sklearn.feature_selection import r_regression

#load data
data, meta = arff.loadarff(r'C:\Users\Dell\envs\mlClass\kalanidhi_kesav_final_project\dataset.arff')
df = pd.DataFrame(data)

#one-hot encoding
categorical_cols = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8']
numericaldf = pd.get_dummies(df, columns=categorical_cols, drop_first=False).astype(float)
all_data = numericaldf.to_numpy()

#separate features from target
X = all_data[:,2:]
y = all_data[:,1]

#convert to tensors (this will be used in section C-II)
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)



#%%
#MODEL ANALYSIS (II)

def model_analyze(model, X, y):
    print('_________________________________________________________________________')
    print('MODEL TYPE: ', model)

    #run cross validation
    n_folds = 10
    preds = cross_val_predict(model, X, y, cv=n_folds,method='predict')

    #calculate MSE
    mse = mean_squared_error(y, preds)
    print("MSE of model: ", mse)

    return mse



#%%
#BASIC VISUALIZATION (III)

#reduce dimensions...
pca2 = PCA(n_components=2)
pca2_results = pca2.fit_transform(numericaldf)

#...and plot
plt.scatter(pca2_results[:,0], pca2_results[:,1], c=y, s=3)
plt.colorbar(label='Testing time')
plt.xlabel('PCA feature 1')
plt.ylabel('PCA feature 2')
plt.title('PCA 2-feature visualization')
plt.show()

#feature 2 looks interesting - worth investigating
pca_X = np.atleast_2d(pca2_results[:,1]).T
pca_lr = LinearRegression()
pca_lr.fit(pca_X, y)

best_fit_X = np.linspace(-40,180,2201)
best_fit_y = best_fit_X*pca_lr.coef_ + pca_lr.intercept_

plt.scatter(pca_X, y, s=5, color='green')
plt.plot(best_fit_X, best_fit_y, color='black', linestyle='--', alpha=0.5)
plt.xlabel('PCA feature 2')
plt.ylabel('Testing time')
plt.title('PCA feature 2 model')
plt.show()

print(r_regression(pca_X, y))

#RESULT: relationship looks very linear and slope is close to 1

model_analyze(pca_lr, X, y)

#CONCLUSION: because PCA just applies linear transformations, it does not actually improve on standard linear regression in terms of MSE
#it may allow for visualization of the data in very compressed settings but doesn't actually carry inference for regression



#%%
#LINEAR MODEL - NO REGULARIZATION (A-I)

#no regularization
lr = LinearRegression()
model_analyze(lr, X, y)

#RESULT: same MSE as LR on PCA feature 2 (MSE = 77.326)



#%%
#LINEAR MODEL - L1 PENALTY (A-II)

#testing different penalties
#l1_results = [[3,2,1,0.75,0.6,0.5,0.4,0.3,0.25,0.2],[0,0,0,0,0,0,0,0,0,0]]
l1_results = [[0.2,0.15,0.1,0.075,0.06,0.05,0.04,0.03,0.025,0.02],
              [0,0,0,0,0,0,0,0,0,0]]

for p in range(10):
    lasso = Lasso(alpha=l1_results[0][p], max_iter=2000)
    l1_results[1][p] = model_analyze(lasso, X, y)

print('_________________________________________________________________________')
print("(L1 Penalty, MSE)")
for p in range(10):
    print("(", l1_results[0][p], ",", l1_results[1][p], ")")

#RESULT: optimal penalty is 0.025 (MSE = 70.980)



#%%
#LINEAR MODEL - L2 PENALTY (A-III)

#testing different penalties
#l2_results = [[3,2,1,0.75,0.6,0.5,0.4,0.3,0.25,0.2],[0,0,0,0,0,0,0,0,0,0]]
#l2_results = [[3,4,5,6,7.5,10,12,15,18,20],[0,0,0,0,0,0,0,0,0,0]]
#l2_results = [[20,22.5,25,28,32,36,40,45,52,60],[0,0,0,0,0,0,0,0,0,0]]
l2_results = [[45,47.5,50,51,52,53,54,55,57.5,60],
              [0,0,0,0,0,0,0,0,0,0]]

for p in range(10):
    ridge = Ridge(alpha=l2_results[0][p])
    l2_results[1][p] = model_analyze(ridge, X, y)

print('_________________________________________________________________________')
print("(L2 Penalty, MSE)")
for p in range(10):
    print("(", l2_results[0][p], ",", l2_results[1][p], ")")

#RESULT: optimal penalty is 52 (MSE = 72.543)



#%%
#LINEAR MODEL - L1/L2 PENALTIES (A-IV)

#testing different penalties and l1/l2 distributions
#l12_results = [[3,2,1,0.75,0.6,0.5,0.4,0.3,0.25,0.2],[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5],[0,0,0,0,0,0,0,0,0,0]]
#l12_results = [[0.2,0.15,0.1,0.075,0.06,0.05,0.04,0.03,0.025,0.02],[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5],[0,0,0,0,0,0,0,0,0,0]]
#l12_results = [[0.02,0.0175,0.015,0.012,0.01,0.009,0.008,0.007,0.006,0.005],[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5],[0,0,0,0,0,0,0,0,0,0]]
l12_results = [[0.015,0.015,0.015,0.015,0.015,0.015,0.015,0.015,0.015,0.015],
               [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
               [0,0,0,0,0,0,0,0,0,0]]

for p in range(10):
    elastic = ElasticNet(alpha=l12_results[0][p], l1_ratio=l12_results[1][p], max_iter=2500)
    l12_results[2][p] = model_analyze(elastic, X, y)

print('_________________________________________________________________________')
print("(Total Penalty, L1 proportion, MSE)")
for p in range(10):
    print("(", l12_results[0][p], ",", l12_results[1][p], ",", l12_results[2][p], ")")

#RESULT: optimal penalty is 0.015 with full l1 penalty (MSE = 71.161)

#CONCLUSION: looks like l1 does better than l2 overall for this problem
#we will remember the optimal model from A-II as l1-op



#%%
#FEATURE SELECTION (B-I)

#STRATEGY 1: variance threshold --> l1_op
print("Variance Threshold transformation")

#only features with a variance above k are selected
#k = 0
#k = 0.04
#k = 0.004
k = 0.01
#k = 0.015
#k = 0.011
#k = 0.008
#k = 0.012

#transform data
vt = VarianceThreshold(k)
X_vt = vt.fit_transform(X)

#analyze!
l1_op = Lasso(alpha=0.025, max_iter=2000)
model_analyze(l1_op, X_vt, y)

#RESULT: optimal variance threshold is 0.01 (MSE = 70.885)


#STRATEGY 2: select k best --> l1_op
print('\n', "Select K best transformation")

#only the n most variant features are selected
#n = 1
#n = 10
#n = 30
#n = 50
#n = 75
#n = 100
#n = 60
#n = 85
#n = 80
#n = 70
#n = 65
#n = 62
#n = 64
n = 66
#n = 67

#transform data
kbest = SelectKBest(f_regression, k=n)
X_Kbest = kbest.fit_transform(X, y)

#analyze!
model_analyze(l1_op, X_Kbest, y)

#RESULT: optimal variance threshold is 66 (MSE = 71.013)



#%%
#PARTIAL LEAST SQUARES REGRESSION (B-II)

#i = 1
#i = 2
#i = 5
#i = 10
#i = 30
#i = 8
i = 6
#i = 7

pls = PLSRegression(n_components=i)
model_analyze(pls, X, y)

#RESULT: optimal component # is 6 (MSE = 74.149)



#%%
#SGD REGRESSOR (C-I)

#STRATEGY 1: constant learning rate
#a = 1
#a = 0.1
#a = 0.01
#a = 0.005
#a = 0.002
#a = 0.001
#a = 0.0005
#a = 0.00025
#a = 0.0002
#a = 0.00015
a = 0.0001
#a = 0.00006

sgd = SGDRegressor(penalty='l1', learning_rate='constant', eta0=a)
model_analyze(sgd, X, y)

#RESULT: optimal learning rate is 0.0001 (MSE is about 74)

#STRATEGY 2: inverse-scaled learning rate (decreases over time)
#a = 0.01
#a = 0.005
a = 0.002
#a = 0.001

sgd = SGDRegressor(penalty='l1', eta0=a)
model_analyze(sgd, X, y)

#RESULT: optimal learning rate is 0.001 (MSE is about 73.9)



#%%
# NEURAL NETWORK (C-II)

#torch is being used to create and train this NN, but torch doesn't include fit() and predict() implementation
#thus, a new class is being made to adapt the NN to our model analysis (and control the parameters we care about)

#enter NNEstimator!
class NNEstimator:
    
    #creates an object
    def __init__(self, lr, batch_size, epochs):
        #set training hyperparameters
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs

        #decide on layers, nodes, and ReLU/dropout placement
        self.model = nn.Sequential(
            nn.Linear(563, 450),
            nn.LayerNorm(450),
            nn.ReLU(),
            nn.Linear(450, 120),
            nn.LayerNorm(120),
            nn.ReLU(),
            nn.Linear(120, 1)
        )

        #set loss and optimizer (MSE and SGD)
        self.error_func = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=0.0001)

    def fit(self, X, y):
        #decide on total repititions and batch size
        epochs = self.epochs
        batch_size = self.batch_size
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        n_train_samples = X_tensor.shape[0]

        #overall training process
        for epoch in range(epochs):
            #training process for an epoch
            for i in range(0, n_train_samples, batch_size):
                #training process for a batch

                #batch creation
                inputs = X_tensor[i:i + batch_size]
                targets = y_tensor[i:i + batch_size]

                #make gradients zero
                self.optimizer.zero_grad()

                #forward pass
                outputs = self.model(inputs).squeeze(-1)

                #loss calculation
                self.loss = self.error_func(outputs, targets)

                #backward pass
                self.loss.backward()

                #update weights
                self.optimizer.step()

                #print results every 100 batches
                if (i // batch_size) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Batch [{i // batch_size + 1}/{n_train_samples // batch_size + (1 if n_train_samples % batch_size != 0 else 0)}], Loss: {self.loss.item():.4f}')
    
    #turn off training mode in the model and get predictions
    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            preds = self.model(X_tensor).squeeze(-1)
        return preds
    
    # following methods required to make this class an estimator
    def get_params(self, deep=True):
        return {
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)

#RECAP: on a high level, training this model consists of adapting the following:
#CONSTRUCTION: layers, nodes, activation placement
#TUNING: learning rate, batch size, epochs, weight decay
nne = NNEstimator(lr=0.0025, batch_size=40, epochs=20)
model_analyze(nne, X, y)

# RESULT: hard to pinpoint optimal architecture but few wide layers with ReLU and layer normalization work well
# lr = 0.0025, weight_decay = 0.0001, batch_size = 40, epochs = 20 (MSE is about 72)



# %%
#DECISION TREE REGRESSOR (D-I)

#adjust tree depth and node splitting
#d = 5
#d = 3
#d = 2
d = 4
#d = 6

#s = 2
#s = 5
s = 10
#s = 20

tree = DecisionTreeRegressor(max_depth=d, min_samples_split=s)
model_analyze(tree, X, y)

#RESULT: optimal (d, s) is (4, 10) (MSE is about 71)



# %%
#RANDOM FOREST REGRESSOR (D-II)

#adjust tree count, tree depth and node splitting
#n = 1
#n = 5
#n = 10
#n = 20
#n = 32
#n = 50
#n = 75
#n = 100
#n = 250

#d = 5
#d = 3
#d = 2
d = 4
#d = 6

#s = 2
#s = 5
s = 10
#s = 20

forest = RandomForestRegressor(n_estimators=n, max_depth=d, min_samples_split=s)
model_analyze(forest, X, y)

#RESULT: optimal (n, d, s) is (50, 4, 10) (MSE is about 70.75)



#%%
#PIPELINE EXPERIMENT (E-I)

#we know the best transformer is variance threshold and the best predictors are l1-op and random forest
#variance threshold has been tested on l1-op but not random forest
#lets create a pipeline to test this

t = 0.01
#t = 0.012
#t = 0.008

#n = 50
#n = 75
n = 90
#n = 110

d = 4
#d = 5
#d = 3

s = 10
#s = 15
#s = 8

pipe = Pipeline([('vt', VarianceThreshold(t)),
                 ('rf', RandomForestRegressor(n_estimators=n, max_depth=d, min_samples_split=s))])
model_analyze(pipe, X, y)

#RESULT: only n could be improved upon using the pipeline structure
#optimal (t, n, d, s) = (0.01, 90, 4, 10) (MSE is about 70.1)

#CONCLUSION: the pipeline is the most accurate model yet and will be used in the final model
pipe.fit(X, y)
dump(pipe, "pipe.pkl")