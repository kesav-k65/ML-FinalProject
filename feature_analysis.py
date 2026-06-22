#%%
#IMPORTS AND PREPROCESSING (I)

#imports
import numpy as np
from scipy.io import arff
import pandas as pd
from sklearn.feature_selection import r_regression
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

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

Xdf = numericaldf[numericaldf.columns[2:]]



#%%
#R-REGRESSION (F-I)

r_vals = r_regression(Xdf,y)
plt.hist(r_vals,bins=100)
plt.xlabel('R-value')
plt.ylabel('# of features')
#frequency chart of r-values

#inspect correlation values
r_zeroes = Xdf.columns[np.abs(r_vals) < 0.05]
r_strong_plus = Xdf.columns[r_vals > 0.5]
r_weak_plus = Xdf.columns[r_vals > 0.25]
r_any_plus = Xdf.columns[r_vals > 0.05]
r_strong_minus = Xdf.columns[r_vals < -0.5]
r_weak_minus = Xdf.columns[r_vals < -0.25]
r_any_minus = Xdf.columns[r_vals < -0.05]

#RESULT: most features have almost no correlation but X127, X261, and X314 are strongly correlated



#%%
#VARIANCE VS IMPORTANCE (F-II)

#we want to plot variance against feature importance to find features with high variance and low-importance
#we will use the random forest from D-II for importance

X_vars = Xdf.var(axis=0)

forest = RandomForestRegressor(n_estimators=50, max_depth=4, min_samples_split=10)
forest.fit(Xdf,y)
imps = forest.feature_importances_

plt.scatter(X_vars, imps, s=5)
plt.xlabel("Feature Variances")
plt.ylabel("Feature Importances")
plt.show()

most_important = Xdf.columns[imps > 0.1]

#high variance and some importance
hvi = (X_vars > 0.2) & (imps > 0.001)
high_var_important = Xdf.columns[hvi]
#X118, X119, X127

#investigate cluster of importance around 0.04 variance
lvi = (X_vars > 0.03) & (X_vars < 0.05) & (imps > 0.001)
low_var_important = Xdf.columns[lvi]
#X29, X54, X76, X136, X232, X263, X279, X5_b'q'

#RESULT: margin features have either high or low variance, dominant feature (X314) has high variance, secondary feature (X315) has low variance
