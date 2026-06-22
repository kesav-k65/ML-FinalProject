#%%
#FINAL TESTING TIME PREDICTION MODEL
from joblib import load

def testing_time_predict(x):
    pipe = load('pipe.pkl')
    preds = pipe.predict(x)
    return preds
print('Prediction model initialized, please input data as numpy array with no IDs')