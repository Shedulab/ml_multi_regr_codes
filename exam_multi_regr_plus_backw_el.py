# Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# Helper print function for better visualization
def special_print(value):
    return print(f'------------------\n{value}')

# Pre-processing Data
# Loading the data
ham_veriler = pd.read_csv('odev_tenis.csv')

# Labelencoder
l_encoder = LabelEncoder()
outlook = ham_veriler.iloc[:,:1].values
windy = ham_veriler.iloc[:,3:4].values
play = ham_veriler.iloc[:,4:5].values

outlook[:,0] = l_encoder.fit_transform(ham_veriler.iloc[:,:1])
windy[:,0] = l_encoder.fit_transform(ham_veriler.iloc[:,3:4])
play[:,0] = l_encoder.fit_transform(ham_veriler.iloc[:,4:5])
# ====



# Onehotencoder
o_hot = OneHotEncoder()
outlook = o_hot.fit_transform(outlook).toarray()
# deleting second column to prevent multicollinearity
windy = np.delete(o_hot.fit_transform(windy).toarray(), obj=1, axis=1)
play = np.delete(o_hot.fit_transform(play).toarray(), obj=1, axis=1)
# ====

# Converting to df and concatenate
outlook = pd.DataFrame(data=outlook, index=range(outlook.shape[0]), columns=['overcast', 'rainy', 'sunny'])
windy = pd.DataFrame(data=windy, index=range(windy.shape[0]), columns=['windy'])
play = pd.DataFrame(data=play, index=range(play.shape[0]), columns=['play'])
temperature = ham_veriler[['temperature']].apply(lambda num : num /100)
humidity = ham_veriler[['humidity']].apply(lambda num : num /100)

processed_df = pd.concat([outlook, temperature, humidity, windy, play], axis=1)
all_features = processed_df.drop(columns=['play'])
# or like this: 
# all_features = processed_df.iloc[:,0:6]

# ====

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(all_features, play, test_size=0.33, random_state=0)

# Fit model and predict
regr = LinearRegression().fit(X_train, y_train)
y_pred = regr.predict(X_test)


# burada y= beta0 + beta1x1 + ... + betanxn + error 
# formulündeki beta0 değerini manuel oluşturduk ve X'e topladık tüm kolonlar ile birlikte
# X = np.append(arr=np.ones((22,1)), values=veri, axis=1)

# Model evaluation with statsmodel
model_1 = sm.OLS(play, all_features).fit()
print(model_1.summary())

# Applying backward elimination for temperature column
backw = all_features.drop(columns=['temperature'])

model_2 = sm.OLS(play, backw).fit()
print(model_2.summary())




