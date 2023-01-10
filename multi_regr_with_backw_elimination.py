# Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


# Helper print function for better visualization
def special_print(value):
    return print(f'------------------\n{value}')

# Pre-processing Data
# Loading the data
veriler = pd.read_csv('veriler.csv')
# special_print(veriler)

# Imputer is used for filling NaN values with the mean of the column (the feature)
# yaş kolonunda bulunan eksik NaN değerler için ortalama ile dolduracağız
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# iloc ile locate edebiliyoruz
yas = veriler.iloc[:,1:4].values

# ortalamaları öğrendik
imputer = imputer.fit(yas)
# şimdi NaN değerlerini doldurmak için geri besliyoruz
yas = imputer.transform(yas)
# special_print(yas)


# Labelencoder and OneHotEncoder
# ülke kategorisini düzgün işlemek için onları makine öğrenmesine uygun yani sayısal formatlamamız lazım
# yine iloc ile locate ediyoruz
ulke = veriler.iloc[:,0:1].values
# special_print(ulke)

cinsiyet = veriler.iloc[:,-1:].values
# special_print(cinsiyet)

# Labelencoder is used for transforming categorical values into numerical values
label_encoder = preprocessing.LabelEncoder()

# ülke kategorisini sayısal formata çevirdik
ulke[:,0] = label_encoder.fit_transform(veriler.iloc[:,0])
cinsiyet[:,0] = label_encoder.fit_transform(veriler.iloc[:,4])
# special_print(ulke)

# one hot encoding ile kategorileri o meşhur one-hot encoder haritasına çeviriyoruz
# önce kütüphane gelsin
ohe = preprocessing.OneHotEncoder()
# şimdi çevir
ulke = ohe.fit_transform(ulke).toarray()
cinsiyet = ohe.fit_transform(cinsiyet).toarray()
# special_print(ulke)


# şimdi elde ettiğimiz verileri bir dataframe içinde birleştirme vakti, önce ülke
sonuc = pd.DataFrame(data=ulke, index=range(ulke.shape[0]), columns=['fr', 'tr', 'us'])
# special_print(sonuc)


# şimdi boy, kilo ve yaş verileri
sonuc2 = pd.DataFrame(data=yas, index=range(yas.shape[0]), columns=['boy', 'kilo', 'yas'])
# special_print(sonuc2)

# son olarak cinsiyet
sonuc3 = pd.DataFrame(data=cinsiyet[:,0:1], index=range(cinsiyet.shape[0]), columns=['cinsiyet'])
# special_print(sonuc3)

# şimdi ayrı ayrı olan dataframe'leri tek bir dataframe'de birleştirme vakti
# burada önemli olan axis = 1 demek row'ları baz alarak eşitle, yani yatay (yan yana) eşitle dataframe'leri
# yoksa axis = 0 dediğimizde dikey eşitleyecekti column'ları baz alarak, dataframeleri üst üste bindirecekti 
s = pd.concat([sonuc,sonuc2], axis=1)
# special_print(s)

# artık tüm verimiz tek bir dataset içinde birleşti s2
s2 = pd.concat([s,sonuc3], axis=1)
# special_print(s2)


# Train test split yapıyoruz, X kolonlarımız s içinde y kolonumuz sonuc3 içinde
X_train, X_test, y_train, y_test = train_test_split(s, sonuc3, test_size=0.33, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = np.around(regressor.predict(X_test))

# Boy tahmini

boy = s2[['boy']]
veri = s2.drop(columns=['boy'])

X_tr, X_te, y_tr, y_te = train_test_split(veri, boy, test_size=0.33, random_state=0)


regr = LinearRegression()
regr.fit(X_tr, y_tr)

y_pre = np.around(regr.predict(X_te))

# burada y= beta0 + beta1x1 + ... + betanxn + error 
# formulündeki beta0 değerini manuel oluşturduk ve X'e topladık tüm kolonlar ile birlikte
X = np.append(arr=np.ones((22,1)), values=veri, axis=1)



# Backward elimination için modelimizi oluşturduk ve summary() ile rapor çektik
# P>|t| kolonuna dikkat edeceğiz ve p-value 0.05 den büyük olan kolonları eleyeceğiz feature'larımızdan
model_1 = sm.OLS(boy, veri).fit()
print(model_1.summary())

# Yaş kolonunun p-value değeri eşik değerini aşıyordu, o kolonu kaldırıp yeni model üretelim
yassiz_veri = veri.drop(columns=['yas'])
model_2 = sm.OLS(boy, yassiz_veri).fit()
print(model_2.summary())








