# Banking predictions 

## 0. Dowlnoad and import packages/dataset

**Import packages and dataset**


```python
import numpy as np
import pandas as pd
import math

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

import scipy.stats as ss
from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, average_precision_score, precision_recall_curve, roc_curve

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier

import optuna
from optuna.samplers import TPESampler

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from sklearn.inspection import permutation_importance

import warnings
warnings.filterwarnings("ignore")

sns.set()
```


```python
df = pd.read_csv('bank_full.csv', delimiter=';')
```


## 1. Exploration des données

### 1.1 Statistiques élémentaires


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>month</th>
      <th>day_of_week</th>
      <th>...</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>emp.var.rate</th>
      <th>cons.price.idx</th>
      <th>cons.conf.idx</th>
      <th>euribor3m</th>
      <th>nr.employed</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>56</td>
      <td>housemaid</td>
      <td>married</td>
      <td>basic.4y</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>telephone</td>
      <td>may</td>
      <td>mon</td>
      <td>...</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.857</td>
      <td>5191.0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>1</th>
      <td>57</td>
      <td>services</td>
      <td>married</td>
      <td>high.school</td>
      <td>unknown</td>
      <td>no</td>
      <td>no</td>
      <td>telephone</td>
      <td>may</td>
      <td>mon</td>
      <td>...</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.857</td>
      <td>5191.0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37</td>
      <td>services</td>
      <td>married</td>
      <td>high.school</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>telephone</td>
      <td>may</td>
      <td>mon</td>
      <td>...</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.857</td>
      <td>5191.0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>3</th>
      <td>40</td>
      <td>admin.</td>
      <td>married</td>
      <td>basic.6y</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>telephone</td>
      <td>may</td>
      <td>mon</td>
      <td>...</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.857</td>
      <td>5191.0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>4</th>
      <td>56</td>
      <td>services</td>
      <td>married</td>
      <td>high.school</td>
      <td>no</td>
      <td>no</td>
      <td>yes</td>
      <td>telephone</td>
      <td>may</td>
      <td>mon</td>
      <td>...</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.857</td>
      <td>5191.0</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
print('Dimension dataframe :', df.shape)
```

    Dimension dataframe : (41188, 21)
    


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 41188 entries, 0 to 41187
    Data columns (total 21 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   age             41188 non-null  object 
     1   job             41188 non-null  object 
     2   marital         41188 non-null  object 
     3   education       41188 non-null  object 
     4   default         41188 non-null  object 
     5   housing         41188 non-null  object 
     6   loan            41188 non-null  object 
     7   contact         41188 non-null  object 
     8   month           41188 non-null  object 
     9   day_of_week     41188 non-null  object 
     10  duration        41188 non-null  int64  
     11  campaign        41188 non-null  object 
     12  pdays           41188 non-null  int64  
     13  previous        41188 non-null  int64  
     14  poutcome        41188 non-null  object 
     15  emp.var.rate    41188 non-null  float64
     16  cons.price.idx  41188 non-null  float64
     17  cons.conf.idx   41188 non-null  float64
     18  euribor3m       41188 non-null  float64
     19  nr.employed     41188 non-null  float64
     20  y               41188 non-null  object 
    dtypes: float64(5), int64(3), object(13)
    memory usage: 6.6+ MB
    

**Describe des valeurs numériques**


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>duration</th>
      <th>pdays</th>
      <th>previous</th>
      <th>emp.var.rate</th>
      <th>cons.price.idx</th>
      <th>cons.conf.idx</th>
      <th>euribor3m</th>
      <th>nr.employed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>41188.000000</td>
      <td>41188.000000</td>
      <td>41188.000000</td>
      <td>41188.000000</td>
      <td>41188.000000</td>
      <td>41188.000000</td>
      <td>41188.000000</td>
      <td>41188.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>258.285010</td>
      <td>962.475454</td>
      <td>0.172963</td>
      <td>0.081886</td>
      <td>93.575664</td>
      <td>-40.502600</td>
      <td>3.621291</td>
      <td>5167.035911</td>
    </tr>
    <tr>
      <th>std</th>
      <td>259.279249</td>
      <td>186.910907</td>
      <td>0.494901</td>
      <td>1.570960</td>
      <td>0.578840</td>
      <td>4.628198</td>
      <td>1.734447</td>
      <td>72.251528</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-3.400000</td>
      <td>92.201000</td>
      <td>-50.800000</td>
      <td>0.634000</td>
      <td>4963.600000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>102.000000</td>
      <td>999.000000</td>
      <td>0.000000</td>
      <td>-1.800000</td>
      <td>93.075000</td>
      <td>-42.700000</td>
      <td>1.344000</td>
      <td>5099.100000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>180.000000</td>
      <td>999.000000</td>
      <td>0.000000</td>
      <td>1.100000</td>
      <td>93.749000</td>
      <td>-41.800000</td>
      <td>4.857000</td>
      <td>5191.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>319.000000</td>
      <td>999.000000</td>
      <td>0.000000</td>
      <td>1.400000</td>
      <td>93.994000</td>
      <td>-36.400000</td>
      <td>4.961000</td>
      <td>5228.100000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4918.000000</td>
      <td>999.000000</td>
      <td>7.000000</td>
      <td>1.400000</td>
      <td>94.767000</td>
      <td>-26.900000</td>
      <td>5.045000</td>
      <td>5228.100000</td>
    </tr>
  </tbody>
</table>
</div>



**Données manquantes**


```python
df.isnull().values.any()
```




    False




```python
df.isnull().sum() # OK
```




    age               0
    job               0
    marital           0
    education         0
    default           0
    housing           0
    loan              0
    contact           0
    month             0
    day_of_week       0
    duration          0
    campaign          0
    pdays             0
    previous          0
    poutcome          0
    emp.var.rate      0
    cons.price.idx    0
    cons.conf.idx     0
    euribor3m         0
    nr.employed       0
    y                 0
    dtype: int64



 Notre dataset ne comporte pas de données manquantes

### 1.2 Nettoyage des données


```python
missing_columns = [col for col in df.columns if (df[col] == ' ').any()]
missing_columns
```




    ['age', 'marital', 'campaign']



Nous voyons que les colonnes **age**, **marital** et **campaign** possèdent au moins une valeur non renseignée

**Variable age**

Nous voyons que la variable **age** est classée en catégorie object comme dtypes : 


```python
df.age.dtype
```




    dtype('O')




```python
df.age.value_counts().keys()
```




    Index(['31', '32', '33', '36', '35', '34', '30', '37', '29', '39', '38', '41',
           '40', '42', '45', '43', '46', '44', '28', '48', '47', '50', '27', '49',
           '52', '51', '53', '56', '26', '54', '55', '57', '25', '58', '24', '59',
           '60', '23', '22', '21', '61', '20', '62', '66', '64', '63', '71', '70',
           '65', '19', '76', '72', '69', '73', '68', '74', '80', '18', '78', '67',
           '75', '88', '81', '77', '83', '82', '85', '79', '86', '84', '17', ' ',
           '92', '98', '89', '91', '87', '94', '143', '158', '95'],
          dtype='object', name='age')




```python
df.query('age == " "')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>month</th>
      <th>day_of_week</th>
      <th>...</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>emp.var.rate</th>
      <th>cons.price.idx</th>
      <th>cons.conf.idx</th>
      <th>euribor3m</th>
      <th>nr.employed</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17352</th>
      <td></td>
      <td>services</td>
      <td>married</td>
      <td>high.school</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>cellular</td>
      <td>jul</td>
      <td>mon</td>
      <td>...</td>
      <td>8</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.4</td>
      <td>93.918</td>
      <td>-42.7</td>
      <td>4.962</td>
      <td>5228.1</td>
      <td>no</td>
    </tr>
    <tr>
      <th>25937</th>
      <td></td>
      <td>blue-collar</td>
      <td>married</td>
      <td>basic.9y</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>cellular</td>
      <td>nov</td>
      <td>wed</td>
      <td>...</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-0.1</td>
      <td>93.200</td>
      <td>-42.0</td>
      <td>4.120</td>
      <td>5195.8</td>
      <td>no</td>
    </tr>
    <tr>
      <th>26661</th>
      <td></td>
      <td>admin.</td>
      <td>single</td>
      <td>university.degree</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>cellular</td>
      <td>nov</td>
      <td>thu</td>
      <td>...</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-0.1</td>
      <td>93.200</td>
      <td>-42.0</td>
      <td>4.076</td>
      <td>5195.8</td>
      <td>no</td>
    </tr>
    <tr>
      <th>38155</th>
      <td></td>
      <td>admin.</td>
      <td>single</td>
      <td>university.degree</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>cellular</td>
      <td>oct</td>
      <td>thu</td>
      <td>...</td>
      <td>3</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-3.4</td>
      <td>92.431</td>
      <td>-26.9</td>
      <td>0.754</td>
      <td>5017.5</td>
      <td>yes</td>
    </tr>
  </tbody>
</table>
<p>4 rows × 21 columns</p>
</div>



Nous observons 2 choses : 
- La présence d'age non renseigné ' '
- La présence d'age abérrant (158 et 143)

Ces deux éléments étant des anomalies, nous retirons les lignes concernées de notre dataframe initial. Enfin nous transformons le type de cette variable en *int*.


```python
# drop des lignes où  l'age n'est pas renseigné
df = df.drop(df.query('age == " "').index)

# Changement du type
df['age'] = df['age'].astype('int16')

# Supression des ages trop élevés
df = df.drop(df.query('age > 100').index)
```

**Variable campaign**

Tout comme la variable **age**,  nous observons également pour la variable **campaign**, censé représenter le nombre de contacts;  des valeurs non renseignées.


```python
df.campaign.value_counts().keys()
```




    Index([ '1',  '2',  '3',    1,    2,  '4',  '5',    3,  '6',  '7',    4,  '8',
              5,  '9', '10',    6, '11', '12',    7, '13', '14',    8, '17', '16',
           '15',    9, '18', '20', '19', '21',   10, '22', '23', '24',   11, '27',
           '29', '26', '28', '25', '30', '31',   12, '35',  ' ', '33', '32',   15,
           '34',   13, '42', '43', '40',   14, '37', '56', '39', '41',   16],
          dtype='object', name='campaign')




```python
df.query('campaign == " "')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>month</th>
      <th>day_of_week</th>
      <th>...</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>emp.var.rate</th>
      <th>cons.price.idx</th>
      <th>cons.conf.idx</th>
      <th>euribor3m</th>
      <th>nr.employed</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>430</th>
      <td>43</td>
      <td>self-employed</td>
      <td>divorced</td>
      <td>university.degree</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>telephone</td>
      <td>may</td>
      <td>tue</td>
      <td>...</td>
      <td></td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.857</td>
      <td>5191.0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>438</th>
      <td>47</td>
      <td>admin.</td>
      <td>divorced</td>
      <td>university.degree</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>telephone</td>
      <td>may</td>
      <td>tue</td>
      <td>...</td>
      <td></td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.857</td>
      <td>5191.0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>24342</th>
      <td>40</td>
      <td>management</td>
      <td>married</td>
      <td>university.degree</td>
      <td>no</td>
      <td>yes</td>
      <td>yes</td>
      <td>cellular</td>
      <td>nov</td>
      <td>mon</td>
      <td>...</td>
      <td></td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-0.1</td>
      <td>93.200</td>
      <td>-42.0</td>
      <td>4.191</td>
      <td>5195.8</td>
      <td>no</td>
    </tr>
    <tr>
      <th>24366</th>
      <td>45</td>
      <td>services</td>
      <td>divorced</td>
      <td>high.school</td>
      <td>no</td>
      <td>yes</td>
      <td>yes</td>
      <td>cellular</td>
      <td>nov</td>
      <td>mon</td>
      <td>...</td>
      <td></td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-0.1</td>
      <td>93.200</td>
      <td>-42.0</td>
      <td>4.191</td>
      <td>5195.8</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
<p>4 rows × 21 columns</p>
</div>




```python
# drop des lignes où  l'age n'est pas renseigné
df = df.drop(df.query('campaign == " "').index)

# Changement du type
df['campaign'] = df['campaign'].astype('int16')
```

**Variable marital**


```python
df.query('marital == " "')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>month</th>
      <th>day_of_week</th>
      <th>...</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>emp.var.rate</th>
      <th>cons.price.idx</th>
      <th>cons.conf.idx</th>
      <th>euribor3m</th>
      <th>nr.employed</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>560</th>
      <td>41</td>
      <td>blue-collar</td>
      <td></td>
      <td>basic.4y</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>telephone</td>
      <td>may</td>
      <td>tue</td>
      <td>...</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.857</td>
      <td>5191.0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>18872</th>
      <td>35</td>
      <td>admin.</td>
      <td></td>
      <td>university.degree</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>cellular</td>
      <td>aug</td>
      <td>mon</td>
      <td>...</td>
      <td>2</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.4</td>
      <td>93.444</td>
      <td>-36.1</td>
      <td>4.970</td>
      <td>5228.1</td>
      <td>no</td>
    </tr>
    <tr>
      <th>18877</th>
      <td>57</td>
      <td>admin.</td>
      <td></td>
      <td>high.school</td>
      <td>unknown</td>
      <td>yes</td>
      <td>no</td>
      <td>cellular</td>
      <td>aug</td>
      <td>mon</td>
      <td>...</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.4</td>
      <td>93.444</td>
      <td>-36.1</td>
      <td>4.970</td>
      <td>5228.1</td>
      <td>no</td>
    </tr>
    <tr>
      <th>35192</th>
      <td>54</td>
      <td>entrepreneur</td>
      <td></td>
      <td>high.school</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>cellular</td>
      <td>may</td>
      <td>fri</td>
      <td>...</td>
      <td>1</td>
      <td>999</td>
      <td>1</td>
      <td>failure</td>
      <td>-1.8</td>
      <td>92.893</td>
      <td>-46.2</td>
      <td>1.250</td>
      <td>5099.1</td>
      <td>no</td>
    </tr>
    <tr>
      <th>35209</th>
      <td>37</td>
      <td>admin.</td>
      <td></td>
      <td>high.school</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>cellular</td>
      <td>may</td>
      <td>fri</td>
      <td>...</td>
      <td>3</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-1.8</td>
      <td>92.893</td>
      <td>-46.2</td>
      <td>1.250</td>
      <td>5099.1</td>
      <td>no</td>
    </tr>
    <tr>
      <th>37706</th>
      <td>66</td>
      <td>retired</td>
      <td></td>
      <td>basic.6y</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>cellular</td>
      <td>aug</td>
      <td>thu</td>
      <td>...</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-2.9</td>
      <td>92.201</td>
      <td>-31.4</td>
      <td>0.851</td>
      <td>5076.2</td>
      <td>yes</td>
    </tr>
  </tbody>
</table>
<p>6 rows × 21 columns</p>
</div>




```python
# drop des lignes où marital n'est pas renseigné
df = df.drop(df.query('marital == " "').index)
```

### 1.3 Analyse univariée

**Variables catégoriques**


```python
# Récupération des variables catégoriques mises à jour
var_cat = df.select_dtypes(include=['object']).columns.tolist()
```


```python
palette = sns.color_palette("Set2", len(var_cat))  # Adjust the palette name as desired

fig, ax = plt.subplots(5, 2, figsize=(10, 14))
fig.subplots_adjust(hspace=0.5, wspace=0.3)

for i, subplot, color in zip(var_cat, ax.flatten(), palette):
    ax = sns.countplot(x=i, data=df, ax=subplot, color=color)  # Use the color palette
    ax.bar_label(ax.containers[0])
    ax.spines[['right', 'top']].set_visible(False)

    subplot.set_title(f'Distribution de {i}', fontsize=12)
    subplot.set_xlabel(None)
    subplot.set_ylabel('Nombre', fontsize=10)
    subplot.tick_params(axis='x', labelrotation=45)

plt.tight_layout()
plt.show()
```


    
![png](banking_predictions_files/banking_predictions_37_0.png)
    


Bien que certaines modalitées de certaines variables soient peu representées, nous conservons ces dernières. A noter qu'aucune de ces variables catégorielles sont ordinales, nous devrons donc transformer toutes ces variables en les one hot encodant. 

**Variables numériques**


```python
# fonction pour visu histogramme, boxplot et QQplot
def visu_plots(df, variable):
    plt.figure(figsize=(16, 4))
    # histogram
    plt.subplot(1, 3, 1)
    sns.histplot(data=df, x=variable, kde=True,)
    plt.title('Histogram')
    # boxplot
    plt.subplot(1, 3, 3)
    sns.boxplot(y=df[variable])
    plt.title('Boxplot')
    # Q-Q plot
    plt.subplot(1, 3, 2)
    ss.probplot(df[variable], dist="norm", plot=plt)
    plt.ylabel('Variable quantiles')
    plt.show()
```


```python
num_rows = 5
num_cols = 2
cols_numeriques = df.select_dtypes(include=['number']).columns.tolist()

fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(10, 20))
axes = axes.flatten()

for i, column in enumerate(cols_numeriques):
    sns.histplot(data=df, x=column, kde=True, ax=axes[i])
    kurtosis = np.round(df[column].kurtosis(), 2)
    skewness = np.round(df[column].skew(), 2)
    axes[i].set_title('Kurtosis : ' +str(kurtosis)+ ', Skewness :' + str(skewness))  # Set subplot title to column name

plt.subplots_adjust(hspace=0.9)
plt.show()
```


    
![png](banking_predictions_files/banking_predictions_41_0.png)
    



```python
## Affichage en détail de la variable revenuFinal
visu_plots(df, 'duration')
visu_plots(df, 'campaign')
```


    
![png](banking_predictions_files/banking_predictions_42_0.png)
    



    
![png](banking_predictions_files/banking_predictions_42_1.png)
    


Nous observons pour certaines variables comme **campaign** ou **duration** une forte assymétrie, sans pour autant avoir de valeurs abbérantes. Nous conservons ces parmètres pour notre modélisation.

### 1.4 Analyse bivariée

Notre target variable **y** étant si le client a souscris ou non, observons dans un premier temps sa distribution : 


```python
df['y'].value_counts(normalize = True)
```




    y
    no     0.887351
    yes    0.112649
    Name: proportion, dtype: float64




```python
sns.countplot(x='y', data=df) 
plt.show()
```


    
![png](banking_predictions_files/banking_predictions_47_0.png)
    


La modalité 'yes' représentant 11%, ce chiffre est suffisamment elevé pour écarter l'hypohtèse de sous-representation de cette dernière.

**Variables Catégorielles**

Pour les variables catégorielles nous affichons un stacked plot en fonction de **y**. Nous calculons également la valeur de la p-value issue d'un test de Khi-carré en complément du graphique.


```python
n_cols = 2
n_rows = math.ceil(len(var_cat[:-1]) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 5 * n_rows))

for i, c in enumerate(var_cat[:-1]):
    row = i // n_cols
    col = i % n_cols

    contingency_table = pd.crosstab(df['y'], df[c])
    chi2, p, _, _ = chi2_contingency(contingency_table)

    df.groupby('y')[c].value_counts(normalize=True).unstack().plot(kind='bar', stacked=True, ax=axes[row, col], color= palette)

    axes[row, col].set_title(c)
    axes[row, col].text(0.5, 1.1, f'Test de Khi-carré p-value: {p:.2e}', ha='center', va='bottom', transform=axes[row, col].transAxes)
    

# Supprimer les axes inutilisés si le nombre de colonnes est impair
if len(var_cat[:-1]) % n_cols != 0:
    for j in range(len(var_cat[:-1]), n_rows * n_cols):
        fig.delaxes(axes.flat[j])

plt.tight_layout()
plt.show()
```


    
![png](banking_predictions_files/banking_predictions_51_0.png)
    


Nous voyons visuellement et au moyen de la p-value (>0.05) que les variables **housing** et **loan**  ont un impact faible sur la target variable. Nous décidons de les retirer du dataframe et conservons le reste des features. 

**Variables Numériques**


```python
n_cols = 2
n_rows = math.ceil(len(cols_numeriques) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 4 * n_rows))

for i, c in enumerate(cols_numeriques):
    row = i // n_cols
    col = i % n_cols

    U1, p = ttest_ind(df.query('y =="yes"')[c], df.query('y =="no"')[c], equal_var=False)

    sns.violinplot(data=df, x='y', y=c, ax=axes[row, col], hue='y', palette="pastel", legend=False)
    axes[row, col].set_title(c)
    axes[row, col].text(0.5, 1.1, f'Test de student p test: {p:.2e}', ha='center', va='bottom', transform=axes[row, col].transAxes)

# Supprimer les axes inutilisés si le nombre de colonnes est impair
if len(cols_numeriques) % n_cols != 0:
    for j in range(len(cols_numeriques), n_rows * n_cols):
        fig.delaxes(axes.flat[j])

plt.tight_layout()
plt.show()
```


    
![png](banking_predictions_files/banking_predictions_54_0.png)
    


Parmi les variables numériques toutes semblent signifcatives. 


```python
## Drop des variables housing et loan

df_clean = df.drop('housing', axis=1)
df_clean = df_clean.drop('loan', axis=1)
```

### 1.5 Analyse des corrélations

**Corrélation des variables numériques**


```python
plt.figure(figsize=(8, 6))
sns.heatmap(df[cols_numeriques].corr(), annot=True)
plt.show()
```


    
![png](banking_predictions_files/banking_predictions_59_0.png)
    


Nous observons des fortes corrélations entre plusieurs variables. Nous décidons de drop les plus corrélées, à savoir **euribor3m** et **emp.var.rate**


```python
df_clean = df_clean.drop('euribor3m', axis=1)
df_clean = df_clean.drop('emp.var.rate', axis=1)
```


```python
new_col_nums = [x for x in cols_numeriques if x not in ['euribor3m', 'emp.var.rate']]

plt.figure(figsize=(7, 5))
sns.heatmap(df[new_col_nums].corr(), annot=True)
plt.show()
```


    
![png](banking_predictions_files/banking_predictions_62_0.png)
    


**Corrélation des variables via le V de Cramer**

De la même manière que nous avons calculé la corrélation entre variables numérique via le coefficient de Pearson, il est possible de quantifier la relation via le V de Cramer.

Ce dernier présente l’avantage d’être plus lisible que la probabilité associée au Khi-deux, et de fournir une mesure absolue de l’intensité de la liaison entre deux variables qualitatives ou quantitatives discrètes, indépendamment du nombre de leurs modalités et de l’effectif de la population.


```python
def cramers_corrected_stat(x,y):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    result=-1
    if len(x.value_counts())==1 :
        print("First variable is constant")
    elif len(y.value_counts())==1:
        print("Second variable is constant")
    else:
        conf_matrix=pd.crosstab(x, y)

        if conf_matrix.shape[0]==2:
            correct=False
        else:
            correct=True

        chi2 = chi2_contingency(conf_matrix, correction=correct)[0]

        n = sum(conf_matrix.sum())
        phi2 = chi2/n
        r,k = conf_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        result=np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))
    return round(result,6)
```


```python
df_clean_corr = df_clean.drop(['cons.price.idx', 'cons.conf.idx'], axis=1)

n_cols = df_clean_corr.shape[1]
cramers_matrix = np.zeros((n_cols, n_cols))

for i in range(n_cols):
    for j in range(i, n_cols):
        v_cramer = cramers_corrected_stat(df_clean_corr.iloc[:, i], df_clean_corr.iloc[:, j])
        cramers_matrix[i, j] = v_cramer
        cramers_matrix[j, i] = v_cramer
```


```python
plt.figure(figsize=(12, 10))
sns.heatmap(cramers_matrix, annot=True, fmt=".2f",
            xticklabels=df_clean_corr.columns, yticklabels=df_clean_corr.columns)
plt.show()
```


    
![png](banking_predictions_files/banking_predictions_67_0.png)
    


L'ensemble des valeurs issues du V de cramer étant strictement **inférieur à 0.8**, nous conservons donc les variables.


## 2. Préparation jeu de données

### 2.1 Split train et test set


```python
X = df_clean.drop('y', axis=1)
y = df_clean['y'].map({'yes': 1, 'no': 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 2.2 Pipeline encoding + scaling


```python
var_cat_final = df_clean.select_dtypes(include=['object']).columns.tolist()[:-1]
var_num_final = df_clean.select_dtypes(exclude=['object']).columns.tolist()


categorical_transformer = OneHotEncoder(handle_unknown='ignore')
numerical_transformer = MinMaxScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, var_num_final),
        ('cat', categorical_transformer, var_cat_final)
    ])

pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

X_train_prepared = pipeline.fit_transform(X_train).toarray()
X_test_prepared = pipeline.transform(X_test).toarray()

print(X_train_prepared.shape)
print(X_test_prepared.shape)
```

    (32937, 55)
    (8235, 55)
    


```python
# Check des proportion du split
print(y_train.value_counts(normalize=True)[0])
print(y_test.value_counts(normalize=True)[0])
print(y.value_counts(normalize=True)[0])
```

    0.8876946898624647
    0.8859744990892532
    0.8873506266394637
    

## 3. Modélisation

Pour la partie modélisation nous utiliserons plusieurs algorithmes de classification : 
- Régression logistique
- Random forest
- Xgboost
- Catboost
- ANN

Nous implémentons dans un premier temps les différents algorithmes avec leurs paramètres initiaux puis tunons les hyperparamètres au moyen de l'approche bayésienne via la librairie Optuna


### 3.1 Modèles initiaux


```python
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(), 
    'Xgboost': XGBClassifier(), 
    'Catboost': CatBoostClassifier()
}

for name, model in models.items():
    if name == 'Catboost': 
        model.fit(X_train_prepared, y_train, plot = False, logging_level='Silent')
    else : 
        model.fit(X_train_prepared, y_train)

    y_pred = model.predict(X_test_prepared)
    
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"---------------------{name} Metrics: ---------------------")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print('Confusion matrix :\n', cm)
```

    ---------------------Logistic Regression Metrics: ---------------------
    Precision: 0.6847
    Recall: 0.4047
    F1-Score: 0.5087
    Confusion matrix :
     [[7121  175]
     [ 559  380]]
    ---------------------Random Forest Metrics: ---------------------
    Precision: 0.6652
    Recall: 0.4952
    F1-Score: 0.5678
    Confusion matrix :
     [[7062  234]
     [ 474  465]]
    ---------------------Xgboost Metrics: ---------------------
    Precision: 0.6457
    Recall: 0.5240
    F1-Score: 0.5785
    Confusion matrix :
     [[7026  270]
     [ 447  492]]
    ---------------------Catboost Metrics: ---------------------
    Precision: 0.6844
    Recall: 0.5474
    F1-Score: 0.6083
    Confusion matrix :
     [[7059  237]
     [ 425  514]]
    

### 3.1 Modèles tunés

**Regression Logistique**


```python
number_trials = 100
```


```python
# Tuning regression logistique
def tune_logistic(trial:optuna.Trial):

    param = {
        'penalty': trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet", None]),
        'C' : trial.suggest_float('C', 1e-6, 1e3, log = True), 
        'max_iter' : trial.suggest_int('max_iter', 100, 1e4, log = True)
    }

    if param['penalty'] == 'elasticnet': 
        param['l1_ratio'] = trial.suggest_float('l1_ratio', 1e-6, 1, log= True)
        param['solver'] = 'saga'

    if param['penalty'] == 'l1':
        param['solver'] = trial.suggest_categorical('solver_l1', ['liblinear','saga'])

    if param['penalty'] == 'l2':
        param['solver'] = trial.suggest_categorical('solver_l2', ['lbfgs','newton-cg','liblinear','saga', 'sag'])

    
    logistic = LogisticRegression(**param)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=21)
    score = cross_val_score(logistic, X_train_prepared, y_train, scoring='f1', cv=kfold).mean()
    
    return score
```


```python
sampler = TPESampler(seed=21)
study_lr = optuna.create_study(direction='maximize', sampler = sampler)
study_lr.optimize(tune_logistic, n_trials = number_trials)
```

    [I 2024-05-31 07:33:15,773] A new study created in memory with name: no-name-632516a0-5cfb-4718-8757-5687966059a7
    [I 2024-05-31 07:33:17,442] Trial 0 finished with value: 0.0 and parameters: {'penalty': 'elasticnet', 'C': 7.133536494718945e-05, 'max_iter': 126, 'l1_ratio': 6.51075515971989e-05}. Best is trial 0 with value: 0.0.
    [I 2024-05-31 07:33:21,162] Trial 1 finished with value: 0.5077047316463643 and parameters: {'penalty': 'l1', 'C': 64.06788036451141, 'max_iter': 184, 'solver_l1': 'saga'}. Best is trial 1 with value: 0.5077047316463643.
    [I 2024-05-31 07:33:22,766] Trial 2 finished with value: 0.15818855760391748 and parameters: {'penalty': 'elasticnet', 'C': 0.0028724355221069154, 'max_iter': 655, 'l1_ratio': 0.01906174113168418}. Best is trial 1 with value: 0.5077047316463643.
    [I 2024-05-31 07:33:24,334] Trial 3 finished with value: 0.3517718497462839 and parameters: {'penalty': 'elasticnet', 'C': 0.04467009126986524, 'max_iter': 216, 'l1_ratio': 6.196604094564109e-05}. Best is trial 1 with value: 0.5077047316463643.
    [I 2024-05-31 07:33:24,745] Trial 4 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 9.049150846739358e-05, 'max_iter': 3975}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:33:25,914] Trial 5 finished with value: 0.48389339748071886 and parameters: {'penalty': 'l2', 'C': 0.7495569903657464, 'max_iter': 592, 'solver_l2': 'saga'}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:33:26,328] Trial 6 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 1.948709492001072, 'max_iter': 808}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:33:26,522] Trial 7 finished with value: 0.0 and parameters: {'penalty': 'l2', 'C': 0.00012047885583491695, 'max_iter': 1514, 'solver_l2': 'lbfgs'}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:33:27,772] Trial 8 finished with value: 0.28631819866636105 and parameters: {'penalty': 'l2', 'C': 0.013990016119831599, 'max_iter': 5344, 'solver_l2': 'saga'}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:33:28,182] Trial 9 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 119.07376306532935, 'max_iter': 244}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:33:28,612] Trial 10 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 1.8760596766700107e-06, 'max_iter': 7886}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:33:29,043] Trial 11 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 0.6313839054241591, 'max_iter': 2348}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:33:29,478] Trial 12 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 2.1693158220606605, 'max_iter': 3123}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:33:29,902] Trial 13 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 1.483732459473229e-06, 'max_iter': 1103}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:33:30,107] Trial 14 finished with value: 0.0 and parameters: {'penalty': 'l1', 'C': 0.00012307946006634378, 'max_iter': 518, 'solver_l1': 'liblinear'}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:33:30,548] Trial 15 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 9.034624284977275, 'max_iter': 3694}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:33:30,974] Trial 16 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 690.6432326294854, 'max_iter': 1812}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:33:31,420] Trial 17 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 0.07271793953890554, 'max_iter': 9971}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:33:31,690] Trial 18 finished with value: 0.0 and parameters: {'penalty': 'l1', 'C': 0.0011887029296602414, 'max_iter': 408, 'solver_l1': 'liblinear'}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:33:32,196] Trial 19 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 3.719819820396309e-05, 'max_iter': 1050}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:33:32,624] Trial 20 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 0.0011114385995185858, 'max_iter': 4440}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:33:33,041] Trial 21 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 82.42699175032381, 'max_iter': 243}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:33:33,478] Trial 22 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 26.18415664316682, 'max_iter': 347}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:33:33,895] Trial 23 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 881.1686237266802, 'max_iter': 790}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:33:34,336] Trial 24 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 0.22719231066317036, 'max_iter': 339}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:33:34,831] Trial 25 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 5.606052896135111, 'max_iter': 141}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:33:41,618] Trial 26 finished with value: 0.5077047316463643 and parameters: {'penalty': 'l1', 'C': 125.44185997122221, 'max_iter': 103, 'solver_l1': 'saga'}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:33:41,790] Trial 27 finished with value: 0.0 and parameters: {'penalty': 'elasticnet', 'C': 7.680440996763988e-06, 'max_iter': 1622, 'l1_ratio': 0.2815965103436196}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:33:42,436] Trial 28 finished with value: 0.5046378373549085 and parameters: {'penalty': 'l2', 'C': 9.845292645649458, 'max_iter': 2704, 'solver_l2': 'liblinear'}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:33:44,100] Trial 29 finished with value: 0.278984831058254 and parameters: {'penalty': 'elasticnet', 'C': 0.011495510103708221, 'max_iter': 5847, 'l1_ratio': 1.298421286595282e-06}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:33:44,608] Trial 30 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 0.21679699951365627, 'max_iter': 838}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:33:45,050] Trial 31 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 1.2402717101602337e-06, 'max_iter': 9261}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:33:45,601] Trial 32 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 1.6849363809813175e-05, 'max_iter': 7011}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:33:46,063] Trial 33 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 7.054939625418251e-06, 'max_iter': 7932}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:33:46,505] Trial 34 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 0.00022405010670724864, 'max_iter': 4136}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:33:46,921] Trial 35 finished with value: 0.0 and parameters: {'penalty': 'elasticnet', 'C': 5.6531863670002665e-06, 'max_iter': 148, 'l1_ratio': 0.006164681835245721}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:33:47,209] Trial 36 finished with value: 0.0 and parameters: {'penalty': 'l1', 'C': 0.0005583569286105244, 'max_iter': 217, 'solver_l1': 'liblinear'}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:33:59,865] Trial 37 finished with value: 0.5077047316463643 and parameters: {'penalty': 'l2', 'C': 326.49401984858315, 'max_iter': 5825, 'solver_l2': 'sag'}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:00,280] Trial 38 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 2.170673856785127, 'max_iter': 1266}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:00,707] Trial 39 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 0.004315938302649112, 'max_iter': 2338}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:01,978] Trial 40 finished with value: 0.5062870107238342 and parameters: {'penalty': 'l2', 'C': 18.769519222442455, 'max_iter': 602, 'solver_l2': 'newton-cg'}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:02,360] Trial 41 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 0.5322567868281967, 'max_iter': 2172}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:02,750] Trial 42 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 1.2433083167806342, 'max_iter': 3124}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:03,126] Trial 43 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 0.07110404702666655, 'max_iter': 4918}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:03,522] Trial 44 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 2.937939826242614e-06, 'max_iter': 3390}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:05,075] Trial 45 finished with value: 0.40985380241314084 and parameters: {'penalty': 'elasticnet', 'C': 0.021943942729440478, 'max_iter': 6755, 'l1_ratio': 0.9786019461469602}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:05,453] Trial 46 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 4.522519982245096e-05, 'max_iter': 1348}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:09,536] Trial 47 finished with value: 0.5073607083750212 and parameters: {'penalty': 'l1', 'C': 3.312652864546117, 'max_iter': 2067, 'solver_l1': 'saga'}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:09,920] Trial 48 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 0.4326773219155866, 'max_iter': 490}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:10,308] Trial 49 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 258.1265921299816, 'max_iter': 814}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:10,690] Trial 50 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 40.810152973166325, 'max_iter': 4435}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:11,081] Trial 51 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 0.14838974890480514, 'max_iter': 2846}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:11,485] Trial 52 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 1.660746842386479, 'max_iter': 3731}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:11,884] Trial 53 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 13.891405803804716, 'max_iter': 2546}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:12,283] Trial 54 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 4.665830234710097, 'max_iter': 1888}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:12,678] Trial 55 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 0.9356561617386416, 'max_iter': 300}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:13,961] Trial 56 finished with value: 0.507466869643492 and parameters: {'penalty': 'l2', 'C': 63.694546109454954, 'max_iter': 5003, 'solver_l2': 'newton-cg'}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:14,349] Trial 57 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 0.009006422842058673, 'max_iter': 1734}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:14,538] Trial 58 finished with value: 0.0 and parameters: {'penalty': 'l1', 'C': 2.6067433538727778e-05, 'max_iter': 972, 'solver_l1': 'saga'}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:14,924] Trial 59 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 0.08819312192571374, 'max_iter': 7954}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:16,534] Trial 60 finished with value: 0.333323955763352 and parameters: {'penalty': 'elasticnet', 'C': 0.03342822609545077, 'max_iter': 466, 'l1_ratio': 1.5066706585226716e-06}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:16,927] Trial 61 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 1.3661323632451192e-06, 'max_iter': 681}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:17,310] Trial 62 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 1.4836447329217727e-06, 'max_iter': 1161}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:17,708] Trial 63 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 1.3653829410426483e-05, 'max_iter': 3830}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:18,100] Trial 64 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 0.000128666546163909, 'max_iter': 1467}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:18,496] Trial 65 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 6.8241700058213866, 'max_iter': 3163}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:18,888] Trial 66 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 5.203878125740336e-06, 'max_iter': 170}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:19,281] Trial 67 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 0.0003636907652992763, 'max_iter': 5938}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:19,502] Trial 68 finished with value: 0.0 and parameters: {'penalty': 'l2', 'C': 2.69169485323196e-06, 'max_iter': 112, 'solver_l2': 'lbfgs'}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:19,899] Trial 69 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 0.2772627805804881, 'max_iter': 291}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:20,284] Trial 70 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 2.635405515765073, 'max_iter': 962}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:20,680] Trial 71 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 29.322818072097462, 'max_iter': 2515}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:21,076] Trial 72 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 7.991697186811083, 'max_iter': 406}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:21,470] Trial 73 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 234.41783398169088, 'max_iter': 2918}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:23,374] Trial 74 finished with value: 0.5044236290073196 and parameters: {'penalty': 'l1', 'C': 0.66288415433059, 'max_iter': 3702, 'solver_l1': 'liblinear'}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:25,074] Trial 75 finished with value: 0.029688865248723318 and parameters: {'penalty': 'elasticnet', 'C': 0.001767840978577961, 'max_iter': 9801, 'l1_ratio': 0.00024229279310675864}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:25,464] Trial 76 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 139.3864873884193, 'max_iter': 4365}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:25,850] Trial 77 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 6.687548666795774e-05, 'max_iter': 8177}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:26,237] Trial 78 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 517.9508768675885, 'max_iter': 2116}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:26,636] Trial 79 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 1.272519227320891e-05, 'max_iter': 6785}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:27,023] Trial 80 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 17.52020231312909, 'max_iter': 1554}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:27,418] Trial 81 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 724.7857857663853, 'max_iter': 1831}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:27,795] Trial 82 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 116.6310754660293, 'max_iter': 1167}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:28,197] Trial 83 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 3.5240234328819323e-06, 'max_iter': 2309}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:28,587] Trial 84 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 1.263275635817253, 'max_iter': 687}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:29,273] Trial 85 finished with value: 0.4998018588412364 and parameters: {'penalty': 'l2', 'C': 3.101025873493198, 'max_iter': 1969, 'solver_l2': 'liblinear'}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:29,699] Trial 86 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 940.7109709945958, 'max_iter': 5473}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:33,671] Trial 87 finished with value: 0.5077047316463643 and parameters: {'penalty': 'l1', 'C': 43.11742639880158, 'max_iter': 3180, 'solver_l1': 'saga'}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:34,079] Trial 88 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 382.710483361172, 'max_iter': 2582}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:37,930] Trial 89 finished with value: 0.5045566661464199 and parameters: {'penalty': 'elasticnet', 'C': 10.082849854391673, 'max_iter': 1344, 'l1_ratio': 0.020287353740974495}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:38,328] Trial 90 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 2.1205442891571664e-06, 'max_iter': 3478}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:38,724] Trial 91 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 0.44188269439425665, 'max_iter': 7329}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:39,109] Trial 92 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 0.12792689257156364, 'max_iter': 8740}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:39,514] Trial 93 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 0.004769350906170949, 'max_iter': 6303}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:39,901] Trial 94 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 4.465613658034019, 'max_iter': 4026}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:40,299] Trial 95 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 1.7745125338638061, 'max_iter': 1646}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:40,682] Trial 96 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 1.0019902377063486e-06, 'max_iter': 9928}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:41,079] Trial 97 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 0.044456766575813704, 'max_iter': 4963}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:41,462] Trial 98 finished with value: 0.5086269596180564 and parameters: {'penalty': None, 'C': 0.25023361973663055, 'max_iter': 4591}. Best is trial 4 with value: 0.5086269596180564.
    [I 2024-05-31 07:34:42,331] Trial 99 finished with value: 0.30174041742859437 and parameters: {'penalty': 'l2', 'C': 0.01920069742089701, 'max_iter': 874, 'solver_l2': 'sag'}. Best is trial 4 with value: 0.5086269596180564.
    

**Random Forest**


```python
def tune_rf(trial):

    param = {
        'n_estimators' : trial.suggest_int("n_estimators", 10, 400, log=True),
        'max_depth' : trial.suggest_int("max_depth", 2, 64),
        'min_samples_split' : trial.suggest_int("min_samples_split", 2, 10),
        'min_samples_leaf' : trial.suggest_int("min_samples_leaf", 1, 10)
    }

    rf = RandomForestClassifier(**param)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=21)
    score = cross_val_score(rf, X_train_prepared, y_train, scoring='f1', cv=kfold).mean()
    
    return score
```


```python
sampler = TPESampler(seed=21)
study_rf = optuna.create_study(direction='maximize', sampler = sampler)
study_rf.optimize(tune_rf, n_trials = number_trials)
```

    [I 2024-05-31 07:34:42,346] A new study created in memory with name: no-name-79baf9bf-f7b3-4201-a22e-4ae1954d00c3
    [I 2024-05-31 07:34:43,399] Trial 0 finished with value: 0.5197520042305315 and parameters: {'n_estimators': 11, 'max_depth': 20, 'min_samples_split': 8, 'min_samples_leaf': 1}. Best is trial 0 with value: 0.5197520042305315.
    [I 2024-05-31 07:34:44,319] Trial 1 finished with value: 0.27879793274560694 and parameters: {'n_estimators': 21, 'max_depth': 5, 'min_samples_split': 4, 'min_samples_leaf': 7}. Best is trial 0 with value: 0.5197520042305315.
    [I 2024-05-31 07:34:46,828] Trial 2 finished with value: 0.49212347544574336 and parameters: {'n_estimators': 30, 'max_depth': 38, 'min_samples_split': 2, 'min_samples_leaf': 9}. Best is trial 0 with value: 0.5197520042305315.
    [I 2024-05-31 07:34:48,046] Trial 3 finished with value: 0.4404682506463386 and parameters: {'n_estimators': 16, 'max_depth': 13, 'min_samples_split': 6, 'min_samples_leaf': 9}. Best is trial 0 with value: 0.5197520042305315.
    [I 2024-05-31 07:35:04,434] Trial 4 finished with value: 0.5320372819790434 and parameters: {'n_estimators': 163, 'max_depth': 63, 'min_samples_split': 8, 'min_samples_leaf': 4}. Best is trial 4 with value: 0.5320372819790434.
    [I 2024-05-31 07:35:08,566] Trial 5 finished with value: 0.4818489754914517 and parameters: {'n_estimators': 44, 'max_depth': 46, 'min_samples_split': 4, 'min_samples_leaf': 9}. Best is trial 4 with value: 0.5320372819790434.
    [I 2024-05-31 07:35:38,742] Trial 6 finished with value: 0.542124768596341 and parameters: {'n_estimators': 289, 'max_depth': 49, 'min_samples_split': 6, 'min_samples_leaf': 2}. Best is trial 6 with value: 0.542124768596341.
    [I 2024-05-31 07:35:41,302] Trial 7 finished with value: 0.5034258736865611 and parameters: {'n_estimators': 29, 'max_depth': 19, 'min_samples_split': 4, 'min_samples_leaf': 5}. Best is trial 6 with value: 0.542124768596341.
    [I 2024-05-31 07:35:46,627] Trial 8 finished with value: 0.47459536026053817 and parameters: {'n_estimators': 73, 'max_depth': 15, 'min_samples_split': 9, 'min_samples_leaf': 8}. Best is trial 6 with value: 0.542124768596341.
    [I 2024-05-31 07:36:03,810] Trial 9 finished with value: 0.4976668331055131 and parameters: {'n_estimators': 228, 'max_depth': 18, 'min_samples_split': 7, 'min_samples_leaf': 7}. Best is trial 6 with value: 0.542124768596341.
    [I 2024-05-31 07:36:33,458] Trial 10 finished with value: 0.5525610783726829 and parameters: {'n_estimators': 330, 'max_depth': 58, 'min_samples_split': 6, 'min_samples_leaf': 1}. Best is trial 10 with value: 0.5525610783726829.
    [I 2024-05-31 07:37:08,965] Trial 11 finished with value: 0.5523317987109062 and parameters: {'n_estimators': 394, 'max_depth': 59, 'min_samples_split': 6, 'min_samples_leaf': 1}. Best is trial 10 with value: 0.5525610783726829.
    [I 2024-05-31 07:37:41,006] Trial 12 finished with value: 0.5303554765540053 and parameters: {'n_estimators': 388, 'max_depth': 64, 'min_samples_split': 10, 'min_samples_leaf': 3}. Best is trial 10 with value: 0.5525610783726829.
    [I 2024-05-31 07:37:52,533] Trial 13 finished with value: 0.5534250809765953 and parameters: {'n_estimators': 125, 'max_depth': 53, 'min_samples_split': 5, 'min_samples_leaf': 1}. Best is trial 13 with value: 0.5534250809765953.
    [I 2024-05-31 07:38:01,723] Trial 14 finished with value: 0.5245606363136237 and parameters: {'n_estimators': 108, 'max_depth': 52, 'min_samples_split': 5, 'min_samples_leaf': 3}. Best is trial 13 with value: 0.5534250809765953.
    [I 2024-05-31 07:38:14,185] Trial 15 finished with value: 0.5504840188317588 and parameters: {'n_estimators': 132, 'max_depth': 35, 'min_samples_split': 3, 'min_samples_leaf': 1}. Best is trial 13 with value: 0.5534250809765953.
    [I 2024-05-31 07:38:21,557] Trial 16 finished with value: 0.5280464111581983 and parameters: {'n_estimators': 84, 'max_depth': 43, 'min_samples_split': 5, 'min_samples_leaf': 3}. Best is trial 13 with value: 0.5534250809765953.
    [I 2024-05-31 07:38:36,197] Trial 17 finished with value: 0.516782635960467 and parameters: {'n_estimators': 181, 'max_depth': 54, 'min_samples_split': 7, 'min_samples_leaf': 5}. Best is trial 13 with value: 0.5534250809765953.
    [I 2024-05-31 07:38:41,038] Trial 18 finished with value: 0.5412597975034366 and parameters: {'n_estimators': 54, 'max_depth': 28, 'min_samples_split': 2, 'min_samples_leaf': 2}. Best is trial 13 with value: 0.5534250809765953.
    [I 2024-05-31 07:39:02,561] Trial 19 finished with value: 0.5382718280722469 and parameters: {'n_estimators': 246, 'max_depth': 29, 'min_samples_split': 5, 'min_samples_leaf': 2}. Best is trial 13 with value: 0.5534250809765953.
    [I 2024-05-31 07:39:11,623] Trial 20 finished with value: 0.5224308529041892 and parameters: {'n_estimators': 110, 'max_depth': 55, 'min_samples_split': 7, 'min_samples_leaf': 4}. Best is trial 13 with value: 0.5534250809765953.
    [I 2024-05-31 07:39:47,510] Trial 21 finished with value: 0.550727554711206 and parameters: {'n_estimators': 399, 'max_depth': 60, 'min_samples_split': 6, 'min_samples_leaf': 1}. Best is trial 13 with value: 0.5534250809765953.
    [I 2024-05-31 07:40:14,348] Trial 22 finished with value: 0.5494604367028888 and parameters: {'n_estimators': 295, 'max_depth': 56, 'min_samples_split': 5, 'min_samples_leaf': 1}. Best is trial 13 with value: 0.5534250809765953.
    [I 2024-05-31 07:40:30,636] Trial 23 finished with value: 0.5446943562330094 and parameters: {'n_estimators': 187, 'max_depth': 41, 'min_samples_split': 7, 'min_samples_leaf': 2}. Best is trial 13 with value: 0.5534250809765953.
    [I 2024-05-31 07:40:59,640] Trial 24 finished with value: 0.5455722630905278 and parameters: {'n_estimators': 328, 'max_depth': 59, 'min_samples_split': 8, 'min_samples_leaf': 1}. Best is trial 13 with value: 0.5534250809765953.
    [I 2024-05-31 07:41:18,436] Trial 25 finished with value: 0.5294847563066287 and parameters: {'n_estimators': 229, 'max_depth': 47, 'min_samples_split': 6, 'min_samples_leaf': 4}. Best is trial 13 with value: 0.5534250809765953.
    [I 2024-05-31 07:41:30,484] Trial 26 finished with value: 0.537930810771678 and parameters: {'n_estimators': 141, 'max_depth': 51, 'min_samples_split': 4, 'min_samples_leaf': 3}. Best is trial 13 with value: 0.5534250809765953.
    [I 2024-05-31 07:41:38,992] Trial 27 finished with value: 0.5438228036875412 and parameters: {'n_estimators': 96, 'max_depth': 59, 'min_samples_split': 3, 'min_samples_leaf': 2}. Best is trial 13 with value: 0.5534250809765953.
    [I 2024-05-31 07:41:55,385] Trial 28 finished with value: 0.512720327343638 and parameters: {'n_estimators': 207, 'max_depth': 64, 'min_samples_split': 6, 'min_samples_leaf': 6}. Best is trial 13 with value: 0.5534250809765953.
    [I 2024-05-31 07:42:22,114] Trial 29 finished with value: 0.546365145146885 and parameters: {'n_estimators': 304, 'max_depth': 44, 'min_samples_split': 9, 'min_samples_leaf': 1}. Best is trial 13 with value: 0.5534250809765953.
    [I 2024-05-31 07:42:27,036] Trial 30 finished with value: 0.4856612080714127 and parameters: {'n_estimators': 63, 'max_depth': 58, 'min_samples_split': 5, 'min_samples_leaf': 10}. Best is trial 13 with value: 0.5534250809765953.
    [I 2024-05-31 07:42:57,894] Trial 31 finished with value: 0.5483971143290334 and parameters: {'n_estimators': 343, 'max_depth': 60, 'min_samples_split': 6, 'min_samples_leaf': 1}. Best is trial 13 with value: 0.5534250809765953.
    [I 2024-05-31 07:43:20,366] Trial 32 finished with value: 0.5477277120082575 and parameters: {'n_estimators': 253, 'max_depth': 51, 'min_samples_split': 7, 'min_samples_leaf': 1}. Best is trial 13 with value: 0.5534250809765953.
    [I 2024-05-31 07:43:20,727] Trial 33 finished with value: 0.16544892445260637 and parameters: {'n_estimators': 10, 'max_depth': 3, 'min_samples_split': 5, 'min_samples_leaf': 2}. Best is trial 13 with value: 0.5534250809765953.
    [I 2024-05-31 07:43:56,113] Trial 34 finished with value: 0.5493430144451732 and parameters: {'n_estimators': 392, 'max_depth': 60, 'min_samples_split': 6, 'min_samples_leaf': 1}. Best is trial 13 with value: 0.5534250809765953.
    [I 2024-05-31 07:44:30,818] Trial 35 finished with value: 0.5447629277654924 and parameters: {'n_estimators': 394, 'max_depth': 56, 'min_samples_split': 3, 'min_samples_leaf': 2}. Best is trial 13 with value: 0.5534250809765953.
    [I 2024-05-31 07:44:42,991] Trial 36 finished with value: 0.5332347152427261 and parameters: {'n_estimators': 145, 'max_depth': 38, 'min_samples_split': 8, 'min_samples_leaf': 3}. Best is trial 13 with value: 0.5534250809765953.
    [I 2024-05-31 07:44:46,506] Trial 37 finished with value: 0.5346157223928742 and parameters: {'n_estimators': 38, 'max_depth': 62, 'min_samples_split': 6, 'min_samples_leaf': 1}. Best is trial 13 with value: 0.5534250809765953.
    [I 2024-05-31 07:44:47,950] Trial 38 finished with value: 0.522228648552099 and parameters: {'n_estimators': 16, 'max_depth': 47, 'min_samples_split': 4, 'min_samples_leaf': 4}. Best is trial 13 with value: 0.5534250809765953.
    [I 2024-05-31 07:45:00,453] Trial 39 finished with value: 0.3375226777609707 and parameters: {'n_estimators': 256, 'max_depth': 8, 'min_samples_split': 6, 'min_samples_leaf': 2}. Best is trial 13 with value: 0.5534250809765953.
    [I 2024-05-31 07:45:26,024] Trial 40 finished with value: 0.5474453316213919 and parameters: {'n_estimators': 287, 'max_depth': 50, 'min_samples_split': 7, 'min_samples_leaf': 1}. Best is trial 13 with value: 0.5534250809765953.
    [I 2024-05-31 07:45:38,421] Trial 41 finished with value: 0.5529078140401014 and parameters: {'n_estimators': 132, 'max_depth': 27, 'min_samples_split': 3, 'min_samples_leaf': 1}. Best is trial 13 with value: 0.5534250809765953.
    [I 2024-05-31 07:45:52,158] Trial 42 finished with value: 0.5442757292176282 and parameters: {'n_estimators': 156, 'max_depth': 25, 'min_samples_split': 2, 'min_samples_leaf': 2}. Best is trial 13 with value: 0.5534250809765953.
    [I 2024-05-31 07:46:10,075] Trial 43 finished with value: 0.548582781810832 and parameters: {'n_estimators': 196, 'max_depth': 26, 'min_samples_split': 4, 'min_samples_leaf': 1}. Best is trial 13 with value: 0.5534250809765953.
    [I 2024-05-31 07:46:20,474] Trial 44 finished with value: 0.5273419341011534 and parameters: {'n_estimators': 124, 'max_depth': 21, 'min_samples_split': 3, 'min_samples_leaf': 3}. Best is trial 13 with value: 0.5534250809765953.
    [I 2024-05-31 07:46:46,946] Trial 45 finished with value: 0.5018912151330931 and parameters: {'n_estimators': 338, 'max_depth': 31, 'min_samples_split': 4, 'min_samples_leaf': 7}. Best is trial 13 with value: 0.5534250809765953.
    [I 2024-05-31 07:46:53,423] Trial 46 finished with value: 0.5476368559436875 and parameters: {'n_estimators': 72, 'max_depth': 34, 'min_samples_split': 5, 'min_samples_leaf': 2}. Best is trial 13 with value: 0.5534250809765953.
    [I 2024-05-31 07:46:57,784] Trial 47 finished with value: 0.540109699524256 and parameters: {'n_estimators': 49, 'max_depth': 22, 'min_samples_split': 7, 'min_samples_leaf': 1}. Best is trial 13 with value: 0.5534250809765953.
    [I 2024-05-31 07:46:59,716] Trial 48 finished with value: 0.4881364134856767 and parameters: {'n_estimators': 23, 'max_depth': 54, 'min_samples_split': 5, 'min_samples_leaf': 8}. Best is trial 13 with value: 0.5534250809765953.
    [I 2024-05-31 07:47:07,543] Trial 49 finished with value: 0.5378224694960883 and parameters: {'n_estimators': 90, 'max_depth': 61, 'min_samples_split': 8, 'min_samples_leaf': 2}. Best is trial 13 with value: 0.5534250809765953.
    [I 2024-05-31 07:47:22,470] Trial 50 finished with value: 0.537439564840164 and parameters: {'n_estimators': 175, 'max_depth': 38, 'min_samples_split': 2, 'min_samples_leaf': 3}. Best is trial 13 with value: 0.5534250809765953.
    [I 2024-05-31 07:47:30,856] Trial 51 finished with value: 0.4573471641067329 and parameters: {'n_estimators': 125, 'max_depth': 12, 'min_samples_split': 3, 'min_samples_leaf': 1}. Best is trial 13 with value: 0.5534250809765953.
    [I 2024-05-31 07:47:38,065] Trial 52 finished with value: 0.5377686245687763 and parameters: {'n_estimators': 76, 'max_depth': 35, 'min_samples_split': 3, 'min_samples_leaf': 1}. Best is trial 13 with value: 0.5534250809765953.
    [I 2024-05-31 07:48:03,138] Trial 53 finished with value: 0.5531793987780055 and parameters: {'n_estimators': 272, 'max_depth': 57, 'min_samples_split': 4, 'min_samples_leaf': 1}. Best is trial 13 with value: 0.5534250809765953.
    [I 2024-05-31 07:48:27,884] Trial 54 finished with value: 0.5515723954805105 and parameters: {'n_estimators': 269, 'max_depth': 57, 'min_samples_split': 4, 'min_samples_leaf': 1}. Best is trial 13 with value: 0.5534250809765953.
    [I 2024-05-31 07:48:51,207] Trial 55 finished with value: 0.5435985843598508 and parameters: {'n_estimators': 263, 'max_depth': 53, 'min_samples_split': 4, 'min_samples_leaf': 2}. Best is trial 13 with value: 0.5534250809765953.
    [I 2024-05-31 07:49:10,701] Trial 56 finished with value: 0.547892581087742 and parameters: {'n_estimators': 215, 'max_depth': 57, 'min_samples_split': 5, 'min_samples_leaf': 1}. Best is trial 13 with value: 0.5534250809765953.
    [I 2024-05-31 07:49:24,672] Trial 57 finished with value: 0.5311469302759959 and parameters: {'n_estimators': 164, 'max_depth': 48, 'min_samples_split': 4, 'min_samples_leaf': 3}. Best is trial 13 with value: 0.5534250809765953.
    [I 2024-05-31 07:49:51,692] Trial 58 finished with value: 0.5427777338782692 and parameters: {'n_estimators': 305, 'max_depth': 63, 'min_samples_split': 4, 'min_samples_leaf': 2}. Best is trial 13 with value: 0.5534250809765953.
    [I 2024-05-31 07:50:01,208] Trial 59 finished with value: 0.5532243599401526 and parameters: {'n_estimators': 104, 'max_depth': 55, 'min_samples_split': 5, 'min_samples_leaf': 1}. Best is trial 13 with value: 0.5534250809765953.
    [I 2024-05-31 07:50:09,717] Trial 60 finished with value: 0.5346043009319412 and parameters: {'n_estimators': 96, 'max_depth': 44, 'min_samples_split': 5, 'min_samples_leaf': 2}. Best is trial 13 with value: 0.5534250809765953.
    [I 2024-05-31 07:50:19,628] Trial 61 finished with value: 0.5457584641025466 and parameters: {'n_estimators': 109, 'max_depth': 55, 'min_samples_split': 5, 'min_samples_leaf': 1}. Best is trial 13 with value: 0.5534250809765953.
    [I 2024-05-31 07:50:51,150] Trial 62 finished with value: 0.5507293650168881 and parameters: {'n_estimators': 342, 'max_depth': 53, 'min_samples_split': 4, 'min_samples_leaf': 1}. Best is trial 13 with value: 0.5534250809765953.
    [I 2024-05-31 07:51:11,044] Trial 63 finished with value: 0.5564932033089484 and parameters: {'n_estimators': 219, 'max_depth': 57, 'min_samples_split': 5, 'min_samples_leaf': 1}. Best is trial 63 with value: 0.5564932033089484.
    [I 2024-05-31 07:51:30,076] Trial 64 finished with value: 0.5461500439955335 and parameters: {'n_estimators': 220, 'max_depth': 58, 'min_samples_split': 6, 'min_samples_leaf': 2}. Best is trial 63 with value: 0.5564932033089484.
    [I 2024-05-31 07:51:40,893] Trial 65 finished with value: 0.5468484265068986 and parameters: {'n_estimators': 119, 'max_depth': 64, 'min_samples_split': 5, 'min_samples_leaf': 1}. Best is trial 63 with value: 0.5564932033089484.
    [I 2024-05-31 07:51:55,754] Trial 66 finished with value: 0.5101331104991086 and parameters: {'n_estimators': 187, 'max_depth': 49, 'min_samples_split': 6, 'min_samples_leaf': 6}. Best is trial 63 with value: 0.5564932033089484.
    [I 2024-05-31 07:52:09,303] Trial 67 finished with value: 0.5457207323170417 and parameters: {'n_estimators': 149, 'max_depth': 62, 'min_samples_split': 5, 'min_samples_leaf': 1}. Best is trial 63 with value: 0.5564932033089484.
    [I 2024-05-31 07:52:30,196] Trial 68 finished with value: 0.5398210760334825 and parameters: {'n_estimators': 239, 'max_depth': 55, 'min_samples_split': 6, 'min_samples_leaf': 2}. Best is trial 63 with value: 0.5564932033089484.
    [I 2024-05-31 07:52:35,925] Trial 69 finished with value: 0.5428595030127366 and parameters: {'n_estimators': 63, 'max_depth': 53, 'min_samples_split': 6, 'min_samples_leaf': 1}. Best is trial 63 with value: 0.5564932033089484.
    [I 2024-05-31 07:52:49,794] Trial 70 finished with value: 0.518132603329826 and parameters: {'n_estimators': 171, 'max_depth': 51, 'min_samples_split': 3, 'min_samples_leaf': 5}. Best is trial 63 with value: 0.5564932033089484.
    [I 2024-05-31 07:53:14,807] Trial 71 finished with value: 0.5492494494772517 and parameters: {'n_estimators': 272, 'max_depth': 57, 'min_samples_split': 4, 'min_samples_leaf': 1}. Best is trial 63 with value: 0.5564932033089484.
    [I 2024-05-31 07:53:46,603] Trial 72 finished with value: 0.5484694108758571 and parameters: {'n_estimators': 351, 'max_depth': 59, 'min_samples_split': 5, 'min_samples_leaf': 1}. Best is trial 63 with value: 0.5564932033089484.
    [I 2024-05-31 07:54:13,599] Trial 73 finished with value: 0.5532984506404194 and parameters: {'n_estimators': 293, 'max_depth': 57, 'min_samples_split': 4, 'min_samples_leaf': 1}. Best is trial 63 with value: 0.5564932033089484.
    [I 2024-05-31 07:54:40,797] Trial 74 finished with value: 0.5423564732822801 and parameters: {'n_estimators': 309, 'max_depth': 61, 'min_samples_split': 5, 'min_samples_leaf': 2}. Best is trial 63 with value: 0.5564932033089484.
    [I 2024-05-31 07:55:14,365] Trial 75 finished with value: 0.554365316534051 and parameters: {'n_estimators': 357, 'max_depth': 56, 'min_samples_split': 3, 'min_samples_leaf': 1}. Best is trial 63 with value: 0.5564932033089484.
    [I 2024-05-31 07:55:33,936] Trial 76 finished with value: 0.5535666552993661 and parameters: {'n_estimators': 202, 'max_depth': 46, 'min_samples_split': 2, 'min_samples_leaf': 1}. Best is trial 63 with value: 0.5564932033089484.
    [I 2024-05-31 07:55:47,470] Trial 77 finished with value: 0.5443212244036622 and parameters: {'n_estimators': 141, 'max_depth': 49, 'min_samples_split': 2, 'min_samples_leaf': 1}. Best is trial 63 with value: 0.5564932033089484.
    [I 2024-05-31 07:56:05,108] Trial 78 finished with value: 0.5438225631962104 and parameters: {'n_estimators': 199, 'max_depth': 52, 'min_samples_split': 2, 'min_samples_leaf': 2}. Best is trial 63 with value: 0.5564932033089484.
    [I 2024-05-31 07:56:27,781] Trial 79 finished with value: 0.5519676882540443 and parameters: {'n_estimators': 240, 'max_depth': 41, 'min_samples_split': 3, 'min_samples_leaf': 1}. Best is trial 63 with value: 0.5564932033089484.
    [I 2024-05-31 07:56:35,508] Trial 80 finished with value: 0.47574369740927314 and parameters: {'n_estimators': 101, 'max_depth': 30, 'min_samples_split': 3, 'min_samples_leaf': 10}. Best is trial 63 with value: 0.5564932033089484.
    [I 2024-05-31 07:57:03,288] Trial 81 finished with value: 0.5452780081558524 and parameters: {'n_estimators': 288, 'max_depth': 56, 'min_samples_split': 2, 'min_samples_leaf': 1}. Best is trial 63 with value: 0.5564932033089484.
    [I 2024-05-31 07:57:38,622] Trial 82 finished with value: 0.5544648563866906 and parameters: {'n_estimators': 375, 'max_depth': 46, 'min_samples_split': 3, 'min_samples_leaf': 1}. Best is trial 63 with value: 0.5564932033089484.
    [I 2024-05-31 07:58:11,970] Trial 83 finished with value: 0.5440119931921613 and parameters: {'n_estimators': 378, 'max_depth': 27, 'min_samples_split': 3, 'min_samples_leaf': 2}. Best is trial 63 with value: 0.5564932033089484.
    [I 2024-05-31 07:58:32,339] Trial 84 finished with value: 0.5497719054690775 and parameters: {'n_estimators': 211, 'max_depth': 45, 'min_samples_split': 2, 'min_samples_leaf': 1}. Best is trial 63 with value: 0.5564932033089484.
    [I 2024-05-31 07:58:44,892] Trial 85 finished with value: 0.5490883878702139 and parameters: {'n_estimators': 134, 'max_depth': 24, 'min_samples_split': 3, 'min_samples_leaf': 1}. Best is trial 63 with value: 0.5564932033089484.
    [I 2024-05-31 07:59:17,040] Trial 86 finished with value: 0.5450429450405829 and parameters: {'n_estimators': 362, 'max_depth': 42, 'min_samples_split': 4, 'min_samples_leaf': 2}. Best is trial 63 with value: 0.5564932033089484.
    [I 2024-05-31 07:59:39,241] Trial 87 finished with value: 0.5545569061643179 and parameters: {'n_estimators': 234, 'max_depth': 47, 'min_samples_split': 3, 'min_samples_leaf': 1}. Best is trial 63 with value: 0.5564932033089484.
    [I 2024-05-31 08:00:04,027] Trial 88 finished with value: 0.4988413180012977 and parameters: {'n_estimators': 321, 'max_depth': 45, 'min_samples_split': 4, 'min_samples_leaf': 8}. Best is trial 63 with value: 0.5564932033089484.
    [I 2024-05-31 08:00:26,665] Trial 89 finished with value: 0.5503214643390555 and parameters: {'n_estimators': 235, 'max_depth': 47, 'min_samples_split': 2, 'min_samples_leaf': 1}. Best is trial 63 with value: 0.5564932033089484.
    [I 2024-05-31 08:00:51,439] Trial 90 finished with value: 0.5404777691491599 and parameters: {'n_estimators': 279, 'max_depth': 50, 'min_samples_split': 3, 'min_samples_leaf': 2}. Best is trial 63 with value: 0.5564932033089484.
    [I 2024-05-31 08:01:08,438] Trial 91 finished with value: 0.5456607128104375 and parameters: {'n_estimators': 179, 'max_depth': 52, 'min_samples_split': 3, 'min_samples_leaf': 1}. Best is trial 63 with value: 0.5564932033089484.
    [I 2024-05-31 08:01:32,252] Trial 92 finished with value: 0.5535202473556401 and parameters: {'n_estimators': 253, 'max_depth': 39, 'min_samples_split': 3, 'min_samples_leaf': 1}. Best is trial 63 with value: 0.5564932033089484.
    [I 2024-05-31 08:02:02,668] Trial 93 finished with value: 0.5519969040876116 and parameters: {'n_estimators': 321, 'max_depth': 40, 'min_samples_split': 3, 'min_samples_leaf': 1}. Best is trial 63 with value: 0.5564932033089484.
    [I 2024-05-31 08:02:27,892] Trial 94 finished with value: 0.5463831220149582 and parameters: {'n_estimators': 259, 'max_depth': 39, 'min_samples_split': 2, 'min_samples_leaf': 1}. Best is trial 63 with value: 0.5564932033089484.
    [I 2024-05-31 08:02:46,509] Trial 95 finished with value: 0.5532666534628975 and parameters: {'n_estimators': 202, 'max_depth': 46, 'min_samples_split': 4, 'min_samples_leaf': 1}. Best is trial 63 with value: 0.5564932033089484.
    [I 2024-05-31 08:03:03,642] Trial 96 finished with value: 0.5441630826708383 and parameters: {'n_estimators': 193, 'max_depth': 46, 'min_samples_split': 3, 'min_samples_leaf': 2}. Best is trial 63 with value: 0.5564932033089484.
    [I 2024-05-31 08:03:24,195] Trial 97 finished with value: 0.5512010262089537 and parameters: {'n_estimators': 224, 'max_depth': 43, 'min_samples_split': 4, 'min_samples_leaf': 1}. Best is trial 63 with value: 0.5564932033089484.
    [I 2024-05-31 08:03:37,966] Trial 98 finished with value: 0.547790098959501 and parameters: {'n_estimators': 156, 'max_depth': 49, 'min_samples_split': 3, 'min_samples_leaf': 2}. Best is trial 63 with value: 0.5564932033089484.
    [I 2024-05-31 08:04:11,252] Trial 99 finished with value: 0.5520353365476307 and parameters: {'n_estimators': 362, 'max_depth': 36, 'min_samples_split': 4, 'min_samples_leaf': 1}. Best is trial 63 with value: 0.5564932033089484.
    

**XgBoost**


```python
def tune_xgb(trial):
    param = {
        "verbosity": 0,
        "objective": "binary:logistic",
        "tree_method": "exact",
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
    }

    if param["booster"] in ["gbtree", "dart"]:
        param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    xgb = XGBClassifier(**param)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=21)
    score = cross_val_score(xgb, X_train_prepared, y_train, scoring='f1', cv=kfold).mean()

    return score
```


```python
sampler = TPESampler(seed=21)
study_xgb = optuna.create_study(direction='maximize', sampler = sampler)
study_xgb.optimize(tune_xgb, n_trials = number_trials)
```

    [I 2024-05-31 08:04:11,271] A new study created in memory with name: no-name-12e3c1ef-868a-4e49-ac20-e277069a5aed
    [I 2024-05-31 08:04:19,839] Trial 0 finished with value: 0.0 and parameters: {'booster': 'dart', 'lambda': 1.4891210381075092e-08, 'alpha': 4.439991330753759e-07, 'subsample': 0.24061860535630145, 'colsample_bytree': 0.4418175151716935, 'max_depth': 7, 'min_child_weight': 4, 'eta': 0.0004663695745275388, 'gamma': 3.602198465642853e-08, 'grow_policy': 'depthwise', 'sample_type': 'weighted', 'normalize_type': 'tree', 'rate_drop': 0.5806053091762127, 'skip_drop': 0.011869171969835073}. Best is trial 0 with value: 0.0.
    [I 2024-05-31 08:04:33,045] Trial 1 finished with value: 0.0 and parameters: {'booster': 'dart', 'lambda': 1.4633835924425082e-06, 'alpha': 0.06804919248114097, 'subsample': 0.9305311756894445, 'colsample_bytree': 0.808604855731685, 'max_depth': 7, 'min_child_weight': 3, 'eta': 2.452100878199486e-06, 'gamma': 1.8687205446770345e-06, 'grow_policy': 'lossguide', 'sample_type': 'uniform', 'normalize_type': 'tree', 'rate_drop': 0.06227901097707788, 'skip_drop': 1.3809388145445195e-06}. Best is trial 0 with value: 0.0.
    [I 2024-05-31 08:04:33,827] Trial 2 finished with value: 0.0 and parameters: {'booster': 'gblinear', 'lambda': 1.9179329862379416e-05, 'alpha': 0.03028728860591447, 'subsample': 0.6802954123057079, 'colsample_bytree': 0.9186722708677555}. Best is trial 0 with value: 0.0.
    [I 2024-05-31 08:04:45,742] Trial 3 finished with value: 0.0 and parameters: {'booster': 'dart', 'lambda': 0.020303043687675028, 'alpha': 0.12625352069467183, 'subsample': 0.7590886311619907, 'colsample_bytree': 0.5635122460885997, 'max_depth': 5, 'min_child_weight': 9, 'eta': 1.3101386034623663e-07, 'gamma': 1.6231998274004853e-05, 'grow_policy': 'lossguide', 'sample_type': 'uniform', 'normalize_type': 'tree', 'rate_drop': 0.0021161633190961444, 'skip_drop': 8.972221235087032e-07}. Best is trial 0 with value: 0.0.
    [I 2024-05-31 08:04:47,157] Trial 4 finished with value: 0.0 and parameters: {'booster': 'gbtree', 'lambda': 4.843638767934291e-05, 'alpha': 0.08178732359407656, 'subsample': 0.7617314336032905, 'colsample_bytree': 0.33438005874636556, 'max_depth': 3, 'min_child_weight': 9, 'eta': 8.334853576944574e-06, 'gamma': 1.1175864183059091e-05, 'grow_policy': 'lossguide'}. Best is trial 0 with value: 0.0.
    [I 2024-05-31 08:04:48,394] Trial 5 finished with value: 0.0 and parameters: {'booster': 'gblinear', 'lambda': 0.07223353735096283, 'alpha': 2.593202919343843e-07, 'subsample': 0.49881687892027726, 'colsample_bytree': 0.5058230088456417}. Best is trial 0 with value: 0.0.
    [I 2024-05-31 08:04:51,371] Trial 6 finished with value: 0.0 and parameters: {'booster': 'gbtree', 'lambda': 0.0001533280428192742, 'alpha': 0.2851152589563356, 'subsample': 0.3972607472055543, 'colsample_bytree': 0.3666424408024065, 'max_depth': 9, 'min_child_weight': 4, 'eta': 0.0025301083930651344, 'gamma': 0.24505546148107646, 'grow_policy': 'lossguide'}. Best is trial 0 with value: 0.0.
    [I 2024-05-31 08:04:52,190] Trial 7 finished with value: 0.0 and parameters: {'booster': 'gblinear', 'lambda': 0.0033432916473246552, 'alpha': 0.017138508886426297, 'subsample': 0.23778330400450065, 'colsample_bytree': 0.25964896093635315}. Best is trial 0 with value: 0.0.
    [I 2024-05-31 08:04:53,441] Trial 8 finished with value: 0.35038865902073735 and parameters: {'booster': 'gblinear', 'lambda': 0.0008281492178157587, 'alpha': 1.609933376620524e-07, 'subsample': 0.5693857753999938, 'colsample_bytree': 0.9842106230778074}. Best is trial 8 with value: 0.35038865902073735.
    [I 2024-05-31 08:04:56,051] Trial 9 finished with value: 0.0 and parameters: {'booster': 'gbtree', 'lambda': 0.21586834629250246, 'alpha': 0.007253371114705163, 'subsample': 0.20168040071427226, 'colsample_bytree': 0.7383433999813551, 'max_depth': 7, 'min_child_weight': 6, 'eta': 3.0828550114298045e-05, 'gamma': 8.062609821711576e-06, 'grow_policy': 'lossguide'}. Best is trial 8 with value: 0.35038865902073735.
    [I 2024-05-31 08:04:57,294] Trial 10 finished with value: 0.28900573842951427 and parameters: {'booster': 'gblinear', 'lambda': 0.0022554303524098315, 'alpha': 3.2297107867436235e-05, 'subsample': 0.5349308938449094, 'colsample_bytree': 0.9967953839937721}. Best is trial 8 with value: 0.35038865902073735.
    [I 2024-05-31 08:04:58,522] Trial 11 finished with value: 0.2986854029657532 and parameters: {'booster': 'gblinear', 'lambda': 0.0018418301576249964, 'alpha': 2.3784507840011378e-05, 'subsample': 0.5516109524252466, 'colsample_bytree': 0.9975213825649049}. Best is trial 8 with value: 0.35038865902073735.
    [I 2024-05-31 08:04:59,801] Trial 12 finished with value: 0.3576263453403376 and parameters: {'booster': 'gblinear', 'lambda': 0.0007071489247892708, 'alpha': 1.2844544978877613e-08, 'subsample': 0.41556673270024347, 'colsample_bytree': 0.744557325824023}. Best is trial 12 with value: 0.3576263453403376.
    [I 2024-05-31 08:05:01,028] Trial 13 finished with value: 0.5046584694697258 and parameters: {'booster': 'gblinear', 'lambda': 1.4569132653558605e-06, 'alpha': 1.8652553542318272e-08, 'subsample': 0.3915483772461418, 'colsample_bytree': 0.7099724764128661}. Best is trial 13 with value: 0.5046584694697258.
    [I 2024-05-31 08:05:02,313] Trial 14 finished with value: 0.5045060721635093 and parameters: {'booster': 'gblinear', 'lambda': 1.08044698741841e-06, 'alpha': 2.8653535223227427e-08, 'subsample': 0.37022128880524624, 'colsample_bytree': 0.7096378209268792}. Best is trial 13 with value: 0.5046584694697258.
    [I 2024-05-31 08:05:03,531] Trial 15 finished with value: 0.5054939895005318 and parameters: {'booster': 'gblinear', 'lambda': 3.61696782470072e-07, 'alpha': 2.393471403169546e-08, 'subsample': 0.3790696535101624, 'colsample_bytree': 0.6371560232890605}. Best is trial 15 with value: 0.5054939895005318.
    [I 2024-05-31 08:05:04,834] Trial 16 finished with value: 0.5051552594918783 and parameters: {'booster': 'gblinear', 'lambda': 1.514608721131437e-08, 'alpha': 3.4075553067287394e-06, 'subsample': 0.34431632186274264, 'colsample_bytree': 0.6556709037738223}. Best is trial 15 with value: 0.5054939895005318.
    [I 2024-05-31 08:05:06,055] Trial 17 finished with value: 0.46996535588703836 and parameters: {'booster': 'gblinear', 'lambda': 1.378522683147118e-08, 'alpha': 0.0007953062403245339, 'subsample': 0.2994192911834013, 'colsample_bytree': 0.6193526144750192}. Best is trial 15 with value: 0.5054939895005318.
    [I 2024-05-31 08:05:17,107] Trial 18 finished with value: 0.571121498572232 and parameters: {'booster': 'dart', 'lambda': 1.435032195647094e-07, 'alpha': 3.92809158417339e-06, 'subsample': 0.4747741747575738, 'colsample_bytree': 0.5896623941024738, 'max_depth': 3, 'min_child_weight': 7, 'eta': 0.27470633075833706, 'gamma': 0.04445235949680559, 'grow_policy': 'depthwise', 'sample_type': 'weighted', 'normalize_type': 'forest', 'rate_drop': 1.1426502568143814e-08, 'skip_drop': 0.42410163676311446}. Best is trial 18 with value: 0.571121498572232.
    [I 2024-05-31 08:05:29,035] Trial 19 finished with value: 0.5704888018955309 and parameters: {'booster': 'dart', 'lambda': 1.328391228899601e-07, 'alpha': 2.755403507297975e-06, 'subsample': 0.4816118550348242, 'colsample_bytree': 0.8448723962870639, 'max_depth': 3, 'min_child_weight': 7, 'eta': 0.3134468709798096, 'gamma': 0.06086372255097973, 'grow_policy': 'depthwise', 'sample_type': 'weighted', 'normalize_type': 'forest', 'rate_drop': 2.677963064141802e-08, 'skip_drop': 0.7631299586602269}. Best is trial 18 with value: 0.571121498572232.
    [I 2024-05-31 08:05:40,064] Trial 20 finished with value: 0.5561332145117067 and parameters: {'booster': 'dart', 'lambda': 1.2226888497637443e-07, 'alpha': 0.00034292122619143054, 'subsample': 0.6515802290615623, 'colsample_bytree': 0.47294741349477654, 'max_depth': 3, 'min_child_weight': 7, 'eta': 0.7091584881497268, 'gamma': 0.0783096761117649, 'grow_policy': 'depthwise', 'sample_type': 'weighted', 'normalize_type': 'forest', 'rate_drop': 1.0975166904163055e-08, 'skip_drop': 0.7737056011543345}. Best is trial 18 with value: 0.571121498572232.
    [I 2024-05-31 08:05:51,223] Trial 21 finished with value: 0.5552879028639692 and parameters: {'booster': 'dart', 'lambda': 1.2317502798671046e-07, 'alpha': 0.00039108658639142615, 'subsample': 0.6570578799178186, 'colsample_bytree': 0.5018459934696545, 'max_depth': 3, 'min_child_weight': 7, 'eta': 0.7105874010435005, 'gamma': 0.07962786765438828, 'grow_policy': 'depthwise', 'sample_type': 'weighted', 'normalize_type': 'forest', 'rate_drop': 1.167737159406342e-08, 'skip_drop': 0.6786902461724312}. Best is trial 18 with value: 0.571121498572232.
    [I 2024-05-31 08:06:02,582] Trial 22 finished with value: 0.546550768814348 and parameters: {'booster': 'dart', 'lambda': 1.3980756666142266e-07, 'alpha': 3.711866481909634e-06, 'subsample': 0.4641060028626045, 'colsample_bytree': 0.8255000935294523, 'max_depth': 3, 'min_child_weight': 7, 'eta': 0.9189524294893106, 'gamma': 0.009538816878515296, 'grow_policy': 'depthwise', 'sample_type': 'weighted', 'normalize_type': 'forest', 'rate_drop': 1.5719522125124535e-08, 'skip_drop': 0.9210135084975125}. Best is trial 18 with value: 0.571121498572232.
    [I 2024-05-31 08:06:13,565] Trial 23 finished with value: 0.29786550948070156 and parameters: {'booster': 'dart', 'lambda': 1.1724980225237876e-05, 'alpha': 5.075192746869644e-06, 'subsample': 0.6262767645964264, 'colsample_bytree': 0.5448614693358667, 'max_depth': 3, 'min_child_weight': 7, 'eta': 0.018891438673778117, 'gamma': 0.0037656614977646646, 'grow_policy': 'depthwise', 'sample_type': 'weighted', 'normalize_type': 'forest', 'rate_drop': 1.3927043637806246e-06, 'skip_drop': 0.013919045034981226}. Best is trial 18 with value: 0.571121498572232.
    [I 2024-05-31 08:06:26,163] Trial 24 finished with value: 0.46314808251588835 and parameters: {'booster': 'dart', 'lambda': 1.0765932297460236e-07, 'alpha': 0.00022301504504994985, 'subsample': 0.7048789392252623, 'colsample_bytree': 0.42970906762619054, 'max_depth': 5, 'min_child_weight': 6, 'eta': 0.04066413867205103, 'gamma': 0.0016297510762281898, 'grow_policy': 'depthwise', 'sample_type': 'weighted', 'normalize_type': 'forest', 'rate_drop': 6.320618775901813e-07, 'skip_drop': 0.027438375238453258}. Best is trial 18 with value: 0.571121498572232.
    [I 2024-05-31 08:06:38,730] Trial 25 finished with value: 0.5712751155075699 and parameters: {'booster': 'dart', 'lambda': 5.598212751284751e-06, 'alpha': 4.3153506221101616e-05, 'subsample': 0.8594864020811109, 'colsample_bytree': 0.8226871499805907, 'max_depth': 5, 'min_child_weight': 8, 'eta': 0.05426864991813993, 'gamma': 0.771507370360972, 'grow_policy': 'depthwise', 'sample_type': 'weighted', 'normalize_type': 'forest', 'rate_drop': 3.883302281271074e-07, 'skip_drop': 0.7531987105776606}. Best is trial 25 with value: 0.5712751155075699.
    [I 2024-05-31 08:06:51,270] Trial 26 finished with value: 0.5462815084352693 and parameters: {'booster': 'dart', 'lambda': 4.445899193350092e-06, 'alpha': 2.9902577888917067e-05, 'subsample': 0.9897383243981942, 'colsample_bytree': 0.8683348497396252, 'max_depth': 5, 'min_child_weight': 10, 'eta': 0.02976336530256721, 'gamma': 0.9465993293230524, 'grow_policy': 'depthwise', 'sample_type': 'weighted', 'normalize_type': 'forest', 'rate_drop': 9.626780641717336e-07, 'skip_drop': 0.00033574268433227204}. Best is trial 25 with value: 0.5712751155075699.
    [I 2024-05-31 08:07:04,060] Trial 27 finished with value: 0.5728537475331085 and parameters: {'booster': 'dart', 'lambda': 5.5227981639076865e-06, 'alpha': 1.4682496179766172e-06, 'subsample': 0.8638716972606894, 'colsample_bytree': 0.8891667292634532, 'max_depth': 5, 'min_child_weight': 8, 'eta': 0.07846191969081892, 'gamma': 0.0006858547223293182, 'grow_policy': 'depthwise', 'sample_type': 'weighted', 'normalize_type': 'forest', 'rate_drop': 7.990400072965397e-07, 'skip_drop': 0.04390525477108218}. Best is trial 27 with value: 0.5728537475331085.
    [I 2024-05-31 08:07:16,645] Trial 28 finished with value: 0.0 and parameters: {'booster': 'dart', 'lambda': 5.367572438356184e-06, 'alpha': 8.380839740650576e-07, 'subsample': 0.8338465488442812, 'colsample_bytree': 0.8937871413880761, 'max_depth': 5, 'min_child_weight': 9, 'eta': 0.004875200770505362, 'gamma': 0.00026894487893444976, 'grow_policy': 'depthwise', 'sample_type': 'weighted', 'normalize_type': 'forest', 'rate_drop': 2.660305481101783e-05, 'skip_drop': 0.028423979097885552}. Best is trial 27 with value: 0.5728537475331085.
    [I 2024-05-31 08:07:29,373] Trial 29 finished with value: 0.5766577121894174 and parameters: {'booster': 'dart', 'lambda': 3.707601973433028e-05, 'alpha': 0.0029476152533528138, 'subsample': 0.8657846998354596, 'colsample_bytree': 0.7845743266671736, 'max_depth': 5, 'min_child_weight': 8, 'eta': 0.07460466852986879, 'gamma': 0.9701028628621946, 'grow_policy': 'depthwise', 'sample_type': 'uniform', 'normalize_type': 'forest', 'rate_drop': 2.0968514344697428e-05, 'skip_drop': 0.00047370169990841376}. Best is trial 29 with value: 0.5766577121894174.
    [I 2024-05-31 08:07:41,824] Trial 30 finished with value: 0.0 and parameters: {'booster': 'dart', 'lambda': 0.0002608256644056684, 'alpha': 0.002926575552169417, 'subsample': 0.8714296527087463, 'colsample_bytree': 0.7832738244540316, 'max_depth': 5, 'min_child_weight': 8, 'eta': 0.0011651910512941745, 'gamma': 0.8743001928173625, 'grow_policy': 'depthwise', 'sample_type': 'uniform', 'normalize_type': 'forest', 'rate_drop': 1.6853501064983368e-05, 'skip_drop': 0.0001924543843840383}. Best is trial 29 with value: 0.5766577121894174.
    [I 2024-05-31 08:07:54,442] Trial 31 finished with value: 0.2807766275554087 and parameters: {'booster': 'dart', 'lambda': 2.3396545587005205e-05, 'alpha': 8.034993358358182e-05, 'subsample': 0.8585278270566454, 'colsample_bytree': 0.9379173403727156, 'max_depth': 5, 'min_child_weight': 8, 'eta': 0.010717174777513508, 'gamma': 0.02097845558433213, 'grow_policy': 'depthwise', 'sample_type': 'uniform', 'normalize_type': 'forest', 'rate_drop': 1.8899124066196355e-07, 'skip_drop': 0.0018481829497646398}. Best is trial 29 with value: 0.5766577121894174.
    [I 2024-05-31 08:08:07,088] Trial 32 finished with value: 0.5810762910765352 and parameters: {'booster': 'dart', 'lambda': 5.8472775300219754e-05, 'alpha': 0.0015803159404299903, 'subsample': 0.9273534591053617, 'colsample_bytree': 0.781519047910665, 'max_depth': 5, 'min_child_weight': 8, 'eta': 0.11380309868955527, 'gamma': 0.000554499503193926, 'grow_policy': 'depthwise', 'sample_type': 'uniform', 'normalize_type': 'forest', 'rate_drop': 9.401509688418544e-06, 'skip_drop': 0.08992074292572018}. Best is trial 32 with value: 0.5810762910765352.
    [I 2024-05-31 08:08:19,550] Trial 33 finished with value: 0.5764525186509066 and parameters: {'booster': 'dart', 'lambda': 7.507060426480242e-05, 'alpha': 0.003023383224925707, 'subsample': 0.9313926618829003, 'colsample_bytree': 0.8048689289254649, 'max_depth': 5, 'min_child_weight': 8, 'eta': 0.07206457202686005, 'gamma': 0.0003120427421634405, 'grow_policy': 'depthwise', 'sample_type': 'uniform', 'normalize_type': 'forest', 'rate_drop': 8.777858385136778e-06, 'skip_drop': 1.1572588398840922e-05}. Best is trial 32 with value: 0.5810762910765352.
    [I 2024-05-31 08:08:33,371] Trial 34 finished with value: 0.5855192710018992 and parameters: {'booster': 'dart', 'lambda': 8.58958358232474e-05, 'alpha': 0.0016691728246485548, 'subsample': 0.9574842786630192, 'colsample_bytree': 0.7788223653529451, 'max_depth': 7, 'min_child_weight': 10, 'eta': 0.10326864613780598, 'gamma': 0.00034078912841795333, 'grow_policy': 'depthwise', 'sample_type': 'uniform', 'normalize_type': 'forest', 'rate_drop': 0.00032574443229425144, 'skip_drop': 5.703685603945796e-06}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:08:47,049] Trial 35 finished with value: 0.0 and parameters: {'booster': 'dart', 'lambda': 6.406280080185774e-05, 'alpha': 0.001916232208875525, 'subsample': 0.9869287671235492, 'colsample_bytree': 0.7703940240910967, 'max_depth': 7, 'min_child_weight': 10, 'eta': 0.00021130088825132605, 'gamma': 0.00026504369570885827, 'grow_policy': 'depthwise', 'sample_type': 'uniform', 'normalize_type': 'forest', 'rate_drop': 0.0004492756163310885, 'skip_drop': 1.5424303379349966e-05}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:08:51,084] Trial 36 finished with value: 0.5774405867383889 and parameters: {'booster': 'gbtree', 'lambda': 0.009753986355130919, 'alpha': 0.02045871541433503, 'subsample': 0.9297563591242087, 'colsample_bytree': 0.7904652921875349, 'max_depth': 9, 'min_child_weight': 10, 'eta': 0.1132244018916067, 'gamma': 4.93878090753005e-05, 'grow_policy': 'depthwise'}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:08:54,875] Trial 37 finished with value: 0.5833287957261557 and parameters: {'booster': 'gbtree', 'lambda': 0.009278060907773224, 'alpha': 0.7176948938955147, 'subsample': 0.9361578105053538, 'colsample_bytree': 0.7127205476270565, 'max_depth': 9, 'min_child_weight': 10, 'eta': 0.11450564174165644, 'gamma': 9.306576835817249e-05, 'grow_policy': 'depthwise'}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:08:58,469] Trial 38 finished with value: 0.0 and parameters: {'booster': 'gbtree', 'lambda': 0.011546956338148572, 'alpha': 0.6309838014453871, 'subsample': 0.9300530426124494, 'colsample_bytree': 0.6784711385343282, 'max_depth': 9, 'min_child_weight': 10, 'eta': 0.005980555607840083, 'gamma': 0.00012467796612051365, 'grow_policy': 'depthwise'}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:09:02,132] Trial 39 finished with value: 0.5790001763638022 and parameters: {'booster': 'gbtree', 'lambda': 0.8897487173728241, 'alpha': 0.042576601113892935, 'subsample': 0.7877755769703119, 'colsample_bytree': 0.690297454250793, 'max_depth': 9, 'min_child_weight': 10, 'eta': 0.14979823332621683, 'gamma': 4.6196193754786544e-05, 'grow_policy': 'depthwise'}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:09:06,122] Trial 40 finished with value: 0.0 and parameters: {'booster': 'gbtree', 'lambda': 0.42469228013776356, 'alpha': 0.14577712115527727, 'subsample': 0.7713820278076347, 'colsample_bytree': 0.6914263460599905, 'max_depth': 9, 'min_child_weight': 9, 'eta': 2.6152813634525304e-08, 'gamma': 1.1870301552497191e-06, 'grow_policy': 'depthwise'}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:09:10,002] Trial 41 finished with value: 0.5778802068372366 and parameters: {'booster': 'gbtree', 'lambda': 0.9245883551777332, 'alpha': 0.035751785624889884, 'subsample': 0.9347680330222237, 'colsample_bytree': 0.7448205588068578, 'max_depth': 9, 'min_child_weight': 10, 'eta': 0.20317662844867063, 'gamma': 3.846688677048968e-05, 'grow_policy': 'depthwise'}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:09:14,015] Trial 42 finished with value: 0.5709870896371109 and parameters: {'booster': 'gbtree', 'lambda': 0.07386956196651953, 'alpha': 0.039296405430068466, 'subsample': 0.7991631877328442, 'colsample_bytree': 0.7332477034194412, 'max_depth': 9, 'min_child_weight': 10, 'eta': 0.26467023222427866, 'gamma': 5.394550651675749e-05, 'grow_policy': 'depthwise'}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:09:17,491] Trial 43 finished with value: 0.5694109063883404 and parameters: {'booster': 'gbtree', 'lambda': 0.7420543439564475, 'alpha': 0.7083722064775986, 'subsample': 0.9089303129205293, 'colsample_bytree': 0.6606120186823176, 'max_depth': 9, 'min_child_weight': 9, 'eta': 0.21114966538054314, 'gamma': 1.6599869588114238e-06, 'grow_policy': 'depthwise'}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:09:20,456] Trial 44 finished with value: 0.0 and parameters: {'booster': 'gbtree', 'lambda': 0.09147618415292955, 'alpha': 0.010285162421923793, 'subsample': 0.7278391588104786, 'colsample_bytree': 0.5712006667375965, 'max_depth': 7, 'min_child_weight': 10, 'eta': 0.0009296614390879194, 'gamma': 1.0391581468866634e-07, 'grow_policy': 'depthwise'}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:09:24,519] Trial 45 finished with value: 0.4862804945459249 and parameters: {'booster': 'gbtree', 'lambda': 0.02610634452412127, 'alpha': 0.19050167546338018, 'subsample': 0.9683972101868363, 'colsample_bytree': 0.7479157430746683, 'max_depth': 9, 'min_child_weight': 9, 'eta': 0.021413402822578503, 'gamma': 0.0013918468200784274, 'grow_policy': 'depthwise'}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:09:28,198] Trial 46 finished with value: 0.34191963082679655 and parameters: {'booster': 'gbtree', 'lambda': 0.00024929113335654034, 'alpha': 0.3709379553131417, 'subsample': 0.7998878962966594, 'colsample_bytree': 0.9382319406941013, 'max_depth': 7, 'min_child_weight': 10, 'eta': 0.011644644766892814, 'gamma': 3.159949276431383e-05, 'grow_policy': 'lossguide'}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:09:32,303] Trial 47 finished with value: 0.5696053140807176 and parameters: {'booster': 'gbtree', 'lambda': 0.22323560067795617, 'alpha': 0.05021322325327231, 'subsample': 0.9030808253414022, 'colsample_bytree': 0.7137774776712451, 'max_depth': 9, 'min_child_weight': 2, 'eta': 0.16610332535140637, 'gamma': 9.939643741689064e-06, 'grow_policy': 'depthwise'}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:09:35,066] Trial 48 finished with value: 0.0 and parameters: {'booster': 'gbtree', 'lambda': 0.9864469828467832, 'alpha': 0.008746766339442009, 'subsample': 0.9566073238407525, 'colsample_bytree': 0.6154202536630213, 'max_depth': 7, 'min_child_weight': 9, 'eta': 0.0026020279426848783, 'gamma': 8.116144242026686e-05, 'grow_policy': 'lossguide'}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:09:39,081] Trial 49 finished with value: 0.0 and parameters: {'booster': 'gbtree', 'lambda': 0.0007335134436521007, 'alpha': 0.10485239362791741, 'subsample': 0.9999127579867086, 'colsample_bytree': 0.7580840577318595, 'max_depth': 9, 'min_child_weight': 10, 'eta': 5.102067109765045e-06, 'gamma': 3.125047387225855e-06, 'grow_policy': 'depthwise'}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:09:42,683] Trial 50 finished with value: 0.0 and parameters: {'booster': 'gbtree', 'lambda': 0.005358573496205673, 'alpha': 0.0011135372031851618, 'subsample': 0.8135040695422827, 'colsample_bytree': 0.8496835010692358, 'max_depth': 7, 'min_child_weight': 5, 'eta': 7.037593791713061e-07, 'gamma': 1.899108357660033e-05, 'grow_policy': 'lossguide'}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:09:46,720] Trial 51 finished with value: 0.5732536548974592 and parameters: {'booster': 'gbtree', 'lambda': 0.030180586444760904, 'alpha': 0.021308764038768954, 'subsample': 0.9030726625380573, 'colsample_bytree': 0.797646175293991, 'max_depth': 9, 'min_child_weight': 10, 'eta': 0.1608748710712495, 'gamma': 0.00015958429651561043, 'grow_policy': 'depthwise'}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:09:50,439] Trial 52 finished with value: 0.580796286610714 and parameters: {'booster': 'gbtree', 'lambda': 0.00015009839264875792, 'alpha': 0.013617510701322954, 'subsample': 0.9367944573889104, 'colsample_bytree': 0.701503142722607, 'max_depth': 9, 'min_child_weight': 9, 'eta': 0.08791285067526773, 'gamma': 4.902088534908437e-05, 'grow_policy': 'depthwise'}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:09:53,895] Trial 53 finished with value: 0.5517952071702402 and parameters: {'booster': 'gbtree', 'lambda': 0.23030392288271745, 'alpha': 0.006200433059129244, 'subsample': 0.9500936406737173, 'colsample_bytree': 0.6876091476708819, 'max_depth': 9, 'min_child_weight': 9, 'eta': 0.37727867507343, 'gamma': 0.0006848376236038213, 'grow_policy': 'depthwise'}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:09:57,256] Trial 54 finished with value: 0.5390435733527943 and parameters: {'booster': 'gbtree', 'lambda': 0.00014873848207102542, 'alpha': 0.03599334355402416, 'subsample': 0.9594175082971719, 'colsample_bytree': 0.6451047562058168, 'max_depth': 9, 'min_child_weight': 9, 'eta': 0.9592849093681542, 'gamma': 4.252819833850074e-06, 'grow_policy': 'depthwise'}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:10:01,204] Trial 55 finished with value: 0.0 and parameters: {'booster': 'gbtree', 'lambda': 0.001580042352234621, 'alpha': 0.07272504014644621, 'subsample': 0.8926100366846391, 'colsample_bytree': 0.7168018443780622, 'max_depth': 9, 'min_child_weight': 10, 'eta': 4.1335950667341015e-05, 'gamma': 4.763101421785981e-07, 'grow_policy': 'depthwise'}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:10:04,201] Trial 56 finished with value: 0.5597291909466016 and parameters: {'booster': 'gbtree', 'lambda': 0.00023894547029553503, 'alpha': 0.30343500877376317, 'subsample': 0.7615304877097356, 'colsample_bytree': 0.6678985267339603, 'max_depth': 7, 'min_child_weight': 9, 'eta': 0.4067279787202618, 'gamma': 2.001028379502227e-05, 'grow_policy': 'lossguide'}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:10:07,781] Trial 57 finished with value: 0.48561066074097087 and parameters: {'booster': 'gbtree', 'lambda': 0.4497927843189952, 'alpha': 0.0009460458420390054, 'subsample': 0.8873982396178686, 'colsample_bytree': 0.5948597863408951, 'max_depth': 9, 'min_child_weight': 9, 'eta': 0.029612402050046945, 'gamma': 0.0007516264579166709, 'grow_policy': 'depthwise'}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:10:11,866] Trial 58 finished with value: 0.5791290905868993 and parameters: {'booster': 'gbtree', 'lambda': 0.00041250121086445785, 'alpha': 0.005836557360571742, 'subsample': 0.8321848247233952, 'colsample_bytree': 0.7331874141335385, 'max_depth': 9, 'min_child_weight': 10, 'eta': 0.1126741525588472, 'gamma': 0.00011911444689963084, 'grow_policy': 'depthwise'}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:10:14,862] Trial 59 finished with value: 0.2038944204186275 and parameters: {'booster': 'gbtree', 'lambda': 0.0005099544285390137, 'alpha': 0.004146648266640014, 'subsample': 0.8287061722210431, 'colsample_bytree': 0.6287166990122064, 'max_depth': 7, 'min_child_weight': 9, 'eta': 0.009000685243450737, 'gamma': 0.0035609494994677136, 'grow_policy': 'depthwise'}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:10:19,205] Trial 60 finished with value: 0.5763335619648973 and parameters: {'booster': 'gbtree', 'lambda': 1.409553549223489e-05, 'alpha': 0.010509405461722267, 'subsample': 0.7254688831225662, 'colsample_bytree': 0.8229296526915684, 'max_depth': 9, 'min_child_weight': 10, 'eta': 0.04978085376340828, 'gamma': 0.00014170398831776535, 'grow_policy': 'depthwise'}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:10:21,726] Trial 61 finished with value: 0.5384076683940382 and parameters: {'booster': 'gbtree', 'lambda': 0.0001174651085949018, 'alpha': 0.014267231707700593, 'subsample': 0.9330697247859048, 'colsample_bytree': 0.2268762049626104, 'max_depth': 9, 'min_child_weight': 10, 'eta': 0.10971468140002927, 'gamma': 3.9751670114344046e-05, 'grow_policy': 'depthwise'}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:10:25,658] Trial 62 finished with value: 0.5500619849925311 and parameters: {'booster': 'gbtree', 'lambda': 3.724179618251001e-05, 'alpha': 0.0005249282561510509, 'subsample': 0.8348782211123699, 'colsample_bytree': 0.7307920263690608, 'max_depth': 9, 'min_child_weight': 10, 'eta': 0.4987195912160405, 'gamma': 0.0003949649352509999, 'grow_policy': 'depthwise'}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:10:29,768] Trial 63 finished with value: 0.5084609905914507 and parameters: {'booster': 'gbtree', 'lambda': 0.0015353172796812673, 'alpha': 0.005618104680076903, 'subsample': 0.9683571395257912, 'colsample_bytree': 0.7625768244938637, 'max_depth': 9, 'min_child_weight': 10, 'eta': 0.024134078679740763, 'gamma': 8.555385265295788e-05, 'grow_policy': 'depthwise'}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:10:33,550] Trial 64 finished with value: 0.5748945746328498 and parameters: {'booster': 'gbtree', 'lambda': 0.12794086936971427, 'alpha': 0.00019596521362711197, 'subsample': 0.9146869385878944, 'colsample_bytree': 0.7031291333334896, 'max_depth': 9, 'min_child_weight': 9, 'eta': 0.15367778977443203, 'gamma': 6.761088457057803e-06, 'grow_policy': 'depthwise'}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:10:34,330] Trial 65 finished with value: 0.0 and parameters: {'booster': 'gblinear', 'lambda': 0.00043792216659129, 'alpha': 0.026946504860182414, 'subsample': 0.780499194083877, 'colsample_bytree': 0.741227170494573}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:10:38,854] Trial 66 finished with value: 0.5800666726861253 and parameters: {'booster': 'gbtree', 'lambda': 0.04267835277731476, 'alpha': 0.0017068826488788216, 'subsample': 0.847307486503111, 'colsample_bytree': 0.855064431993026, 'max_depth': 9, 'min_child_weight': 4, 'eta': 0.05219749061987179, 'gamma': 0.0020815633354413525, 'grow_policy': 'depthwise'}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:10:54,697] Trial 67 finished with value: 0.5775145268197396 and parameters: {'booster': 'dart', 'lambda': 0.05240091093163528, 'alpha': 0.0019886966566343054, 'subsample': 0.8449495599706212, 'colsample_bytree': 0.8506515366874436, 'max_depth': 9, 'min_child_weight': 3, 'eta': 0.04424990029947856, 'gamma': 0.0038175746247411948, 'grow_policy': 'depthwise', 'sample_type': 'uniform', 'normalize_type': 'tree', 'rate_drop': 0.0019654294566272914, 'skip_drop': 1.0474259829872149e-08}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:11:08,694] Trial 68 finished with value: 0.44859309285630056 and parameters: {'booster': 'dart', 'lambda': 0.0027027935283791917, 'alpha': 0.0017317721270370394, 'subsample': 0.8762625336044966, 'colsample_bytree': 0.8891130455851539, 'max_depth': 7, 'min_child_weight': 4, 'eta': 0.01638093685786647, 'gamma': 0.002024265182040802, 'grow_policy': 'depthwise', 'sample_type': 'uniform', 'normalize_type': 'tree', 'rate_drop': 0.00023157832281599371, 'skip_drop': 5.996484558487764e-08}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:11:13,651] Trial 69 finished with value: 0.0 and parameters: {'booster': 'gbtree', 'lambda': 0.01201884615980719, 'alpha': 1.2915129114638217e-05, 'subsample': 0.7336535895881429, 'colsample_bytree': 0.9184710928871676, 'max_depth': 9, 'min_child_weight': 5, 'eta': 0.0037363823065861422, 'gamma': 0.0005105625461487432, 'grow_policy': 'depthwise'}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:11:14,937] Trial 70 finished with value: 0.25508609533598825 and parameters: {'booster': 'gblinear', 'lambda': 0.004777747057956787, 'alpha': 0.00019126548105398352, 'subsample': 0.9834849696572574, 'colsample_bytree': 0.8161003432904717}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:11:19,065] Trial 71 finished with value: 0.5794293346900881 and parameters: {'booster': 'gbtree', 'lambda': 0.420221202701566, 'alpha': 0.09047861691373528, 'subsample': 0.9455430140978787, 'colsample_bytree': 0.7726982011516041, 'max_depth': 9, 'min_child_weight': 6, 'eta': 0.08530335686451763, 'gamma': 0.009708489149924737, 'grow_policy': 'depthwise'}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:11:23,232] Trial 72 finished with value: 0.5808486317139534 and parameters: {'booster': 'gbtree', 'lambda': 0.3944790407420316, 'alpha': 0.9391954440694456, 'subsample': 0.8820296940106862, 'colsample_bytree': 0.7733657327856404, 'max_depth': 9, 'min_child_weight': 6, 'eta': 0.0753191201482413, 'gamma': 0.008009535786820243, 'grow_policy': 'depthwise'}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:11:27,404] Trial 73 finished with value: 0.5816659612918603 and parameters: {'booster': 'gbtree', 'lambda': 0.05008504850886715, 'alpha': 0.541053042028111, 'subsample': 0.8868504844129026, 'colsample_bytree': 0.7754639979670619, 'max_depth': 9, 'min_child_weight': 6, 'eta': 0.07444902385702883, 'gamma': 0.00832688514089381, 'grow_policy': 'depthwise'}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:11:31,611] Trial 74 finished with value: 0.5811721872691985 and parameters: {'booster': 'gbtree', 'lambda': 0.03287253441803756, 'alpha': 0.8707123534265764, 'subsample': 0.9476391965808999, 'colsample_bytree': 0.8451409252120169, 'max_depth': 9, 'min_child_weight': 6, 'eta': 0.05393591045098148, 'gamma': 0.01211902708196573, 'grow_policy': 'depthwise'}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:11:46,823] Trial 75 finished with value: 0.5594740487442305 and parameters: {'booster': 'dart', 'lambda': 0.04156508335000771, 'alpha': 0.8965862354695056, 'subsample': 0.9128312218005807, 'colsample_bytree': 0.8693916675878822, 'max_depth': 9, 'min_child_weight': 6, 'eta': 0.04240147797117713, 'gamma': 0.015438737409741433, 'grow_policy': 'depthwise', 'sample_type': 'uniform', 'normalize_type': 'tree', 'rate_drop': 0.024804997387126902, 'skip_drop': 1.6625559274401714e-05}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:11:49,386] Trial 76 finished with value: 0.5646286023255016 and parameters: {'booster': 'gbtree', 'lambda': 0.12059342255063538, 'alpha': 0.3765034463864593, 'subsample': 0.8826915936828004, 'colsample_bytree': 0.8327350705547848, 'max_depth': 5, 'min_child_weight': 5, 'eta': 0.41700118436491446, 'gamma': 0.039057156010884324, 'grow_policy': 'depthwise'}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:12:04,450] Trial 77 finished with value: 0.5776159480534891 and parameters: {'booster': 'dart', 'lambda': 0.019196238704427958, 'alpha': 0.9661458004203932, 'subsample': 0.8581626359747915, 'colsample_bytree': 0.8025383404826981, 'max_depth': 9, 'min_child_weight': 6, 'eta': 0.06945783015687422, 'gamma': 0.11714302320111071, 'grow_policy': 'depthwise', 'sample_type': 'uniform', 'normalize_type': 'tree', 'rate_drop': 0.0014569653046357471, 'skip_drop': 6.954115933728952e-07}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:12:07,876] Trial 78 finished with value: 0.36046201512932574 and parameters: {'booster': 'gbtree', 'lambda': 0.018854059771491422, 'alpha': 0.20443587907343774, 'subsample': 0.9758986740368919, 'colsample_bytree': 0.8742370184584539, 'max_depth': 7, 'min_child_weight': 7, 'eta': 0.01261296918801609, 'gamma': 0.0026581838384733695, 'grow_policy': 'depthwise'}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:12:12,515] Trial 79 finished with value: 0.10501015569085133 and parameters: {'booster': 'gbtree', 'lambda': 0.00661462407236595, 'alpha': 0.4882167529301984, 'subsample': 0.5946930014694932, 'colsample_bytree': 0.9112164590764047, 'max_depth': 9, 'min_child_weight': 5, 'eta': 0.006906163847814064, 'gamma': 0.006530253342767016, 'grow_policy': 'lossguide'}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:12:24,033] Trial 80 finished with value: 0.4921810112461519 and parameters: {'booster': 'dart', 'lambda': 2.8992796323012034e-06, 'alpha': 0.20559351401854178, 'subsample': 0.9153080436391772, 'colsample_bytree': 0.9657400124399919, 'max_depth': 3, 'min_child_weight': 7, 'eta': 0.03426524697977198, 'gamma': 0.0013173322490855348, 'grow_policy': 'depthwise', 'sample_type': 'uniform', 'normalize_type': 'forest', 'rate_drop': 0.012520166346659091, 'skip_drop': 0.002533855394562695}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:12:28,094] Trial 81 finished with value: 0.5773128309485983 and parameters: {'booster': 'gbtree', 'lambda': 0.1748439273951054, 'alpha': 0.10105848844267169, 'subsample': 0.942082326907809, 'colsample_bytree': 0.7676132784470981, 'max_depth': 9, 'min_child_weight': 6, 'eta': 0.07863227739290286, 'gamma': 0.011573885084936997, 'grow_policy': 'depthwise'}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:12:32,074] Trial 82 finished with value: 0.5709722576454193 and parameters: {'booster': 'gbtree', 'lambda': 0.39954958283469316, 'alpha': 0.5365052950226461, 'subsample': 0.9476502862026183, 'colsample_bytree': 0.7802408302744066, 'max_depth': 9, 'min_child_weight': 6, 'eta': 0.2560377414409437, 'gamma': 0.00619373182078786, 'grow_policy': 'depthwise'}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:12:36,243] Trial 83 finished with value: 0.5819124748906328 and parameters: {'booster': 'gbtree', 'lambda': 0.46547977624877884, 'alpha': 0.06925694943327615, 'subsample': 0.9940830330854037, 'colsample_bytree': 0.8408575093664672, 'max_depth': 9, 'min_child_weight': 6, 'eta': 0.07190313761316965, 'gamma': 0.026686001409661344, 'grow_policy': 'depthwise'}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:12:40,247] Trial 84 finished with value: 0.542296332264806 and parameters: {'booster': 'gbtree', 'lambda': 0.04943714627499994, 'alpha': 0.1567379341572561, 'subsample': 0.999009046713887, 'colsample_bytree': 0.8417021176120432, 'max_depth': 9, 'min_child_weight': 4, 'eta': 0.6519631029278637, 'gamma': 0.22604256741855377, 'grow_policy': 'depthwise'}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:12:42,744] Trial 85 finished with value: 0.467035683574046 and parameters: {'booster': 'gbtree', 'lambda': 8.105372114519492e-05, 'alpha': 0.45822992667481266, 'subsample': 0.9755785488956709, 'colsample_bytree': 0.8052024640849927, 'max_depth': 5, 'min_child_weight': 5, 'eta': 0.021471528368812803, 'gamma': 0.024657869133260502, 'grow_policy': 'depthwise'}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:12:47,125] Trial 86 finished with value: 0.5817776575258988 and parameters: {'booster': 'gbtree', 'lambda': 0.08604031082044279, 'alpha': 0.2648495183173291, 'subsample': 0.8950429983316046, 'colsample_bytree': 0.86107434032977, 'max_depth': 9, 'min_child_weight': 6, 'eta': 0.05945499479244395, 'gamma': 0.03811187439179955, 'grow_policy': 'depthwise'}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:12:47,842] Trial 87 finished with value: 0.0 and parameters: {'booster': 'gblinear', 'lambda': 0.08297661809010212, 'alpha': 0.26452878064143726, 'subsample': 0.8910673020349347, 'colsample_bytree': 0.9430569099407703}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:13:01,797] Trial 88 finished with value: 0.5634528371186406 and parameters: {'booster': 'dart', 'lambda': 0.2830197217555597, 'alpha': 0.06915068932501868, 'subsample': 0.9239486632768541, 'colsample_bytree': 0.3283276891837372, 'max_depth': 9, 'min_child_weight': 6, 'eta': 0.10965418041661161, 'gamma': 0.04350940843580524, 'grow_policy': 'depthwise', 'sample_type': 'uniform', 'normalize_type': 'forest', 'rate_drop': 7.232462859051271e-05, 'skip_drop': 6.27738039675137e-05}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:13:05,865] Trial 89 finished with value: 0.5661679788109371 and parameters: {'booster': 'gbtree', 'lambda': 2.250727492709542e-05, 'alpha': 0.5962294242568487, 'subsample': 0.9634985241544763, 'colsample_bytree': 0.8924678516808169, 'max_depth': 9, 'min_child_weight': 7, 'eta': 0.3302263361126628, 'gamma': 0.023894623501737308, 'grow_policy': 'depthwise'}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:13:09,848] Trial 90 finished with value: 0.0 and parameters: {'booster': 'gbtree', 'lambda': 9.648924957047644e-06, 'alpha': 0.947961400115871, 'subsample': 0.438886880705387, 'colsample_bytree': 0.7895401846030156, 'max_depth': 9, 'min_child_weight': 6, 'eta': 0.0002484493866991435, 'gamma': 0.10396299872949007, 'grow_policy': 'depthwise'}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:13:14,086] Trial 91 finished with value: 0.5816807196184046 and parameters: {'booster': 'gbtree', 'lambda': 0.038289248280460084, 'alpha': 0.1258473978021484, 'subsample': 0.8984993878272722, 'colsample_bytree': 0.866001005151414, 'max_depth': 9, 'min_child_weight': 8, 'eta': 0.05925883571795305, 'gamma': 0.005532204278549119, 'grow_policy': 'depthwise'}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:13:18,345] Trial 92 finished with value: 0.5621203819443986 and parameters: {'booster': 'gbtree', 'lambda': 0.16374796854974205, 'alpha': 0.28204937922160417, 'subsample': 0.5115643321312787, 'colsample_bytree': 0.8352036640982768, 'max_depth': 9, 'min_child_weight': 8, 'eta': 0.22971595195317995, 'gamma': 0.0676412869122269, 'grow_policy': 'depthwise'}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:13:22,715] Trial 93 finished with value: 0.562674629468721 and parameters: {'booster': 'gbtree', 'lambda': 0.06946149840305443, 'alpha': 0.1383145919747024, 'subsample': 0.901273598785709, 'colsample_bytree': 0.8176129566892948, 'max_depth': 9, 'min_child_weight': 7, 'eta': 0.037095317382683346, 'gamma': 0.03019916762933858, 'grow_policy': 'depthwise'}. Best is trial 34 with value: 0.5855192710018992.
    [I 2024-05-31 08:13:27,036] Trial 94 finished with value: 0.5868775523872289 and parameters: {'booster': 'gbtree', 'lambda': 0.025964667955578884, 'alpha': 0.06244096421739463, 'subsample': 0.872950277861732, 'colsample_bytree': 0.8593913565656731, 'max_depth': 9, 'min_child_weight': 6, 'eta': 0.06164293016905939, 'gamma': 0.015529967527323904, 'grow_policy': 'depthwise'}. Best is trial 94 with value: 0.5868775523872289.
    [I 2024-05-31 08:13:31,594] Trial 95 finished with value: 0.4618685327154175 and parameters: {'booster': 'gbtree', 'lambda': 0.0287979151307092, 'alpha': 0.05498740026440463, 'subsample': 0.8134602807395863, 'colsample_bytree': 0.8681114886042203, 'max_depth': 9, 'min_child_weight': 6, 'eta': 0.016403261051904004, 'gamma': 0.007129055139595104, 'grow_policy': 'lossguide'}. Best is trial 94 with value: 0.5868775523872289.
    [I 2024-05-31 08:13:34,288] Trial 96 finished with value: 0.577335923632021 and parameters: {'booster': 'gbtree', 'lambda': 0.019563545361481242, 'alpha': 0.3580811601457349, 'subsample': 0.8776822606459987, 'colsample_bytree': 0.9105466745095658, 'max_depth': 5, 'min_child_weight': 7, 'eta': 0.05896949256512852, 'gamma': 0.017871401090512083, 'grow_policy': 'depthwise'}. Best is trial 94 with value: 0.5868775523872289.
    [I 2024-05-31 08:13:49,761] Trial 97 finished with value: 0.5825096491193038 and parameters: {'booster': 'dart', 'lambda': 0.1125881794010637, 'alpha': 0.6510230734975653, 'subsample': 0.9559207370303667, 'colsample_bytree': 0.8601897002087727, 'max_depth': 9, 'min_child_weight': 6, 'eta': 0.1229934417637614, 'gamma': 0.0010976931094366187, 'grow_policy': 'depthwise', 'sample_type': 'uniform', 'normalize_type': 'forest', 'rate_drop': 3.315028330839972e-06, 'skip_drop': 2.9674940867573198e-06}. Best is trial 94 with value: 0.5868775523872289.
    [I 2024-05-31 08:14:05,001] Trial 98 finished with value: 0.574500159519603 and parameters: {'booster': 'dart', 'lambda': 0.008771349212485091, 'alpha': 0.2285564106504612, 'subsample': 0.9975321502984437, 'colsample_bytree': 0.8764332383816283, 'max_depth': 9, 'min_child_weight': 5, 'eta': 0.15475411741206802, 'gamma': 0.00023309706509351902, 'grow_policy': 'depthwise', 'sample_type': 'uniform', 'normalize_type': 'forest', 'rate_drop': 3.5728686286431397e-06, 'skip_drop': 2.1346719473068894e-06}. Best is trial 94 with value: 0.5868775523872289.
    [I 2024-05-31 08:14:19,175] Trial 99 finished with value: 0.5560716610241367 and parameters: {'booster': 'dart', 'lambda': 0.014411700076433295, 'alpha': 0.12623842372658825, 'subsample': 0.9531132108259482, 'colsample_bytree': 0.9708920787806169, 'max_depth': 7, 'min_child_weight': 6, 'eta': 0.02743384745305306, 'gamma': 0.24749310782322978, 'grow_policy': 'depthwise', 'sample_type': 'uniform', 'normalize_type': 'forest', 'rate_drop': 9.971229982154957e-05, 'skip_drop': 1.9931577278772751e-07}. Best is trial 94 with value: 0.5868775523872289.
    

**CatBoost**


```python
def tune_catboost(trial: optuna.Trial) -> float:
    param = {
        "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1, log=True),
        "depth": trial.suggest_int("depth", 1, 12),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        ),
        "used_ram_limit": "3gb",
        "eval_metric": "Accuracy",
    }

    if param["bootstrap_type"] == "Bayesian":
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif param["bootstrap_type"] == "Bernoulli":
        param["subsample"] = trial.suggest_float("subsample", 0.1, 1, log=True)

    gbm = CatBoostClassifier(**param, logging_level='Silent')

    kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=21)
    score = cross_val_score(gbm, X_train_prepared, y_train, scoring='f1', cv=kfold).mean()
    
    return score
```


```python
sampler = TPESampler(seed=21)
study_catboost = optuna.create_study(direction='maximize', sampler = sampler)
study_catboost.optimize(tune_catboost, n_trials = 50)
```

    [I 2024-05-31 08:14:19,199] A new study created in memory with name: no-name-170bc8c2-434d-448b-82b5-9921293e5611
    [I 2024-05-31 08:15:45,548] Trial 0 finished with value: 0.36954010302386553 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.05259765072681896, 'depth': 1, 'boosting_type': 'Ordered', 'bootstrap_type': 'Bernoulli', 'subsample': 0.3833463003091129}. Best is trial 0 with value: 0.36954010302386553.
    [I 2024-05-31 08:16:08,469] Trial 1 finished with value: 0.31813709923474 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.013590659107103026, 'depth': 3, 'boosting_type': 'Plain', 'bootstrap_type': 'Bernoulli', 'subsample': 0.24224232832813447}. Best is trial 0 with value: 0.36954010302386553.
    [I 2024-05-31 08:17:37,539] Trial 2 finished with value: 0.36447007867055603 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.018649610789480263, 'depth': 11, 'boosting_type': 'Ordered', 'bootstrap_type': 'Bayesian', 'bagging_temperature': 2.839430450306338}. Best is trial 0 with value: 0.36954010302386553.
    [I 2024-05-31 08:19:05,913] Trial 3 finished with value: 0.4220319482603242 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.03502421090809937, 'depth': 3, 'boosting_type': 'Ordered', 'bootstrap_type': 'Bayesian', 'bagging_temperature': 6.527560731939891}. Best is trial 3 with value: 0.4220319482603242.
    [I 2024-05-31 08:20:44,011] Trial 4 finished with value: 0.5659983141734967 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.06458884962281519, 'depth': 8, 'boosting_type': 'Ordered', 'bootstrap_type': 'Bernoulli', 'subsample': 0.7720677949150249}. Best is trial 4 with value: 0.5659983141734967.
    [I 2024-05-31 08:21:08,064] Trial 5 finished with value: 0.4160677940907428 and parameters: {'objective': 'Logloss', 'colsample_bylevel': 0.020226240401145887, 'depth': 10, 'boosting_type': 'Plain', 'bootstrap_type': 'MVS'}. Best is trial 4 with value: 0.5659983141734967.
    [I 2024-05-31 08:21:37,107] Trial 6 finished with value: 0.5646063621191597 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.06272675765855494, 'depth': 8, 'boosting_type': 'Plain', 'bootstrap_type': 'Bernoulli', 'subsample': 0.7312834579957102}. Best is trial 4 with value: 0.5659983141734967.
    [I 2024-05-31 08:22:00,561] Trial 7 finished with value: 0.364072107904838 and parameters: {'objective': 'Logloss', 'colsample_bylevel': 0.015109374061559683, 'depth': 11, 'boosting_type': 'Plain', 'bootstrap_type': 'MVS'}. Best is trial 4 with value: 0.5659983141734967.
    [I 2024-05-31 08:22:25,091] Trial 8 finished with value: 0.5186264769047999 and parameters: {'objective': 'Logloss', 'colsample_bylevel': 0.07200163532837518, 'depth': 3, 'boosting_type': 'Plain', 'bootstrap_type': 'Bayesian', 'bagging_temperature': 5.232026990109716}. Best is trial 4 with value: 0.5659983141734967.
    [I 2024-05-31 08:22:48,206] Trial 9 finished with value: 0.358826634176137 and parameters: {'objective': 'Logloss', 'colsample_bylevel': 0.016154854505208056, 'depth': 10, 'boosting_type': 'Plain', 'bootstrap_type': 'Bayesian', 'bagging_temperature': 0.3510298901951303}. Best is trial 4 with value: 0.5659983141734967.
    [I 2024-05-31 08:24:25,126] Trial 10 finished with value: 0.5758370662604548 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.08891372676631633, 'depth': 7, 'boosting_type': 'Ordered', 'bootstrap_type': 'Bernoulli', 'subsample': 0.9486163873604913}. Best is trial 10 with value: 0.5758370662604548.
    [I 2024-05-31 08:26:02,663] Trial 11 finished with value: 0.5761730298609979 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.09927156048867768, 'depth': 7, 'boosting_type': 'Ordered', 'bootstrap_type': 'Bernoulli', 'subsample': 0.9703270897747115}. Best is trial 11 with value: 0.5761730298609979.
    [I 2024-05-31 08:27:37,502] Trial 12 finished with value: 0.5671112255181564 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.09746805745471916, 'depth': 6, 'boosting_type': 'Ordered', 'bootstrap_type': 'Bernoulli', 'subsample': 0.9573406860796835}. Best is trial 11 with value: 0.5761730298609979.
    [I 2024-05-31 08:29:11,465] Trial 13 finished with value: 0.5580385542262487 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.09858080540396018, 'depth': 6, 'boosting_type': 'Ordered', 'bootstrap_type': 'Bernoulli', 'subsample': 0.10662280178065826}. Best is trial 11 with value: 0.5761730298609979.
    [I 2024-05-31 08:30:44,105] Trial 14 finished with value: 0.5225680723679348 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.036936359945123946, 'depth': 8, 'boosting_type': 'Ordered', 'bootstrap_type': 'Bernoulli', 'subsample': 0.48294935567877745}. Best is trial 11 with value: 0.5761730298609979.
    [I 2024-05-31 08:32:10,438] Trial 15 finished with value: 0.26615218495934956 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.010159255026991948, 'depth': 5, 'boosting_type': 'Ordered', 'bootstrap_type': 'Bernoulli', 'subsample': 0.5462958725900512}. Best is trial 11 with value: 0.5761730298609979.
    [I 2024-05-31 08:33:41,912] Trial 16 finished with value: 0.5314501167035729 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.04693112290972879, 'depth': 5, 'boosting_type': 'Ordered', 'bootstrap_type': 'MVS'}. Best is trial 11 with value: 0.5761730298609979.
    [I 2024-05-31 08:35:12,539] Trial 17 finished with value: 0.4447853599728082 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.0272147328225744, 'depth': 9, 'boosting_type': 'Ordered', 'bootstrap_type': 'Bernoulli', 'subsample': 0.2076353018086354}. Best is trial 11 with value: 0.5761730298609979.
    [I 2024-05-31 08:36:49,168] Trial 18 finished with value: 0.5699435456756863 and parameters: {'objective': 'Logloss', 'colsample_bylevel': 0.07809693016499103, 'depth': 7, 'boosting_type': 'Ordered', 'bootstrap_type': 'Bernoulli', 'subsample': 0.932614743849422}. Best is trial 11 with value: 0.5761730298609979.
    [I 2024-05-31 08:39:39,434] Trial 19 finished with value: 0.569340663028145 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.05043643507616937, 'depth': 12, 'boosting_type': 'Ordered', 'bootstrap_type': 'MVS'}. Best is trial 11 with value: 0.5761730298609979.
    [I 2024-05-31 08:41:12,062] Trial 20 finished with value: 0.5639345420760393 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.08289954529255483, 'depth': 5, 'boosting_type': 'Ordered', 'bootstrap_type': 'Bernoulli', 'subsample': 0.585796743608267}. Best is trial 11 with value: 0.5761730298609979.
    [I 2024-05-31 08:42:48,881] Trial 21 finished with value: 0.5705186792705381 and parameters: {'objective': 'Logloss', 'colsample_bylevel': 0.0835493094815589, 'depth': 7, 'boosting_type': 'Ordered', 'bootstrap_type': 'Bernoulli', 'subsample': 0.958041885096952}. Best is trial 11 with value: 0.5761730298609979.
    [I 2024-05-31 08:44:26,085] Trial 22 finished with value: 0.5725061064206203 and parameters: {'objective': 'Logloss', 'colsample_bylevel': 0.09797466341886736, 'depth': 7, 'boosting_type': 'Ordered', 'bootstrap_type': 'Bernoulli', 'subsample': 0.9580368484924835}. Best is trial 11 with value: 0.5761730298609979.
    [I 2024-05-31 08:46:03,242] Trial 23 finished with value: 0.5755148203876583 and parameters: {'objective': 'Logloss', 'colsample_bylevel': 0.09585382755661644, 'depth': 7, 'boosting_type': 'Ordered', 'bootstrap_type': 'Bernoulli', 'subsample': 0.6842144700123579}. Best is trial 11 with value: 0.5761730298609979.
    [I 2024-05-31 08:47:44,879] Trial 24 finished with value: 0.5692921532302726 and parameters: {'objective': 'Logloss', 'colsample_bylevel': 0.0607431557933592, 'depth': 9, 'boosting_type': 'Ordered', 'bootstrap_type': 'Bernoulli', 'subsample': 0.6405596877175006}. Best is trial 11 with value: 0.5761730298609979.
    [I 2024-05-31 08:49:14,189] Trial 25 finished with value: 0.5160296342981726 and parameters: {'objective': 'Logloss', 'colsample_bylevel': 0.045049171868864084, 'depth': 4, 'boosting_type': 'Ordered', 'bootstrap_type': 'Bernoulli', 'subsample': 0.45057076927320316}. Best is trial 11 with value: 0.5761730298609979.
    [I 2024-05-31 08:50:42,520] Trial 26 finished with value: 0.46015058725356406 and parameters: {'objective': 'Logloss', 'colsample_bylevel': 0.02624112707891976, 'depth': 6, 'boosting_type': 'Ordered', 'bootstrap_type': 'Bernoulli', 'subsample': 0.7197355178465242}. Best is trial 11 with value: 0.5761730298609979.
    [I 2024-05-31 08:52:28,296] Trial 27 finished with value: 0.5712426718406329 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.07765131136854216, 'depth': 9, 'boosting_type': 'Ordered', 'bootstrap_type': 'Bernoulli', 'subsample': 0.3267523705825551}. Best is trial 11 with value: 0.5761730298609979.
    [I 2024-05-31 08:54:05,267] Trial 28 finished with value: 0.5675798981482887 and parameters: {'objective': 'Logloss', 'colsample_bylevel': 0.05692081979907183, 'depth': 8, 'boosting_type': 'Ordered', 'bootstrap_type': 'MVS'}. Best is trial 11 with value: 0.5761730298609979.
    [I 2024-05-31 08:55:36,101] Trial 29 finished with value: 0.5122281477919584 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.07112923266114, 'depth': 4, 'boosting_type': 'Ordered', 'bootstrap_type': 'Bayesian', 'bagging_temperature': 9.6165248638938}. Best is trial 11 with value: 0.5761730298609979.
    [I 2024-05-31 08:57:12,569] Trial 30 finished with value: 0.5728139055570385 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.08838288460999526, 'depth': 7, 'boosting_type': 'Ordered', 'bootstrap_type': 'Bernoulli', 'subsample': 0.7599182039996459}. Best is trial 11 with value: 0.5761730298609979.
    [I 2024-05-31 08:58:38,774] Trial 31 finished with value: 0.4181722902290077 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.088759526700124, 'depth': 1, 'boosting_type': 'Ordered', 'bootstrap_type': 'Bernoulli', 'subsample': 0.7799478751579374}. Best is trial 11 with value: 0.5761730298609979.
    [I 2024-05-31 09:00:15,026] Trial 32 finished with value: 0.5697708590194845 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.08636707710775754, 'depth': 7, 'boosting_type': 'Ordered', 'bootstrap_type': 'Bernoulli', 'subsample': 0.7895093850345182}. Best is trial 11 with value: 0.5761730298609979.
    [I 2024-05-31 09:01:48,850] Trial 33 finished with value: 0.5673558392452402 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.07327967703963884, 'depth': 6, 'boosting_type': 'Ordered', 'bootstrap_type': 'Bernoulli', 'subsample': 0.6161694119924297}. Best is trial 11 with value: 0.5761730298609979.
    [I 2024-05-31 09:03:27,862] Trial 34 finished with value: 0.5716581550922543 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.06750572371266302, 'depth': 8, 'boosting_type': 'Ordered', 'bootstrap_type': 'Bernoulli', 'subsample': 0.985852458202778}. Best is trial 11 with value: 0.5761730298609979.
    [I 2024-05-31 09:05:02,017] Trial 35 finished with value: 0.5612443089261101 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.05343273861944594, 'depth': 7, 'boosting_type': 'Ordered', 'bootstrap_type': 'Bernoulli', 'subsample': 0.7981744285908245}. Best is trial 11 with value: 0.5761730298609979.
    [I 2024-05-31 09:05:28,365] Trial 36 finished with value: 0.5504222084653311 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.09033548792865853, 'depth': 5, 'boosting_type': 'Plain', 'bootstrap_type': 'Bayesian', 'bagging_temperature': 9.51182614593699}. Best is trial 11 with value: 0.5761730298609979.
    [I 2024-05-31 09:07:03,518] Trial 37 finished with value: 0.5353945625858201 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.04024605330216716, 'depth': 9, 'boosting_type': 'Ordered', 'bootstrap_type': 'Bernoulli', 'subsample': 0.505647275996746}. Best is trial 11 with value: 0.5761730298609979.
    [I 2024-05-31 09:07:32,543] Trial 38 finished with value: 0.5699574716341704 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.06481145703955883, 'depth': 8, 'boosting_type': 'Plain', 'bootstrap_type': 'Bernoulli', 'subsample': 0.6607657853649218}. Best is trial 11 with value: 0.5761730298609979.
    [I 2024-05-31 09:09:27,342] Trial 39 finished with value: 0.5722447686542746 and parameters: {'objective': 'Logloss', 'colsample_bylevel': 0.058092669510875675, 'depth': 10, 'boosting_type': 'Ordered', 'bootstrap_type': 'MVS'}. Best is trial 11 with value: 0.5761730298609979.
    [I 2024-05-31 09:09:54,526] Trial 40 finished with value: 0.5729520857502741 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.0987479981513065, 'depth': 6, 'boosting_type': 'Plain', 'bootstrap_type': 'Bayesian', 'bagging_temperature': 0.3893439883451961}. Best is trial 11 with value: 0.5761730298609979.
    [I 2024-05-31 09:10:21,719] Trial 41 finished with value: 0.5725110667428537 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.09758968453661337, 'depth': 6, 'boosting_type': 'Plain', 'bootstrap_type': 'Bayesian', 'bagging_temperature': 0.3422325632270944}. Best is trial 11 with value: 0.5761730298609979.
    [I 2024-05-31 09:10:49,991] Trial 42 finished with value: 0.562589681490696 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.08227435811924096, 'depth': 7, 'boosting_type': 'Plain', 'bootstrap_type': 'Bayesian', 'bagging_temperature': 2.6374635089661638}. Best is trial 11 with value: 0.5761730298609979.
    [I 2024-05-31 09:11:17,191] Trial 43 finished with value: 0.5633344155161131 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.09963399735094648, 'depth': 6, 'boosting_type': 'Plain', 'bootstrap_type': 'Bayesian', 'bagging_temperature': 2.7453270000512315}. Best is trial 11 with value: 0.5761730298609979.
    [I 2024-05-31 09:11:47,041] Trial 44 finished with value: 0.5490443896424169 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.07478069286908477, 'depth': 8, 'boosting_type': 'Plain', 'bootstrap_type': 'Bayesian', 'bagging_temperature': 7.203289124292185}. Best is trial 11 with value: 0.5761730298609979.
    [I 2024-05-31 09:12:13,196] Trial 45 finished with value: 0.5612313328218015 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.08796426654372912, 'depth': 5, 'boosting_type': 'Plain', 'bootstrap_type': 'Bayesian', 'bagging_temperature': 1.848510663061687}. Best is trial 11 with value: 0.5761730298609979.
    [I 2024-05-31 09:12:38,187] Trial 46 finished with value: 0.5467828887488623 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.0903307241109837, 'depth': 4, 'boosting_type': 'Plain', 'bootstrap_type': 'Bernoulli', 'subsample': 0.15424042783847386}. Best is trial 11 with value: 0.5761730298609979.
    [I 2024-05-31 09:14:12,581] Trial 47 finished with value: 0.5704654090506154 and parameters: {'objective': 'Logloss', 'colsample_bylevel': 0.06672122095563666, 'depth': 7, 'boosting_type': 'Ordered', 'bootstrap_type': 'Bernoulli', 'subsample': 0.4230286611877532}. Best is trial 11 with value: 0.5761730298609979.
    [I 2024-05-31 09:15:39,822] Trial 48 finished with value: 0.3658547206779779 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.02080274695569518, 'depth': 6, 'boosting_type': 'Ordered', 'bootstrap_type': 'Bayesian', 'bagging_temperature': 5.808099692047835}. Best is trial 11 with value: 0.5761730298609979.
    [I 2024-05-31 09:16:09,431] Trial 49 finished with value: 0.5653251575380815 and parameters: {'objective': 'CrossEntropy', 'colsample_bylevel': 0.07727526429698184, 'depth': 8, 'boosting_type': 'Plain', 'bootstrap_type': 'Bernoulli', 'subsample': 0.3278786422067817}. Best is trial 11 with value: 0.5761730298609979.
    

**ANN**


```python
# Function to create model
def nn_cl_bo2(neurons, activation, optimizer, learning_rate, batch_size, epochs,
              layers1, layers2, normalization, dropout, dropout_rate, get_model = False):
    optimizerL = ['SGD',
                  'Adam', 'RMSprop', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl']
    optimizerD= {'Adam':Adam(learning_rate=learning_rate), 'SGD':SGD(learning_rate=learning_rate),
                 'RMSprop':RMSprop(learning_rate=learning_rate), 'Adadelta':Adadelta(learning_rate=learning_rate),
                 'Adagrad':Adagrad(learning_rate=learning_rate), 'Adamax':Adamax(learning_rate=learning_rate),
                 'Nadam':Nadam(learning_rate=learning_rate), 'Ftrl':Ftrl(learning_rate=learning_rate)}
    activationL = ['relu', 'sigmoid', 'softplus', 'softsign', 'tanh', 'selu',
                   'elu','relu']

    activation = activationL[activation]
    optimizer = optimizerD[optimizerL[optimizer]]

    def nn_cl_fun():
        nn = Sequential()
        nn.add(Dense(units=neurons, activation=activation,  input_shape=(X_train_prepared.shape[1],)))
        if normalization:
            nn.add(BatchNormalization())
        for i in range(layers1):
             nn.add(Dense(units=neurons, activation=activation))
        if dropout:
            nn.add(Dropout(dropout_rate))
        for i in range(layers2):
            nn.add(Dense(units=neurons, activation=activation))
        if dropout:
            nn.add(Dropout(dropout_rate))

        nn.add(Dense(units =1, activation = 'sigmoid'))
        nn.compile(optimizer=optimizer, loss='binary_crossentropy')
        return nn

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=round(epochs/4))
    nn = KerasClassifier(nn_cl_fun(), epochs=epochs, batch_size=batch_size, validation_split=0.2, metrics= ['f1_score'], verbose = 0)
    if get_model:
        return nn
    
    kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=21)
    score = cross_val_score(nn, X_train_prepared, y_train, scoring='f1', cv=kfold).mean()

    return score

```


```python
def tune_ann(trial:optuna.Trial):
  # list des hyperparmaètres
    neurons = trial.suggest_int('neurons', 4, 32)
    activation = trial.suggest_int('activation', 0, 7)
    optimizer = trial.suggest_int('optimizer', 0, 7)
    learning_rate = trial.suggest_float('learning_rate', 1e-3, 1, log = True)
    batch_size = trial.suggest_int('batch_size', 1, 50, log = True)
    epochs = trial.suggest_int('epochs', 10, 100)
    layers1 = trial.suggest_int('layers1', 1, 2)
    layers2 = trial.suggest_int('layers2', 0, 2)
    normalization = trial.suggest_categorical('normalization', [True, False]),
    dropout = trial.suggest_categorical('dropout', [True, False]),
    dropout_rate = trial.suggest_float('dropout_rate', 0, 0.3)

    return nn_cl_bo2(neurons=neurons, activation=activation, optimizer=optimizer, learning_rate=learning_rate,
              batch_size=batch_size, epochs=epochs,  layers1=layers1, layers2=layers2,
              normalization=normalization, dropout=dropout, dropout_rate=dropout_rate
              )
```


```python
sampler = TPESampler(seed=21)
study_ann = optuna.create_study(direction='maximize', sampler = sampler)
study_ann.optimize(tune_ann, n_trials = 50)
```

    [I 2024-05-31 09:16:09,462] A new study created in memory with name: no-name-11bf16f8-d1c0-4065-887a-12a3a2149df1
    [I 2024-05-31 09:21:02,365] Trial 0 finished with value: 0.0 and parameters: {'neurons': 5, 'activation': 2, 'optimizer': 5, 'learning_rate': 0.0011610441905361485, 'batch_size': 1, 'epochs': 14, 'layers1': 1, 'layers2': 1, 'normalization': False, 'dropout': False, 'dropout_rate': 0.03997215577552432}. Best is trial 0 with value: 0.0.
    [I 2024-05-31 09:21:52,723] Trial 1 finished with value: 0.281031199675184 and parameters: {'neurons': 9, 'activation': 3, 'optimizer': 6, 'learning_rate': 0.1891609587474209, 'batch_size': 44, 'epochs': 79, 'layers1': 1, 'layers2': 1, 'normalization': True, 'dropout': False, 'dropout_rate': 0.22822682089938184}. Best is trial 1 with value: 0.281031199675184.
    [I 2024-05-31 09:30:40,936] Trial 2 finished with value: 0.48448314507456 and parameters: {'neurons': 18, 'activation': 1, 'optimizer': 2, 'learning_rate': 0.007109337552527563, 'batch_size': 2, 'epochs': 51, 'layers1': 2, 'layers2': 0, 'normalization': True, 'dropout': True, 'dropout_rate': 0.18444741334453038}. Best is trial 2 with value: 0.48448314507456.
    [I 2024-05-31 09:35:10,555] Trial 3 finished with value: 0.5388038194596628 and parameters: {'neurons': 22, 'activation': 3, 'optimizer': 3, 'learning_rate': 0.2694465631697866, 'batch_size': 8, 'epochs': 91, 'layers1': 1, 'layers2': 2, 'normalization': True, 'dropout': True, 'dropout_rate': 0.1363170922832249}. Best is trial 3 with value: 0.5388038194596628.
    [I 2024-05-31 09:55:45,655] Trial 4 finished with value: 0.10543106131341426 and parameters: {'neurons': 12, 'activation': 6, 'optimizer': 1, 'learning_rate': 0.015991510790116688, 'batch_size': 1, 'epochs': 63, 'layers1': 2, 'layers2': 0, 'normalization': True, 'dropout': True, 'dropout_rate': 0.24580687372636909}. Best is trial 3 with value: 0.5388038194596628.
    [I 2024-05-31 09:56:35,539] Trial 5 finished with value: 0.5723821635838595 and parameters: {'neurons': 17, 'activation': 5, 'optimizer': 3, 'learning_rate': 0.39107247392103084, 'batch_size': 13, 'epochs': 25, 'layers1': 1, 'layers2': 2, 'normalization': False, 'dropout': False, 'dropout_rate': 0.24125350273353913}. Best is trial 5 with value: 0.5723821635838595.
    [I 2024-05-31 10:02:29,888] Trial 6 finished with value: 0.49816210768330904 and parameters: {'neurons': 30, 'activation': 1, 'optimizer': 6, 'learning_rate': 0.003389914059548476, 'batch_size': 3, 'epochs': 44, 'layers1': 2, 'layers2': 1, 'normalization': True, 'dropout': True, 'dropout_rate': 0.062490915300902415}. Best is trial 5 with value: 0.5723821635838595.
    [I 2024-05-31 10:29:15,893] Trial 7 finished with value: 0.16215646048708976 and parameters: {'neurons': 27, 'activation': 2, 'optimizer': 5, 'learning_rate': 0.5901659442141964, 'batch_size': 1, 'epochs': 81, 'layers1': 1, 'layers2': 1, 'normalization': False, 'dropout': True, 'dropout_rate': 0.022368360351132422}. Best is trial 5 with value: 0.5723821635838595.
    [I 2024-05-31 10:48:15,263] Trial 8 finished with value: 0.13246638892030285 and parameters: {'neurons': 21, 'activation': 5, 'optimizer': 1, 'learning_rate': 0.06986999682226873, 'batch_size': 1, 'epochs': 52, 'layers1': 2, 'layers2': 2, 'normalization': True, 'dropout': True, 'dropout_rate': 0.0006301502678520921}. Best is trial 5 with value: 0.5723821635838595.
    [I 2024-05-31 10:53:08,853] Trial 9 finished with value: 0.40782302072802656 and parameters: {'neurons': 23, 'activation': 4, 'optimizer': 3, 'learning_rate': 0.020340302738147163, 'batch_size': 3, 'epochs': 35, 'layers1': 2, 'layers2': 2, 'normalization': False, 'dropout': True, 'dropout_rate': 0.06864046093886547}. Best is trial 5 with value: 0.5723821635838595.
    [I 2024-05-31 10:53:28,469] Trial 10 finished with value: 0.0 and parameters: {'neurons': 15, 'activation': 7, 'optimizer': 0, 'learning_rate': 0.8829695114821658, 'batch_size': 16, 'epochs': 12, 'layers1': 1, 'layers2': 2, 'normalization': False, 'dropout': False, 'dropout_rate': 0.29544705719536124}. Best is trial 5 with value: 0.5723821635838595.
    [I 2024-05-31 10:56:50,312] Trial 11 finished with value: 0.5451858950035693 and parameters: {'neurons': 24, 'activation': 4, 'optimizer': 3, 'learning_rate': 0.16866740416140152, 'batch_size': 11, 'epochs': 90, 'layers1': 1, 'layers2': 2, 'normalization': False, 'dropout': False, 'dropout_rate': 0.1325741983257798}. Best is trial 5 with value: 0.5723821635838595.
    [I 2024-05-31 10:59:23,197] Trial 12 finished with value: 0.5534041551279212 and parameters: {'neurons': 26, 'activation': 5, 'optimizer': 4, 'learning_rate': 0.08966564265165505, 'batch_size': 16, 'epochs': 100, 'layers1': 1, 'layers2': 2, 'normalization': False, 'dropout': False, 'dropout_rate': 0.12903024728770407}. Best is trial 5 with value: 0.5723821635838595.
    [I 2024-05-31 10:59:48,148] Trial 13 finished with value: 0.5722051249220452 and parameters: {'neurons': 31, 'activation': 6, 'optimizer': 4, 'learning_rate': 0.06971081977202268, 'batch_size': 31, 'epochs': 27, 'layers1': 1, 'layers2': 2, 'normalization': False, 'dropout': False, 'dropout_rate': 0.1825319089185175}. Best is trial 5 with value: 0.5723821635838595.
    [I 2024-05-31 11:00:09,038] Trial 14 finished with value: 0.0 and parameters: {'neurons': 32, 'activation': 7, 'optimizer': 7, 'learning_rate': 0.045356209338963775, 'batch_size': 45, 'epochs': 28, 'layers1': 1, 'layers2': 2, 'normalization': False, 'dropout': False, 'dropout_rate': 0.19948108198924439}. Best is trial 5 with value: 0.5723821635838595.
    [I 2024-05-31 11:00:36,065] Trial 15 finished with value: 0.5451708873745464 and parameters: {'neurons': 15, 'activation': 6, 'optimizer': 4, 'learning_rate': 0.4593563340522702, 'batch_size': 26, 'epochs': 26, 'layers1': 1, 'layers2': 1, 'normalization': False, 'dropout': False, 'dropout_rate': 0.2642186430320374}. Best is trial 5 with value: 0.5723821635838595.
    [I 2024-05-31 11:02:02,713] Trial 16 finished with value: 0.10099907190042037 and parameters: {'neurons': 18, 'activation': 5, 'optimizer': 2, 'learning_rate': 0.11140033834749842, 'batch_size': 6, 'epochs': 21, 'layers1': 1, 'layers2': 2, 'normalization': False, 'dropout': False, 'dropout_rate': 0.17764899507325282}. Best is trial 5 with value: 0.5723821635838595.
    [I 2024-05-31 11:02:42,732] Trial 17 finished with value: 0.5563757525209978 and parameters: {'neurons': 9, 'activation': 6, 'optimizer': 5, 'learning_rate': 0.03284732321715036, 'batch_size': 23, 'epochs': 37, 'layers1': 1, 'layers2': 0, 'normalization': False, 'dropout': False, 'dropout_rate': 0.21996834778732793}. Best is trial 5 with value: 0.5723821635838595.
    [I 2024-05-31 11:03:35,278] Trial 18 finished with value: 0.0005370569280343716 and parameters: {'neurons': 19, 'activation': 7, 'optimizer': 2, 'learning_rate': 0.32895528674177177, 'batch_size': 30, 'epochs': 61, 'layers1': 1, 'layers2': 1, 'normalization': False, 'dropout': False, 'dropout_rate': 0.28592157018279246}. Best is trial 5 with value: 0.5723821635838595.
    [I 2024-05-31 11:04:52,800] Trial 19 finished with value: 0.0 and parameters: {'neurons': 28, 'activation': 5, 'optimizer': 4, 'learning_rate': 0.9943099654096184, 'batch_size': 11, 'epochs': 33, 'layers1': 1, 'layers2': 2, 'normalization': False, 'dropout': False, 'dropout_rate': 0.17135784389058903}. Best is trial 5 with value: 0.5723821635838595.
    [I 2024-05-31 11:05:24,609] Trial 20 finished with value: 0.0 and parameters: {'neurons': 13, 'activation': 0, 'optimizer': 7, 'learning_rate': 0.06515879417132968, 'batch_size': 17, 'epochs': 19, 'layers1': 2, 'layers2': 1, 'normalization': False, 'dropout': False, 'dropout_rate': 0.09773242365236676}. Best is trial 5 with value: 0.5723821635838595.
    [I 2024-05-31 11:06:05,021] Trial 21 finished with value: 0.5770407778234681 and parameters: {'neurons': 4, 'activation': 6, 'optimizer': 5, 'learning_rate': 0.02622366414731324, 'batch_size': 26, 'epochs': 42, 'layers1': 1, 'layers2': 0, 'normalization': False, 'dropout': False, 'dropout_rate': 0.21085630780768957}. Best is trial 21 with value: 0.5770407778234681.
    [I 2024-05-31 11:06:38,889] Trial 22 finished with value: 0.5468752018459738 and parameters: {'neurons': 6, 'activation': 6, 'optimizer': 5, 'learning_rate': 0.008888306694862225, 'batch_size': 33, 'epochs': 43, 'layers1': 1, 'layers2': 0, 'normalization': False, 'dropout': False, 'dropout_rate': 0.21500603377771374}. Best is trial 21 with value: 0.5770407778234681.
    [I 2024-05-31 11:08:13,772] Trial 23 finished with value: 0.23780895123580495 and parameters: {'neurons': 8, 'activation': 4, 'optimizer': 6, 'learning_rate': 0.015084242187233504, 'batch_size': 11, 'epochs': 44, 'layers1': 1, 'layers2': 0, 'normalization': False, 'dropout': False, 'dropout_rate': 0.2608331938500796}. Best is trial 21 with value: 0.5770407778234681.
    [I 2024-05-31 11:08:41,030] Trial 24 finished with value: 0.5561538796671459 and parameters: {'neurons': 11, 'activation': 6, 'optimizer': 4, 'learning_rate': 0.037917295876939455, 'batch_size': 22, 'epochs': 23, 'layers1': 1, 'layers2': 1, 'normalization': False, 'dropout': False, 'dropout_rate': 0.15663793224266254}. Best is trial 21 with value: 0.5770407778234681.
    [I 2024-05-31 11:08:59,458] Trial 25 finished with value: 0.22754887254041029 and parameters: {'neurons': 4, 'activation': 5, 'optimizer': 3, 'learning_rate': 0.004218953819596292, 'batch_size': 50, 'epochs': 29, 'layers1': 1, 'layers2': 2, 'normalization': False, 'dropout': False, 'dropout_rate': 0.24366811574106717}. Best is trial 21 with value: 0.5770407778234681.
    [I 2024-05-31 11:10:44,873] Trial 26 finished with value: 0.523199569326859 and parameters: {'neurons': 32, 'activation': 7, 'optimizer': 4, 'learning_rate': 0.14944845193998704, 'batch_size': 8, 'epochs': 38, 'layers1': 1, 'layers2': 0, 'normalization': False, 'dropout': False, 'dropout_rate': 0.19852797518000545}. Best is trial 21 with value: 0.5770407778234681.
    [I 2024-05-31 11:10:55,281] Trial 27 finished with value: 0.5759760541825374 and parameters: {'neurons': 15, 'activation': 6, 'optimizer': 5, 'learning_rate': 0.02389605870444378, 'batch_size': 33, 'epochs': 10, 'layers1': 1, 'layers2': 1, 'normalization': False, 'dropout': False, 'dropout_rate': 0.2733405964786116}. Best is trial 21 with value: 0.5770407778234681.
    [I 2024-05-31 11:11:19,073] Trial 28 finished with value: 0.28988173455978977 and parameters: {'neurons': 15, 'activation': 4, 'optimizer': 6, 'learning_rate': 0.02219982664314045, 'batch_size': 18, 'epochs': 16, 'layers1': 1, 'layers2': 0, 'normalization': False, 'dropout': False, 'dropout_rate': 0.26625788148945634}. Best is trial 21 with value: 0.5770407778234681.
    [I 2024-05-31 11:12:10,358] Trial 29 finished with value: 0.5368171683389075 and parameters: {'neurons': 6, 'activation': 5, 'optimizer': 5, 'learning_rate': 0.0021305254611215544, 'batch_size': 5, 'epochs': 10, 'layers1': 1, 'layers2': 1, 'normalization': False, 'dropout': False, 'dropout_rate': 0.278008595557476}. Best is trial 21 with value: 0.5770407778234681.
    [I 2024-05-31 11:12:24,087] Trial 30 finished with value: 0.492955908336101 and parameters: {'neurons': 20, 'activation': 7, 'optimizer': 5, 'learning_rate': 0.0011702669688616438, 'batch_size': 35, 'epochs': 15, 'layers1': 1, 'layers2': 1, 'normalization': False, 'dropout': False, 'dropout_rate': 0.24194285546012445}. Best is trial 21 with value: 0.5770407778234681.
    [I 2024-05-31 11:12:42,470] Trial 31 finished with value: 0.5779060052755283 and parameters: {'neurons': 17, 'activation': 6, 'optimizer': 4, 'learning_rate': 0.011073014022318013, 'batch_size': 36, 'epochs': 22, 'layers1': 1, 'layers2': 2, 'normalization': False, 'dropout': False, 'dropout_rate': 0.19621775000806127}. Best is trial 31 with value: 0.5779060052755283.
    [I 2024-05-31 11:12:58,980] Trial 32 finished with value: 0.5851045063184845 and parameters: {'neurons': 17, 'activation': 6, 'optimizer': 6, 'learning_rate': 0.009319362752873667, 'batch_size': 38, 'epochs': 18, 'layers1': 1, 'layers2': 2, 'normalization': False, 'dropout': False, 'dropout_rate': 0.2109709092553926}. Best is trial 32 with value: 0.5851045063184845.
    [I 2024-05-31 11:13:08,318] Trial 33 finished with value: 0.5273209408980313 and parameters: {'neurons': 17, 'activation': 6, 'optimizer': 6, 'learning_rate': 0.009587658428972372, 'batch_size': 39, 'epochs': 10, 'layers1': 1, 'layers2': 0, 'normalization': False, 'dropout': False, 'dropout_rate': 0.21136334935470816}. Best is trial 32 with value: 0.5851045063184845.
    [I 2024-05-31 11:13:21,424] Trial 34 finished with value: 0.0 and parameters: {'neurons': 13, 'activation': 7, 'optimizer': 7, 'learning_rate': 0.00655299923932529, 'batch_size': 50, 'epochs': 18, 'layers1': 1, 'layers2': 1, 'normalization': True, 'dropout': False, 'dropout_rate': 0.2013325258832279}. Best is trial 32 with value: 0.5851045063184845.
    [I 2024-05-31 11:14:38,505] Trial 35 finished with value: 0.5571379540453147 and parameters: {'neurons': 10, 'activation': 3, 'optimizer': 6, 'learning_rate': 0.013723616625035243, 'batch_size': 22, 'epochs': 61, 'layers1': 1, 'layers2': 2, 'normalization': True, 'dropout': False, 'dropout_rate': 0.15181331080461652}. Best is trial 32 with value: 0.5851045063184845.
    [I 2024-05-31 11:15:03,340] Trial 36 finished with value: 0.5460699333066841 and parameters: {'neurons': 16, 'activation': 6, 'optimizer': 5, 'learning_rate': 0.0058152852584022925, 'batch_size': 38, 'epochs': 32, 'layers1': 1, 'layers2': 1, 'normalization': False, 'dropout': False, 'dropout_rate': 0.22818595833908734}. Best is trial 32 with value: 0.5851045063184845.
    [I 2024-05-31 11:15:26,036] Trial 37 finished with value: 0.48768561245191533 and parameters: {'neurons': 13, 'activation': 6, 'optimizer': 6, 'learning_rate': 0.023814451013801648, 'batch_size': 28, 'epochs': 21, 'layers1': 2, 'layers2': 0, 'normalization': True, 'dropout': True, 'dropout_rate': 0.11379169615604566}. Best is trial 32 with value: 0.5851045063184845.
    [I 2024-05-31 11:16:52,896] Trial 38 finished with value: 0.5786567418501413 and parameters: {'neurons': 20, 'activation': 7, 'optimizer': 5, 'learning_rate': 0.010793375689574876, 'batch_size': 21, 'epochs': 68, 'layers1': 1, 'layers2': 2, 'normalization': False, 'dropout': False, 'dropout_rate': 0.16629217962078668}. Best is trial 32 with value: 0.5851045063184845.
    [I 2024-05-31 11:18:14,167] Trial 39 finished with value: 0.0 and parameters: {'neurons': 25, 'activation': 7, 'optimizer': 7, 'learning_rate': 0.00419602684283227, 'batch_size': 22, 'epochs': 66, 'layers1': 1, 'layers2': 2, 'normalization': False, 'dropout': True, 'dropout_rate': 0.15904892690189595}. Best is trial 32 with value: 0.5851045063184845.
    [I 2024-05-31 11:20:30,223] Trial 40 finished with value: 0.5506910098069695 and parameters: {'neurons': 22, 'activation': 2, 'optimizer': 5, 'learning_rate': 0.010847533395228315, 'batch_size': 14, 'epochs': 73, 'layers1': 2, 'layers2': 2, 'normalization': True, 'dropout': False, 'dropout_rate': 0.168297817637493}. Best is trial 32 with value: 0.5851045063184845.
    [I 2024-05-31 11:21:10,496] Trial 41 finished with value: 0.5221304409720278 and parameters: {'neurons': 20, 'activation': 7, 'optimizer': 5, 'learning_rate': 0.024371258570143977, 'batch_size': 41, 'epochs': 56, 'layers1': 1, 'layers2': 2, 'normalization': False, 'dropout': False, 'dropout_rate': 0.22701744578399408}. Best is trial 32 with value: 0.5851045063184845.
    [I 2024-05-31 11:22:20,940] Trial 42 finished with value: 0.5684296964171525 and parameters: {'neurons': 19, 'activation': 6, 'optimizer': 5, 'learning_rate': 0.01287004686866548, 'batch_size': 29, 'epochs': 74, 'layers1': 1, 'layers2': 2, 'normalization': False, 'dropout': False, 'dropout_rate': 0.19071930550382188}. Best is trial 32 with value: 0.5851045063184845.
    [I 2024-05-31 11:23:28,260] Trial 43 finished with value: 0.44039501022580485 and parameters: {'neurons': 22, 'activation': 5, 'optimizer': 6, 'learning_rate': 0.006759126939680346, 'batch_size': 20, 'epochs': 49, 'layers1': 1, 'layers2': 2, 'normalization': False, 'dropout': False, 'dropout_rate': 0.21034771386504997}. Best is trial 32 with value: 0.5851045063184845.
    [I 2024-05-31 11:24:16,610] Trial 44 finished with value: 0.5740273274165558 and parameters: {'neurons': 17, 'activation': 6, 'optimizer': 4, 'learning_rate': 0.046694864213971184, 'batch_size': 37, 'epochs': 68, 'layers1': 1, 'layers2': 1, 'normalization': False, 'dropout': True, 'dropout_rate': 0.2521849596608845}. Best is trial 32 with value: 0.5851045063184845.
    [I 2024-05-31 11:27:12,618] Trial 45 finished with value: 0.4062396117711099 and parameters: {'neurons': 14, 'activation': 7, 'optimizer': 6, 'learning_rate': 0.002697631259235903, 'batch_size': 2, 'epochs': 14, 'layers1': 1, 'layers2': 2, 'normalization': False, 'dropout': False, 'dropout_rate': 0.13693406287010268}. Best is trial 32 with value: 0.5851045063184845.
    [I 2024-05-31 11:28:11,382] Trial 46 finished with value: 0.3949041026319959 and parameters: {'neurons': 19, 'activation': 5, 'optimizer': 3, 'learning_rate': 0.017406255197991768, 'batch_size': 26, 'epochs': 56, 'layers1': 1, 'layers2': 2, 'normalization': False, 'dropout': False, 'dropout_rate': 0.10135179989062221}. Best is trial 32 with value: 0.5851045063184845.
    [I 2024-05-31 11:30:50,420] Trial 47 finished with value: 0.5805888844658535 and parameters: {'neurons': 16, 'activation': 4, 'optimizer': 5, 'learning_rate': 0.02917540753862082, 'batch_size': 13, 'epochs': 81, 'layers1': 1, 'layers2': 2, 'normalization': False, 'dropout': False, 'dropout_rate': 0.23554143268927036}. Best is trial 32 with value: 0.5851045063184845.
    [I 2024-05-31 11:35:16,349] Trial 48 finished with value: 0.5615335550545555 and parameters: {'neurons': 24, 'activation': 3, 'optimizer': 4, 'learning_rate': 0.008695059402012003, 'batch_size': 8, 'epochs': 89, 'layers1': 1, 'layers2': 2, 'normalization': True, 'dropout': True, 'dropout_rate': 0.19083619687459583}. Best is trial 32 with value: 0.5851045063184845.
    [I 2024-05-31 11:38:08,120] Trial 49 finished with value: 0.0 and parameters: {'neurons': 21, 'activation': 4, 'optimizer': 6, 'learning_rate': 0.04818179053989475, 'batch_size': 13, 'epochs': 85, 'layers1': 1, 'layers2': 2, 'normalization': False, 'dropout': False, 'dropout_rate': 0.23404207161845317}. Best is trial 32 with value: 0.5851045063184845.
    

## 4. Résultats

### 4.1 Comparaison des modèles


```python
# Si solver_l1 ou solver_l2 remettre solver comme key
best_param_lr = {k if not k.startswith('solver_') else 'solver': v for k, v in study_lr.best_params.items()}

best_model = {
    'best_lr': LogisticRegression(**best_param_lr),
    'best_rf': RandomForestClassifier(**study_rf.best_params), 
    'best_xgb': XGBClassifier(**study_xgb.best_params), 
    'best_catboost': CatBoostClassifier(**study_catboost.best_params),
    'best_ann':  nn_cl_bo2(**study_ann.best_params, get_model=True)
    }
```


```python
metrics = ['recall', 'precision', 'f1', 'accuracy', 'auc_score']
results = {}

for name, model in best_model.items():

    if name == 'best_catboost': 
        model.fit(X_train_prepared, y_train, plot = False, logging_level='Silent')
    else : 
        model.fit(X_train_prepared, y_train)
    
    y_pred = model.predict(X_test_prepared)
    y_pred_proba = model.predict_proba(X_test_prepared)[:, 1]

    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)

    results[name] = {
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'accuracy': accuracy,
        'auc_score': auc_score
    }

df_results = pd.DataFrame.from_dict(results, orient='index')
df_results.columns = metrics

df_results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>recall</th>
      <th>precision</th>
      <th>f1</th>
      <th>accuracy</th>
      <th>auc_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>best_lr</th>
      <td>0.421725</td>
      <td>0.678082</td>
      <td>0.520026</td>
      <td>0.911233</td>
      <td>0.934354</td>
    </tr>
    <tr>
      <th>best_rf</th>
      <td>0.489883</td>
      <td>0.677467</td>
      <td>0.568603</td>
      <td>0.915240</td>
      <td>0.944159</td>
    </tr>
    <tr>
      <th>best_xgb</th>
      <td>0.560170</td>
      <td>0.676963</td>
      <td>0.613054</td>
      <td>0.919369</td>
      <td>0.949215</td>
    </tr>
    <tr>
      <th>best_catboost</th>
      <td>0.537806</td>
      <td>0.695592</td>
      <td>0.606607</td>
      <td>0.920461</td>
      <td>0.949933</td>
    </tr>
    <tr>
      <th>best_ann</th>
      <td>0.534611</td>
      <td>0.632242</td>
      <td>0.579342</td>
      <td>0.911475</td>
      <td>0.938765</td>
    </tr>
  </tbody>
</table>
</div>




```python
model_names = list(best_model.keys())
plt.figure(figsize=(10, 8))

for i, (name, model) in enumerate(best_model.items()):
    y_pred_proba = model.predict_proba(X_test_prepared)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.plot(recall, precision, label=f'{name} (AP = {average_precision_score(y_test, y_pred_proba):.2f})')

plt.legend(loc='lower left')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Courbe Precision/Recall')

plt.show()
```


    
![png](banking_predictions_files/banking_predictions_101_0.png)
    



```python
model_names = list(best_model.keys())
plt.figure(figsize=(10, 8))

for i, (name, model) in enumerate(best_model.items()):
    
    y_pred_proba = model.predict_proba(X_test_prepared)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_test, y_pred_proba):.2f})')


plt.plot([0, 1], [0, 1], 'k--', lw=2)

plt.legend(loc='lower right')
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('Courbe ROC')
plt.show()
```


    
![png](banking_predictions_files/banking_predictions_102_0.png)
    


### 4.2 Feature importance

Nous décidons également de regarder la feature importance de notre meilleur modèle. Pour
cela nous utilisons la méthode des permutations implémentée par Sklearn.
Cette méthode consiste à mesurer la diminution de la performance d'un modèle lorsqu'on
permute de manière aléatoire les valeurs d'une caractéristique particulière, ce qui permet de
quantifier l'impact de cette caractéristique sur les prédictions du modèle. Plus la performance
du modèle diminue après permutation, plus la caractéristique est jugée importante


```python
from sklearn.inspection import permutation_importance

num_cols = var_num_final
cat_cols = pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(var_cat_final)
all_cols = num_cols + list(cat_cols)

result = permutation_importance(
    best_model['best_rf'], X_test_prepared, y_test, n_repeats=10, random_state=42, n_jobs=2
)

```


```python
features_importances = pd.Series(result.importances_mean, index=all_cols).sort_values(ascending=False)


fig, ax = plt.subplots(figsize=(10, 6))
features_importances.plot.bar(yerr=result.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()
```


    
![png](banking_predictions_files/banking_predictions_106_0.png)
    


Nous voyons dès lors que sur les 55 variables, beaucoup ne sont que peu significatives.
Également que la variable duration a une part importante dans l’explicabilité.

### 4.3 Calibration des modèles

Dans cette dernière partie, nous analysons la calibration du modèle. La sortie des modèles étant des scores de probabilités on attendrait d'un modèle parfaitement calibré que lorsque ce dernier prédit une probabilité de 90%,
la proportion observée soit effectivement de 90% ; et ce pour tout pourcentage.


```python
from sklearn.calibration import CalibrationDisplay

fig = plt.figure(figsize=(10, 10))
gs = GridSpec(4, 2)
colors = plt.get_cmap("Dark2")

ax_calibration_curve = fig.add_subplot(gs[:2, :2])
calibration_displays = {}

for i, (name, model) in enumerate(models.items()):
    if name == 'Catboost': 
        model.fit(X_train_prepared, y_train, plot=False, logging_level='Silent')
    else:
        model.fit(X_train_prepared, y_train)
    
    display = CalibrationDisplay.from_estimator(
        model,
        X_test_prepared,
        y_test,
        n_bins=10,
        name=name,
        ax=ax_calibration_curve,
        color=colors(i),
    )
    calibration_displays[name] = display

ax_calibration_curve.set_title("Calibration plots for all models")

grid_positions = [(2, 0), (2, 1), (3, 0), (3, 1)]
for i, (name, _) in enumerate(models.items()):
    row, col = grid_positions[i]
    ax = fig.add_subplot(gs[row, col])

    ax.hist(
        calibration_displays[name].y_prob,
        range=(0, 1),
        bins=10,
        label=name,
        color=colors(i),
    )
    ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")

plt.tight_layout()
plt.show()
```


    
![png](banking_predictions_files/banking_predictions_110_0.png)
    


L'ensemble des modèles sont relativement bien calibrés avec Catboost comme modèle ayant la meilleure calibration. 

