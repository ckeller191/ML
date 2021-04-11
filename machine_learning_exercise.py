#Part One
"""
from sklearn.datasets import fetch_california_housing
cali = fetch_california_housing()

import seaborn as sns
import pandas as pd

cali_df = pd.DataFrame(cali.data, columns=cali.feature_names)



sns.set(font_scale=1.1)
sns.set_style('whitegrid')
grid = sns.pairplot(data=cali_df, vars=cali_df.columns)

import matplotlib.pyplot as plt
plt.show()
"""


#Part Two

import pandas as pd1

nyc = pd1.read_csv('ave_yearly_temp_nyc_1895-2017.csv')

nyc.columns = ['Date', 'Temperature', 'Anomaly']

nyc.Date = nyc.Date.floordiv(100)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(nyc.Date.values.reshape(-1, 1), nyc.Temperature.values, 
random_state=11)

from sklearn.linear_model import LinearRegression

linear_regresion = LinearRegression()

linear_regresion.fit(X=X_train, y=y_train)

predicted = linear_regresion.predict(X_test)

expected = y_test

for p, e in zip(predicted[::5], expected[::5]):
    print(f'predicted: {p:.2f}, expected: {e:.2f}')

