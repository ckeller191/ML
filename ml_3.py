from sklearn.datasets import fetch_california_housing

california = fetch_california_housing() #Bunch object
"""
print(california.DESCR)

print(california.data.shape)

print(california.target.shape)

print(california.feature_names)
"""

import pandas as pd

pd.set_option("precision", 4)
pd.set_option("max_columns", 9)
pd.set_option("display.width", None)


california_df = pd.DataFrame(california.data, columns=california.feature_names)

california_df["MedHouseValue"] = pd.Series(california.target)

#print(california_df.head()) #print top 5 rows of dataframe


#print(california_df.describe())

sample_df = california_df.sample(frac=0.1, random_state=17)
#takes 10% of data

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=2)
sns.set_style("whitegrid")

for feature in california.feature_names:
    plt.figure(figsize=(8, 4.5))
    sns.scatterplot(
        data=sample_df,
        x=feature,
        y="MedHouseValue",
        hue="MedHouseValue",
        palette="cool",
        legend=False
        )


#plt.show()



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(california.data, california.target, random_state=11)

from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()

linear_regression.fit(X=x_train, y=y_train)


#for i, name in enumerate(california.feature_names):
    #print(f"{name}: {linear_regression.coef_[i]}")


predicted = linear_regression.predict(x_test)
#print(predicted[:5])


expected = y_test
#print(expected[:5])


df = pd.DataFrame()

df["Expected"] = pd.Series(expected)

df["Predicted"] = pd.Series(predicted)

#print(df[:10])


import matplotlib.pyplot as plt2

figure = plt2.figure(figsize=(9, 9))

axes = sns.scatterplot(
    data=df, x="Expected", y="Predicted", hue="Predicted", palette="cool",
    legend=False
)

start = min(expected.min(), predicted.min())
print(start)

end = max(expected.max(), predicted.max())
print(end)

axes.set_xlim(start, end)
axes.set_ylim(start, end)


line = plt2.plot([start, end], [start, end])

plt2.show()


