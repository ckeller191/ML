import pandas as pd

classes = pd.read_csv('animal_classes.csv')

train = pd.read_csv('animals_train.csv')

test = pd.read_csv('animals_test.csv')


train_copy = train.copy()
train_data = train_copy.drop(columns= ["class_number"])

test_copy = test.copy()
test_data = test_copy.drop(columns= ["animal_name"])

target_train = train["class_number"]

from sklearn.neighbors import KNeighborsClassifier


knn = KNeighborsClassifier()

knn.fit(X=train_data, y=target_train)

predicted = knn.predict(X=test_data)


animalnames = test["animal_name"]

class_prediction = [classes["Class_Type"][classes["Class_Number"] == i].values[0] for i in predicted]


final_df = pd.DataFrame({'animal_nar':animalnames, 'prediction':class_prediction})

final_df.to_csv("predictions.csv", index=False, sep=' ')





