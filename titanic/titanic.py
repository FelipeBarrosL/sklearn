import pandas as pd
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv("sklearn/titanic/train.csv")
test_data = pd.read_csv("sklearn/titanic/test.csv")

#Percentage of men and women that survived
women = train_data.loc[train_data['Sex'] == 'female']['Survived']
rate_women  = sum(women)/len(women)
men = train_data.loc[train_data['Sex'] == 'male']['Survived']
rate_men  = sum(men)/len(men)

#Selecting data for model development
y = train_data['Survived']
features = ["Pclass", 'Sex', 'SibSp', 'Parch']
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

#Building tree
model = RandomForestClassifier(n_estimators = 100, max_depth = 5, random_state = 1)
model.fit(X,y)
predicitions = model.predict(X_test)

#Saving prediciton in csv file
output = pd.DataFrame({'PassengerId' : test_data.PassengerId, 'Survived': predicitions})
output.to_csv('sklearn/titanic/submission.csv', index=False)