import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Oraganizing data -> man = 0, woman = 1
music_data = pd.read_csv('sklearn/simple_music/music.csv')
X = music_data.drop(columns=['genre'])
y = music_data['genre']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#Building model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

#Validating model
score = accuracy_score(y_test,predictions)
print(score)
print(predictions)