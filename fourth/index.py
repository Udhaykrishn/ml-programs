import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


X = np.array([[80,75],[95,90],[60,65],[45,30],[30,40],[85,95],[70,60],[50,55],[40,45],[60,70]])
y = np.array([1,1,0,0,0,1,1,0,0,1])

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

acc = accuracy_score(y_test,y_pred)
print("Accuracy on the test set: {:.2f}".format(acc))

exam_score1 = float(input("Enter Exam Score 1: "))
exam_score2 = float(input("Enter Exam Score 2: "))

user_input = np.array([[exam_score1,exam_score2]])

pre_outcome = knn.predict(user_input)

if pre_outcome[0] == 1:
    print("Based on the exam scores provided, the student is predicted to pass")
else:
    print("Based on the exam scores provided, the student is predicted to fail.")
