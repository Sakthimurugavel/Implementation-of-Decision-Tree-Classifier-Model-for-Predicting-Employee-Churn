# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: SAKTHIVEL M
RegisterNumber:  212222240088
*/
import pandas as pd
data=pd.read_csv("/content/Employee_EX6.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours",
        "time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt= DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:

## 1.Head:
![ou 1 ml](https://github.com/Sakthimurugavel/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707246/8df06384-9d61-40d0-87ce-7a0121b408c2)
## 2.Accuracy:
![ou 2 ml](https://github.com/Sakthimurugavel/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707246/36e47c74-4f5c-48b3-ae19-94985a4e34fe)
## 3. Predict:
![ou 3 ml](https://github.com/Sakthimurugavel/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707246/245502a1-0849-41fa-b36b-72d752479729)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
