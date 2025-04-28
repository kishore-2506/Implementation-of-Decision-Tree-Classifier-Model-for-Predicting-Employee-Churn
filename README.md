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

Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Kishore M
RegisterNumber:  212223040100

```

```python

import pandas as pd

data = pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data["salary"] = le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project", "average_montly_hours",
"time_spend_company", "Work_accident","promotion_last_5years","salary"]]
x.head()

y = data["left"]

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn. tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt. predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)

accuracy
dt.predict([[0.5,0.8,9,260, 6,0,1,2]])

```

## Output:

### DATASET:

![image](https://github.com/user-attachments/assets/99fb663f-8269-4e8c-8f01-6a19cab16ffa)

![image](https://github.com/user-attachments/assets/582459c4-ffd6-41f3-ae49-43d2a021fe99)

![image](https://github.com/user-attachments/assets/af4951cd-4eba-43c2-b025-c738c78d028d)

![image](https://github.com/user-attachments/assets/22927521-9276-4632-a431-1d3c3deff7c5)




### LABELLED DATA:

![image](https://github.com/user-attachments/assets/9ad48017-5be8-4ef1-82de-86cb8b260e14)


### ACCURACY:

![image](https://github.com/user-attachments/assets/73d19cd0-bc51-4df0-9d13-dd7eb2af3799)

![image](https://github.com/user-attachments/assets/3b7c34eb-a9e5-4042-8b24-8719cc665187)




## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
