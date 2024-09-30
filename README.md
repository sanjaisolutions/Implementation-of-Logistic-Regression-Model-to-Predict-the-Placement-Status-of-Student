# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Data Preparation and Scaling.
2.Model Training and Prediction.
3.Model intializing.
3.Inverse Transformation and Evaluation.

## Program:
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: R.SANJAI
RegisterNumber: 212223040180 
*/

import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Removes the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")# A library for large linear classification
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)# Accuracy Score = (TP+TN)/
#accuracy_score(y_true,y_prednormalize=False)
accuracy

from sklearn.metrics import confusion_matrix
confusion = (y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
# 1.PLACEMENT DATA:
![Screenshot 2024-09-30 093503](https://github.com/user-attachments/assets/04022351-3efc-4c96-83ed-fbfe9fccc6c9)

# 2.SALARY DATA:
![image](https://github.com/user-attachments/assets/3ad75df0-5516-4fdb-8dab-daf89260741a)

# 3.CHECKING THE NULL() FUNCTION:
![Screenshot 2024-09-30 093721](https://github.com/user-attachments/assets/7682ece6-b2f1-41be-bd4a-d897502a484e)

# 4.DATA DUPLICATE:
![Screenshot 2024-09-30 093818](https://github.com/user-attachments/assets/823b4178-cf7a-4abd-ab71-d0d50554ebb3)

# 5.PRINT DATA:
![Screenshot 2024-09-30 093908](https://github.com/user-attachments/assets/64715a1e-e95e-4798-866a-c88a085fe5e1)

# 6.DATA STATUS:
![Screenshot 2024-09-30 094013](https://github.com/user-attachments/assets/9d53b8ce-4ca4-445b-8a3e-901532724a02)
![Screenshot 2024-09-30 094055](https://github.com/user-attachments/assets/4f23624a-6fbc-4e74-a7ce-325adeec57c8)

# 7.Y_PREDICATION ARRAY
![Screenshot 2024-09-30 094114](https://github.com/user-attachments/assets/c827b871-192f-4ae0-9d51-b1b96a3ab30e)

# 8.ACCURACY VALUE:
![Screenshot 2024-09-30 094136](https://github.com/user-attachments/assets/9253ef4e-78c0-45a6-a300-5cf58ff0f90e)

# 9.CONFUSION ARRAY:

![Screenshot 2024-09-30 094221](https://github.com/user-attachments/assets/9b47b27d-d338-46ea-9369-c31276b8b843)

# 10.CLASSIFICATION REPORT:

![Screenshot 2024-09-30 094314](https://github.com/user-attachments/assets/13ee32d4-d752-4ab9-9574-1bca0d998385)

# 11.PREDICTION OF LR:



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
