import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
import nltk
import re
from nltk.corpus import stopwords
import string

data = pd.read_csv("C:\\users\\HP\\Downloads\\consumercomplaints.csv")
print(data.head())


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm
%matplotlib inline

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score



#The dataset contains an Unnamed column. I’ll remove the column and move further
data = data.drop("Unnamed: 0",axis=1)

#Now let’s have a look if the dataset contains null values or not
print(data.isnull().sum())

#The dataset contains so many null values. I’ll drop all the rows containing null values and move further
data = data.dropna()

#The product column in the dataset contains the labels. Here the labels represent the nature of the complaints reported by the consumers. Let’s
#have a look at all the labels and their frequency
print(data["Product"].value_counts())

data = data[["Consumer complaint narrative", "Product"]]
x = np.array(data["Consumer complaint narrative"])
y = np.array(data["Product"])

cv = CountVectorizer()
X = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.33, 
                                                    random_state=42)


sgdmodel = SGDClassifier()
sgdmodel.fit(X_train,y_train)


user = input("Enter a Text: ")
data = cv.transform([user]).toarray()
output = sgdmodel.predict(data)
print(output)


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

y_pred = sgdmodel.predict(X_test)

print("Confusion Matrix")
matrix = confusion_matrix(y_test, y_pred)
print(matrix)
# Classification Report
print("\nClassification Report")
report = classification_report(y_test, y_pred)
print(report)
# Accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('SGD Classifier Accuracy of the model: {:.2f}%'.format(accuracy*100))


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()

#train the model using the fit method. In the fit method, pass training datasets in it. x_train and y_train are the training datasets.
model.fit(X_train,y_train)

#Now predict the results using predict method.

y_pred=model.predict(X_test)

#The accuracy score tells us how accurately the model we build will predict and 
#the confusion matrix has a matrix with Actual values and predicted values. 
#For that, import accuracy_score and confusion_matrix from the sci-kit learn metric library.

from sklearn.metrics import accuracy_score,confusion_matrix
confusion_matrix(y_test,y_pred)

accuracy=accuracy_score(y_test,y_pred)*100
print("Accuracy of the model is {:.2f}".format(accuracy))

model=SVC()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
from sklearn.metrics import accuracy_score,confusion_matrix
confusion_matrix(y_test,y_pred)
accuracy=accuracy_score(y_test,y_pred)*100
print("Support Vector Classifier accuracy {:.2f}".format(accuracy))


accuracy_scores = np.zeros(4)

# Support Vector Classifier
clf = SVC().fit(X_train, y_train)
prediction = clf.predict(X_test)
accuracy_scores[0] = accuracy_score(y_test, prediction)*100
print('Support Vector Classifier accuracy: {}%'.format(accuracy_scores[0]))


'''' Examples of Complaints
1] On XXXX/XXXX/2022, I called Citi XXXX XXXX XXXX XXXX XXXX Customer Service at XXXX. I did not want to pay {$99.00} for the 
next year membership and wanted to cancel my card account. A customer service representative told me if I pay the {$99.00} 
membership fee and spending {$1000.00} in 3 months, I can get XXXX mileage reward points of XXXX XXXX. I believed what he 
said and paid {$99.00} membership fee on XXXX/XXXX/2022.   I spent more than {$1000.00} in 3 months since XXXX/XXXX/2022. 
On XXXX/XXXX/2022, I called the card Customer Service about my reward mileage points. I was total the reward mileage points 
are NOT XXXX. I can only get XXXX mileage points instead. I believe that the Citi XXXX XXXX XXXX XXXX XXXX Customer Service 
cheated me. This is business fraud!
''''

''''
2] Investigation took more than 30 days and nothing was changed when clearly there are misleading, incorrect, inaccurate 
items on my credit report..i have those two accounts attached showing those inaccuracies... I need them to follow the 
law because this is a violation of my rights!! The EVIDENCE IS IN BLACK AND WHITE ....
''''
