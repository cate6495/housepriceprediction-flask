import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv("housing.csv")
print(df.head())

#separating the data into feature and target variables
X = df[['Avg. Area Income','Avg. Area House Age','Avg. Area Number of Rooms','Avg. Area Number of Bedrooms','Area Population']]
y = df['Price']

#X = df[["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width","Sepal_Length"]]
#y = df["Class"]

#separating dataset into training and split sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

#instantiate the model
linreg = LinearRegression()

#fit the model
linreg.fit(X_train, y_train)

#make a pickle file of the model
pickle.dump(linreg, open("model.pkl","wb"))