import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


data = pd.read_csv("air_quality.csv")


data = data[['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3', 'AQI_Bucket']]


data = data.dropna()


X = data[['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']]
y = data['AQI_Bucket']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


model = DecisionTreeClassifier()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)


print("\nSample Predictions:")
for i in range(5):
    print("Predicted:", y_pred[i], "| Actual:", y_test.iloc[i])


data['AQI_Bucket'].value_counts().plot(kind='bar')
plt.title("Air Quality Category Distribution")
plt.xlabel("Category")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

data[['PM2.5', 'PM10', 'NO2']].head(50).plot()
plt.title("Pollution Trends (First 50 Records)")
plt.xlabel("Index")
plt.ylabel("Pollution Level")
plt.show()

data.groupby('AQI_Bucket')[['PM2.5','PM10','NO2']].mean().plot(kind='bar')
plt.title("Average Pollution per AQI Category")
plt.show()


print("\n--- Try Your Own Input ---")
pm25 = float(input("Enter PM2.5: "))
pm10 = float(input("Enter PM10: "))
no2 = float(input("Enter NO2: "))
co = float(input("Enter CO: "))
so2 = float(input("Enter SO2: "))
o3 = float(input("Enter O3: "))

user_data = [[pm25, pm10, no2, co, so2, o3]]
result = model.predict(user_data)

print("\nPredicted Air Quality Category:", result[0])