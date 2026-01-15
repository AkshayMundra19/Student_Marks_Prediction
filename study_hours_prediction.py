#import necessary Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Dataset
data = {
    "Hours": [1,2,3,4,5,6,7,8,9,10],
    "Marks": [20,25,35,45,50,60,65,75,85,95]
}

df = pd.DataFrame(data)
print(df)

# Plot data
plt.scatter(df["Hours"], df["Marks"])
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks")
plt.savefig('study_hours_vs_marks.png')

# Split data
X = df[["Hours"]]
y = df["Marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Test prediction
y_pred = model.predict(X_test)
print("Predicted Marks:", y_pred)

# Accuracy
print("Model Accuracy (R2 Score):", r2_score(y_test, y_pred))

# Example prediction
hours = 5  # Example study hours
result = model.predict([[hours]])
print("Predicted Marks for 5 hours:", round(result[0], 2))

# Multiple prediction example
sample_hours = [[2], [4], [6], [8]]
sample_marks = model.predict(sample_hours)

for h, m in zip(sample_hours, sample_marks):
    print(f"{h[0]} hours -> {round(m,2)} marks")

print("Prediction completed successfully")
