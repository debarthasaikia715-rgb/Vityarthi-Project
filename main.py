# student_predictor.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

# ===== MODULE 1: Data Preparation =====
print("=== Student Performance Predictor ===")
print("Loading and preparing data...")

# Create simple sample data
data = {
    'study_hours': [2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 4, 5, 6, 7, 8],
    'attendance': [70, 75, 80, 85, 90, 65, 80, 85, 90, 95, 75, 85, 90, 95, 98],
    'midterm_score': [60, 65, 70, 75, 80, 55, 68, 72, 78, 85, 62, 70, 76, 82, 88],
    'final_score': [65, 70, 75, 80, 85, 58, 72, 78, 82, 88, 68, 76, 80, 86, 92]
}

df = pd.DataFrame(data)
X = df[['study_hours', 'attendance', 'midterm_score']]
y = df['final_score']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===== MODULE 2: Model Training =====
print("Training models...")

# Train Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Train Decision Tree
dt_model = DecisionTreeRegressor(max_depth=3, random_state=42)
dt_model.fit(X_train, y_train)

# Evaluate models
lr_pred = lr_model.predict(X_test)
dt_pred = dt_model.predict(X_test)

lr_error = mean_absolute_error(y_test, lr_pred)
dt_error = mean_absolute_error(y_test, dt_pred)

print(f"Linear Regression Error: {lr_error:.2f}")
print(f"Decision Tree Error: {dt_error:.2f}")

# ===== MODULE 3: Prediction Interface =====
print("\n--- Make a Prediction ---")
print("Enter student details:")

try:
    study_hrs = float(input("Study hours per day: "))
    attendance = float(input("Attendance percentage: "))
    midterm = float(input("Mid-term score: "))
    
    # Create input array
    student_data = np.array([[study_hrs, attendance, midterm]])
    
    # Make predictions
    lr_prediction = lr_model.predict(student_data)[0]
    dt_prediction = dt_model.predict(student_data)[0]
    
    print(f"\n--- Prediction Results ---")
    print(f"Linear Regression prediction: {lr_prediction:.1f}%")
    print(f"Decision Tree prediction: {dt_prediction:.1f}%")
    
    # Show which model is better
    if lr_error < dt_error:
        print(f"Recommended model: Linear Regression (More Accurate)")
    else:
        print(f"Recommended model: Decision Tree (More Accurate)")
        
except ValueError:
    print("Please enter valid numbers!")