import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
from numpy import array
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt




    def load_csv(self):
        self.filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.filepath:
            self.file_label.config(text=self.filepath)

    def train_model(self):
        if self.filepath:
            df = pd.read_csv(self.filepath)
            x = array(df.iloc[:200, 0:5]).astype(np.float64)
            y = array(df.iloc[:200, 4:5]).astype(np.float64)
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

            if self.algo_var.get() == "KNN":
                self.knn = neighbors.KNeighborsRegressor(n_neighbors=3, p=2)
                self.knn.fit(X_train, y_train)
                y_predict = self.knn.predict(X_test)

                # Calculate errors
                mse = mean_squared_error(y_test, y_predict)
                mae = mean_absolute_error(y_test, y_predict)
                rmse = np.sqrt(mse)

                print(f"MSE: {mse}, MAE: {mae}, RMSE: {rmse}")

                # Plot
                plt.plot(range(0, len(y_test)), y_test, 'ro', label='Original data')
                plt.plot(range(0, len(y_predict)), y_predict, 'bo', label='Fitted line')
                for i in range(0, len(y_test)):
                    plt.plot([i, i], [y_test[i], y_predict[i]], 'green')
                plt.title('KNN Result')
                plt.legend()
                plt.show()

    def predict_score(self):
        if hasattr(self, 'knn'):
            # Get input values from entry fields
            age = float(self.age_entry.get())
            study_hours = float(self.study_hours_entry.get())
            prev_grade = float(self.prev_grade_entry.get())
            attendance = float(self.attendance_entry.get())
            participation = float(self.participation_entry.get())

            input_data = np.array([[age, study_hours, prev_grade, attendance, participation]]).astype(np.float64)
            prediction = self.knn.predict(input_data)
            self.result_label.config(text=f"Prediction Result: {prediction[0][0]:.2f}")

            # Get real score (if available)
            real_score = self.real_entry.get()
            if real_score:
                real_score = float(real_score)
                self.plot_comparison(real_score, prediction[0][0])

    def plot_comparison(self, real_score, predicted_score):
        # Data for bar chart
        labels = ['Actual Score', 'Predicted Score']
        values = [real_score, predicted_score]

        # Create bar chart
        plt.bar(labels, values, color=['red', 'blue'])
        plt.ylabel('Scores')
        plt.title('Actual vs Predicted Score')

        # Show the chart
        plt.show()


import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
from numpy import array
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class StudentScorePredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("Student Score Predictor")
        self.filepath = ''
        self.models = {
            'KNN': neighbors.KNeighborsRegressor(n_neighbors=3),
            'Hồi quy tuyến tính': LinearRegression(),
            'Cây quyết định': DecisionTreeRegressor(),
            'Vector hỗ trợ': SVR()
        }
        self.model_errors = {}

        # Frame for CSV selection
        self.frame1 = tk.Frame(self.root)
        self.frame1.pack(pady=10)

        self.label = tk.Label(self.frame1, text="Choose CSV file:")
        self.label.pack(side=tk.LEFT)

        self.csv_button = tk.Button(self.frame1, text="Browse", command=self.load_csv)
        self.csv_button.pack(side=tk.LEFT)

        self.file_label = tk.Label(self.frame1, text="No file selected")
        self.file_label.pack(side=tk.LEFT)

        # Frame for algorithm selection
        self.frame2 = tk.Frame(self.root)
        self.frame2.pack(pady=10)

        self.algo_label = tk.Label(self.frame2, text="Choose Algorithm:")
        self.algo_label.pack(side=tk.LEFT)

        self.algo_var = tk.StringVar(value="KNN")
        self.algo_menu = tk.OptionMenu(self.frame2, self.algo_var, *self.models.keys())
        self.algo_menu.pack(side=tk.LEFT)

        # Button to train model
        self.train_button = tk.Button(self.root, text="Train Model", command=self.train_model)
        self.train_button.pack(pady=10)

        # Frame for input data
        self.frame3 = tk.Frame(self.root)
        self.frame3.pack(pady=10)

        # Labels and entry fields for student data input
        self.age_label = tk.Label(self.frame3, text="Age:")
        self.age_label.grid(row=0, column=0)
        self.age_entry = tk.Entry(self.frame3)
        self.age_entry.grid(row=0, column=1)

        self.study_hours_label = tk.Label(self.frame3, text="Study Hours:")
        self.study_hours_label.grid(row=1, column=0)
        self.study_hours_entry = tk.Entry(self.frame3)
        self.study_hours_entry.grid(row=1, column=1)

        self.prev_grade_label = tk.Label(self.frame3, text="Previous Grade:")
        self.prev_grade_label.grid(row=2, column=0)
        self.prev_grade_entry = tk.Entry(self.frame3)
        self.prev_grade_entry.grid(row=2, column=1)

        self.attendance_label = tk.Label(self.frame3, text="Attendance (%):")
        self.attendance_label.grid(row=3, column=0)
        self.attendance_entry = tk.Entry(self.frame3)
        self.attendance_entry.grid(row=3, column=1)

        self.participation_label = tk.Label(self.frame3, text="Participation (%):")
        self.participation_label.grid(row=4, column=0)
        self.participation_entry = tk.Entry(self.frame3)
        self.participation_entry.grid(row=4, column=1)

        # Button to predict
        self.predict_button = tk.Button(self.root, text="Predict", command=self.predict_score)
        self.predict_button.pack(pady=10)

        # Label for showing result
        self.result_label = tk.Label(self.root, text="Prediction Result: ")
        self.result_label.pack(pady=10)

        # Frame for real score input (optional)
        self.frame4 = tk.Frame(self.root)
        self.frame4.pack(pady=10)

        self.real_label = tk.Label(self.frame4, text="Enter actual score (if available):")
        self.real_label.grid(row=0, column=0)
        self.real_entry = tk.Entry(self.frame4)
        self.real_entry.grid(row=0, column=1)

    def load_csv(self):
        self.filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.filepath:
            self.file_label.config(text=self.filepath)

    def train_model(self):
        if self.filepath:
            df = pd.read_csv(self.filepath)
            x = array(df.iloc[:, 0:5]).astype(np.float64)  # Assuming the first 5 columns are features
            y = array(df.iloc[:, 5]).astype(np.float64)  # Assuming the 6th column is the target score

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

            # Standardize the data
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Train all models and calculate errors
            for name, model in self.models.items():
                model.fit(X_train, y_train)
                y_predict = model.predict(X_test)

                # Calculate errors
                mse = mean_squared_error(y_test, y_predict)
                mae = mean_absolute_error(y_test, y_predict)
                rmse = np.sqrt(mse)

                self.model_errors[name] = {'MSE': mse, 'MAE': mae, 'RMSE': rmse}

                print(f"{name} - MSE: {mse}, MAE: {mae}, RMSE: {rmse}")

            # Plot error comparison
            self.plot_error_comparison()

    def plot_error_comparison(self):
        labels = list(self.model_errors.keys())
        mse_values = [errors['MSE'] for errors in self.model_errors.values()]
        mae_values = [errors['MAE'] for errors in self.model_errors.values()]
        rmse_values = [errors['RMSE'] for errors in self.model_errors.values()]

        x = np.arange(len(labels))  # the label locations

        fig, ax = plt.subplots()
        bar_width = 0.25
        ax.bar(x - bar_width, mse_values, width=bar_width, label='MSE')
        ax.bar(x, mae_values, width=bar_width, label='MAE')
        ax.bar(x + bar_width, rmse_values, width=bar_width, label='RMSE')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_xlabel('Models')
        ax.set_ylabel('Error Values')
        ax.set_title('Error Comparison of Different Algorithms')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        plt.tight_layout()
        plt.show()

    def predict_score(self):
        if self.algo_var.get() in self.models:
            # Get input values from entry fields
            age = float(self.age_entry.get())
            study_hours = float(self.study_hours_entry.get())
            prev_grade = float(self.prev_grade_entry.get())
            attendance = float(self.attendance_entry.get())
            participation = float(self.participation_entry.get())

            input_data = np.array([[age, study_hours, prev_grade, attendance, participation]]).astype(np.float64)

            # Standardize input data
            scaler = StandardScaler()
            input_data = scaler.fit_transform(input_data)

            model = self.models[self.algo_var.get()]
            prediction = model.predict(input_data)
            self.result_label.config(text=f"Prediction Result: {prediction[0]:.2f}")

            # Get real score (if available)
            real_score = self.real_entry.get()
            if real_score:
                real_score = float(real_score)
                self.plot_comparison(real_score, prediction[0])

    def plot_comparison(self, real_score, predicted_score):
        # Data for bar chart
        labels = ['Actual Score', 'Predicted Score']
        values = [real_score, predicted_score]

        # Create bar chart
        plt.bar(labels, values, color=['red', 'blue'])
        plt.ylabel('Scores')
        plt.title('Actual vs Predicted Score')

        # Show the chart
        plt.show()


# Create the main window
root = tk.Tk()
app = StudentScorePredictor(root)
root.mainloop()

# Create the main window
root = tk.Tk()
app = StudentScorePredictor(root)
root.mainloop()
