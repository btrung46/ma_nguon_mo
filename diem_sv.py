import pandas as pd
from numpy import array
import matplotlib.pyplot as plt
import numpy as np
from tkinter import Tk, Frame, Button, Label
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load data from CSV
df = pd.read_csv('D:\doc_ma_nguoi_mo\mnm\diemPython.csv', index_col=0, header=0)
in_data = array(df.iloc[:, :])

tongsv = in_data[:, 1]
diemA1 = in_data[:, 3]
diemB1 = in_data[:, 5]
diemC1 = in_data[:, 7]

diemA2 = in_data[:, 11]
diemB2 = in_data[:, 12]

diemA3 = in_data[:, 13]
diemB3 = in_data[:, 14]

index = np.arange(len(diemA1))
width = 0.30


# Function to create the first chart (Grades A, B, C)
def plot_grade_chart():
    fig, ax = plt.subplots()
    ax.bar(index, diemA1, width, color='red', label="Diem A")
    ax.bar(index + width, diemB1, width, color='green', label="Diem B")
    ax.bar(index + 2 * width, diemC1, width, color='yellow', label="Diem C")
    ax.set_xlabel('Lớp')
    ax.set_ylabel('Số sinh viên đạt điểm')
    ax.legend(loc='upper right')
    ax.set_xticks(index)
    display_chart(fig)


# Function to create the second chart (Chuẩn L1, L2)
def plot_standard_chart():
    fig, ax = plt.subplots()
    ax.bar(index, diemA2, width, color='blue', label="Chuẩn L1")
    ax.bar(index + width, diemB2, width, color='gray', label="Chuẩn L2")
    ax.set_xlabel('Lớp')
    ax.set_ylabel('Số sinh viên đạt chuẩn')
    ax.legend(loc='upper right')
    ax.set_xticks(index)
    display_chart(fig)


# Function to create the third chart (TX1, TX2)
def plot_tx_chart():
    fig, ax = plt.subplots()
    ax.bar(index, diemA3, width, color='black', label="TX1")
    ax.bar(index + width, diemB3, width, color='brown', label="TX2")
    ax.set_xlabel('Lớp')
    ax.set_ylabel('Số sinh viên')
    ax.legend(loc='upper right')
    ax.set_xticks(index)
    display_chart(fig)


# Function to display a given chart on the canvas
def display_chart(fig):
    for widget in frame.winfo_children():
        widget.destroy()

    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)


# Create the Tkinter window
root = Tk()
root.title("Biểu đồ kết quả sinh viên")
root.geometry("800x600")

# Create a frame to hold the Matplotlib plot
frame = Frame(root)
frame.pack(fill="both", expand=True)

# Create a frame for buttons
button_frame = Frame(root)
button_frame.pack(fill="x")

# Create buttons to switch between charts
btn_grades = Button(button_frame, text="Hiển thị Biểu đồ Diem A, B, C", command=plot_grade_chart)
btn_grades.pack(side="left", padx=10, pady=10)

btn_standard = Button(button_frame, text="Hiển thị Biểu đồ Chuẩn L1, L2", command=plot_standard_chart)
btn_standard.pack(side="left", padx=10, pady=10)

btn_tx = Button(button_frame, text="Hiển thị Biểu đồ TX1, TX2", command=plot_tx_chart)
btn_tx.pack(side="left", padx=10, pady=10)

# Start the Tkinter event loop
root.mainloop()
