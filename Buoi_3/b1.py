import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox

# Khởi tạo biến x cho các phép toán
x = sp.symbols('x')


def tinh_dao_ham(ham_so):
    # Tính đạo hàm
    dao_ham = sp.diff(ham_so, x)
    return dao_ham


def tinh_tich_phan(ham_so, a=None, b=None):
    # Nếu a và b có giá trị, tính tích phân xác định
    if a is not None and b is not None:
        tich_phan = sp.integrate(ham_so, (x, a, b))
    else:
        # Tính tích phân không xác định
        tich_phan = sp.integrate(ham_so, x)
    return tich_phan


def ve_do_thi(ham_so, khoang=(-10, 10)):
    # Chuyển hàm sympy thành hàm numpy để vẽ đồ thị
    ham_so_numpy = sp.lambdify(x, ham_so, 'numpy')

    # Tạo mảng giá trị cho x trong khoảng (khoang[0], khoang[1])
    x_vals = np.linspace(khoang[0], khoang[1], 400)
    y_vals = ham_so_numpy(x_vals)

    # Vẽ đồ thị
    plt.plot(x_vals, y_vals, label=f'Đồ thị của {ham_so}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Đồ thị hàm số')
    plt.grid(True)
    plt.legend()
    plt.show()


def tinh_cuc_tri(ham_so):
    # Tính đạo hàm bậc nhất
    dao_ham_1 = tinh_dao_ham(ham_so)

    # Giải phương trình đạo hàm bậc nhất = 0 để tìm điểm dừng
    diem_dung = sp.solve(dao_ham_1, x)

    cuc_dai = []
    cuc_tieu = []

    # Xét đạo hàm bậc hai để phân loại cực trị
    dao_ham_2 = sp.diff(dao_ham_1, x)

    for diem in diem_dung:
        gia_tri_dao_ham_2 = dao_ham_2.subs(x, diem)

        if gia_tri_dao_ham_2 > 0:
            cuc_tieu.append(diem)
        elif gia_tri_dao_ham_2 < 0:
            cuc_dai.append(diem)

    return cuc_dai, cuc_tieu


# Hàm để xử lý khi nhấn nút "Tính toán"
def xu_ly_nhap():
    try:
        ham_input = ham_so_entry.get()  # Lấy hàm số từ entry
        ham_so = sp.sympify(ham_input)  # Chuyển đổi sang sympy biểu thức

        # Tính đạo hàm
        dao_ham = tinh_dao_ham(ham_so)

        # Tính tích phân
        tich_phan = tinh_tich_phan(ham_so)

        # Tính cực trị
        cuc_dai, cuc_tieu = tinh_cuc_tri(ham_so)

        # Hiển thị kết quả
        ket_qua = f"Đạo hàm: {dao_ham}\n"
        ket_qua += f"Tích phân không xác định: {tich_phan}\n"
        ket_qua += f"Cực đại tại: {cuc_dai}\n"
        ket_qua += f"Cực tiểu tại: {cuc_tieu}\n"
        ket_qua_text.insert(tk.END, ket_qua)  # Hiển thị trong Text box

        # Vẽ đồ thị
        ve_do_thi(ham_so)

    except Exception as e:
        messagebox.showerror("Lỗi", f"Đã xảy ra lỗi: {e}")


# Giao diện Tkinter
root = tk.Tk()
root.title("Ứng dụng hỗ trợ học giải tích")

# Label và Entry để nhập hàm số
label = tk.Label(root, text="Nhập hàm số:")
label.pack(pady=10)

ham_so_entry = tk.Entry(root, width=50)
ham_so_entry.pack(pady=10)

# Nút tính toán
tinh_button = tk.Button(root, text="Tính toán", command=xu_ly_nhap)
tinh_button.pack(pady=10)

# Text box để hiển thị kết quả
ket_qua_text = tk.Text(root, height=10, width=60)
ket_qua_text.pack(pady=10)

# Chạy giao diện
root.mainloop()
