import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image
import os

class MainWindow:
    def __init__(self, master):
        self.master = master
        master.title("气体动力参数计算 v1.0")
        master.geometry("800x600")#该数值由背景图大小确定，如后续需调整另行修改

        # 加载背景图片
        try:
            img_path = os.path.join(os.path.dirname(__file__), "background.jpg")
            self.bg_image = ImageTk.PhotoImage(Image.open("background.jpg"))
        except Exception as e:
            print("背景图片加载失败:", e)
            self.bg_image = None

        # 创建画布放置背景
        self.canvas = tk.Canvas(master, width=800, height=600)#该数值由背景图大小确定，如后续需调整另行修改
        self.canvas.pack(fill="both", expand=True)
        if self.bg_image:
            self.canvas.create_image(0, 0, image=self.bg_image, anchor="nw")

        # 创建功能按钮
        button_style = {'font': ('微软雅黑', 14), 'width': 20, 'height': 2}
        y_pos = 150
        btn1 = tk.Button(master, text="1. 滞止参数与气动函数", 
                        command=self.open_stagnation, **button_style)
        btn2 = tk.Button(master, text="2. 膨胀波计算", 
                        command=self.open_expansion, **button_style)
        btn3 = tk.Button(master, text="3. 激波计算", 
                        command=self.open_shockwave, **button_style)
        btn4 = tk.Button(master, text="4. 一维定常管内流动", 
                        command=self.open_flow, **button_style)

        # 将按钮添加到画布上
        self.canvas.create_window(400, y_pos, window=btn1)
        self.canvas.create_window(400, y_pos+100, window=btn2)
        self.canvas.create_window(400, y_pos+200, window=btn3)
        self.canvas.create_window(400, y_pos+300, window=btn4)

    def open_stagnation(self):
        StagnationWindow(tk.Toplevel(self.master))

    def open_expansion(self):
        # 类似实现其他模块
        pass

    def open_shockwave(self):
        pass

    def open_flow(self):
        pass

class StagnationWindow:
    def __init__(self, master):
        self.master = master
        master.title("滞止参数计算")
        master.geometry("500x400")

        # 输入参数框架
        input_frame = ttk.LabelFrame(master, text="输入参数")
        input_frame.pack(pady=10, padx=20, fill="x")

        ttk.Label(input_frame, text="马赫数 Ma:").grid(row=0, column=0, padx=5, pady=5)
        self.entry_ma = ttk.Entry(input_frame)
        self.entry_ma.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="比热比 γ:").grid(row=1, column=0, padx=5, pady=5)
        self.entry_gamma = ttk.Entry(input_frame)
        self.entry_gamma.insert(0, "1.4")  # 默认值
        self.entry_gamma.grid(row=1, column=1, padx=5, pady=5)

        # 计算按钮
        btn_calc = ttk.Button(master, text="计算", command=self.calculate)
        btn_calc.pack(pady=10)

        # 结果显示框架
        result_frame = ttk.LabelFrame(master, text="计算结果")
        result_frame.pack(pady=10, padx=20, fill="x")

        self.labels = {
            'T_ratio': ttk.Label(result_frame, text="总温比 T0/T = "),
            'P_ratio': ttk.Label(result_frame, text="总压比 P0/P = "),
            'rho_ratio': ttk.Label(result_frame, text="总密比 ρ0/ρ = ")
        }
        for i, (_, label) in enumerate(self.labels.items()):
            label.grid(row=i, column=0, sticky="w", padx=5, pady=2)

    def calculate(self):
        try:
            Ma = float(self.entry_ma.get())
            gamma = float(self.entry_gamma.get())
            
            # 计算气动函数
            T_ratio = 1 + (gamma-1)/2 * Ma**2
            P_ratio = T_ratio**(gamma/(gamma-1))
            rho_ratio = P_ratio**(1/gamma)
            
            # 更新结果
            self.labels['T_ratio'].config(text=f"总温比 T0/T = {T_ratio:.4f}")
            self.labels['P_ratio'].config(text=f"总压比 P0/P = {P_ratio:.4f}")
            self.labels['rho_ratio'].config(text=f"总密比 ρ0/ρ = {rho_ratio:.4f}")
            
        except ValueError:
            tk.messagebox.showerror("输入错误", "请输入有效的数字参数")

if __name__ == "__main__":
    root = tk.Tk()
    app = MainWindow(root)
    root.mainloop()