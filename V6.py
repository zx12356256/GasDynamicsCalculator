import math
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import os
import sys
import pygame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import mpmath as mp
from scipy.optimize import brentq

# 设置matplotlib后端和字体
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
# 如果需要使用微软雅黑，可以将上面一行替换为：
# rcParams['font.sans-serif'] = ['Microsoft YaHei']

rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

def resource_path(relative_path):
    """ 获取资源的绝对路径 """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


class VideoCanvas(tk.Canvas):
    def __init__(self, master=None, video_source="background.mp4", audio_source="background_audio.mp3", width=1280,
                 height=720):
        super().__init__(master, width=width, height=height)
        self.width = width
        self.height = height
        self.photo = None

        # 处理资源路径
        self.video_source = resource_path(video_source)
        self.audio_source = resource_path(audio_source)

        if not os.path.exists(self.video_source):
            raise FileNotFoundError(f"未找到视频文件: {self.video_source}")
        if not os.path.exists(self.audio_source):
            raise FileNotFoundError(
                f"未找到音频文件: {self.audio_source}. 如果你需要从视频中提取音频，请使用 moviepy 或其他工具.")

        # 初始化视频流
        self.vid = cv2.VideoCapture(self.video_source)
        if not self.vid.isOpened():
            raise IOError(f"无法打开视频文件: {self.video_source}")

        # 初始化音频播放器
        pygame.mixer.init()
        pygame.mixer.music.load(self.audio_source)
        pygame.mixer.music.play(-1)  # 循环播放

        # 开始播放视频
        self.play_video()

    def play_video(self):
        ret, frame = self.vid.read()
        if not ret:
            # 循环播放
            self.vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.vid.read()

        if ret:
            frame = cv2.resize(frame, (self.winfo_width(), self.winfo_height()))
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            self.photo = ImageTk.PhotoImage(image=img)
            self.create_image(0, 0, image=self.photo, anchor='nw')

        self.after(15, self.play_video)  # 约60 FPS

    def stop(self):
        pygame.mixer.music.stop()
        self.vid.release()


class MainWindow:
    def __init__(self, master):
        self.master = master
        master.title("气体动力参数计算")
        master.geometry("1280x720")  # 设置窗口大小
        master.resizable(False, False)  # 禁止调整窗口大小

        # 居中显示主窗口
        screen_width = master.winfo_screenwidth()
        screen_height = master.winfo_screenheight()
        x_position = int((screen_width / 2) - (1280 / 2))
        y_position = int((screen_height / 2) - (720 / 2))
        master.geometry(f"1280x720+{x_position}+{y_position}")

        # 创建画布并设置视频背景（含音频）
        self.canvas = VideoCanvas(master, video_source="background.mp4", audio_source="background_audio.mp3",
                                  width=1280, height=720)
        self.canvas.pack(fill="both", expand=True)

        # 创建功能按钮和静音按钮
        button_style = {'font': ('微软雅黑', 14), 'width': 20, 'height': 2}
        btn1 = tk.Button(master, text="1. 滞止参数与气动函数", command=self.open_stagnation, **button_style)
        btn2 = tk.Button(master, text="2. 膨胀波计算", command=self.open_expansion, **button_style)
        btn3 = tk.Button(master, text="3. 激波计算", command=self.open_combined_shockwave,
                         **button_style)  # 更新命令为 open_combined_shockwav
        btn4 = tk.Button(master, text="4. 一维定常管内流动", command=self.open_flow, **button_style)

        # 添加静音按钮
        self.mute_button = tk.Button(master, text="静音", command=self.toggle_mute, **{'font': ('微软雅黑', 12)})
        self.is_muted = False

        # 设置按钮位置（田字形）
        btn_width = 200  # 根据按钮宽度调整
        btn_height = 60
        center_x = 1280 // 2
        center_y = 720 // 2

        # 放置普通功能按钮
        self.canvas.create_window(center_x + btn_width - 40, center_y + btn_height + 80, window=btn1)
        self.canvas.create_window(center_x + btn_width + 250, center_y + btn_height + 80, window=btn2)
        self.canvas.create_window(center_x + btn_width - 40, center_y + btn_height + 180, window=btn3)
        self.canvas.create_window(center_x + btn_width + 250, center_y + btn_height + 180, window=btn4)

        # 放置静音按钮于右上角
        self.mute_button.place(x=1280 - 100, y=20)  # 假设按钮宽度大约为100px

    def toggle_mute(self):
        """ 切换静音状态 """
        if self.is_muted:
            pygame.mixer.music.unpause()
            self.mute_button.config(text="静音")
        else:
            pygame.mixer.music.pause()
            self.mute_button.config(text="取消静音")
        self.is_muted = not self.is_muted

    def open_stagnation(self):
        stagnation_window = StagnationWindow(tk.Toplevel(self.master))
        stagnation_window.master.geometry("650x600+215+205")  

    def open_expansion(self):
        expansion_window = ExpansionWindow(tk.Toplevel(self.master))
        expansion_window.master.geometry("650x500+215+205")  

    def open_combined_shockwave(self):
        combined_shockwave_window = CombinedShockwaveWindow(tk.Toplevel(self.master))
        combined_shockwave_window.master.geometry("650x500+200+200")  # 设置初始位置和大小

    def open_flow(self):
        flowWindow_window = FlowWindow(tk.Toplevel(self.master))
        flowWindow_window.master.geometry("650x500+200+200")


class StagnationWindow:
    def __init__(self, master):
        self.master = master
        master.title("气动函数计算器")
        master.geometry("650x600")
        
        # 创建选项卡
        self.notebook = ttk.Notebook(master)
        
        # 滞止参数计算标签页
        self.stagnation_frame = ttk.Frame(self.notebook)
        self.create_stagnation_tab()
        
        # 临界参数计算标签页
        self.critical_frame = ttk.Frame(self.notebook)
        self.create_critical_tab()
        
        # 马赫数-速度系数转换标签页
        self.conversion_frame = ttk.Frame(self.notebook)
        self.create_conversion_tab()
        
        # 滞止参数计算(已知静参数)标签页
        self.static_frame = ttk.Frame(self.notebook)
        self.create_static_tab()
        
        # 总参数计算(已知静参数)标签页
        self.total_frame = ttk.Frame(self.notebook)
        self.create_total_tab()

        self.notebook.add(self.stagnation_frame, text="滞止参数(绝能等熵)")
        self.notebook.add(self.critical_frame, text="临界参数")
        self.notebook.add(self.conversion_frame, text="Ma-λ转换及气动函数")
        self.notebook.add(self.static_frame, text="总参数")
        self.notebook.add(self.total_frame, text="静参数")
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)
        
        # 设置默认值
        self.gamma = 1.4
    
    def create_stagnation_tab(self):
        """创建滞止参数计算标签页"""
        frame = self.stagnation_frame
        
        # 输入参数框架
        input_frame = ttk.LabelFrame(frame, text="输入参数")
        input_frame.pack(pady=10, padx=10, fill="x")
        
        ttk.Label(input_frame, text="马赫数 Ma:").grid(row=0, column=0, padx=5, pady=5)
        self.entry_ma1 = ttk.Entry(input_frame)
        self.entry_ma1.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="比热比 γ:").grid(row=1, column=0, padx=5, pady=5)
        self.entry_gamma1 = ttk.Entry(input_frame)
        self.entry_gamma1.insert(0, "1.4")
        self.entry_gamma1.grid(row=1, column=1, padx=5, pady=5)
        
        # 计算按钮
        btn_calc = ttk.Button(frame, text="计算滞止参数", command=self.calculate_stagnation)
        btn_calc.pack(pady=10)
        
        # 结果显示框架
        result_frame = ttk.LabelFrame(frame, text="计算结果")
        result_frame.pack(pady=10, padx=10, fill="x")
        
        self.stagnation_labels = {
            'T_ratio': ttk.Label(result_frame, text="总温比 T*/T = "),
            'P_ratio': ttk.Label(result_frame, text="总压比 P*/P = "),
            'rho_ratio': ttk.Label(result_frame, text="总密比 ρ*/ρ = "),
            'a_ratio': ttk.Label(result_frame, text="声速比 c*/c = "),
            
        }
        
        for i, (_, label) in enumerate(self.stagnation_labels.items()):
            label.grid(row=i, column=0, sticky="w", padx=5, pady=2)

    def calculate_stagnation(self):
        """计算滞止参数"""
        try:
            Ma = float(self.entry_ma1.get())
            gamma = float(self.entry_gamma1.get())
            
            # 计算滞止参数
            T_ratio = 1 + (gamma - 1) / 2 * Ma ** 2
            P_ratio = T_ratio ** (gamma / (gamma - 1))
            rho_ratio = P_ratio ** (1 / gamma)
            a_ratio = math.sqrt(T_ratio)
           
            
            # 更新结果
            self.stagnation_labels['T_ratio'].config(text=f"总温比 T*/T = {T_ratio:.4f}")
            self.stagnation_labels['P_ratio'].config(text=f"总压比 P*/P = {P_ratio:.4f}")
            self.stagnation_labels['rho_ratio'].config(text=f"总密比 ρ*/ρ = {rho_ratio:.4f}")
            self.stagnation_labels['a_ratio'].config(text=f"声速比 c*/c = {a_ratio:.4f}")
            
            
        except ValueError:
            messagebox.showerror("输入错误", "请输入有效的数字参数")
    def create_critical_tab(self):
        """创建临界参数计算标签页"""
        frame = self.critical_frame
        
        # 输入参数框架
        input_frame = ttk.LabelFrame(frame, text="输入参数")
        input_frame.pack(pady=10, padx=10, fill="x")
        
        
        ttk.Label(input_frame, text="比热比 γ:").grid(row=0, column=0, padx=5, pady=5)
        self.entry_gamma2 = ttk.Entry(input_frame)
        self.entry_gamma2.insert(0, "1.4")
        self.entry_gamma2.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="总温 T*(K):").grid(row=1, column=0, padx=5, pady=5)
        self.entry_Tz = ttk.Entry(input_frame)
        self.entry_Tz.grid(row=1, column=1, padx=5, pady=5)
       
        ttk.Label(input_frame, text="气体常数 R (J/kg·K):").grid(row=2, column=0, padx=5, pady=5)
        self.entry_R2 = ttk.Entry(input_frame)
        self.entry_R2.insert(0, "287")
        self.entry_R2.grid(row=2, column=1, padx=5, pady=5)
        # 计算按钮
        btn_calc = ttk.Button(frame, text="计算", command=self.calculate_critical)
        btn_calc.pack(pady=10)
        
        # 结果显示框架
        result_frame = ttk.LabelFrame(frame, text="临界参数")
        result_frame.pack(pady=10, padx=10, fill="x")
        
        self.critical_labels = {
            'T_star2': ttk.Label(result_frame, text="临界温度比 Tcr/T* = "),
            'P_star2': ttk.Label(result_frame, text="临界压力比 Pcr/P* = "),
            'rho_star2': ttk.Label(result_frame, text="临界密度比 ρcr/ρ* = "),
            'c_star2': ttk.Label(result_frame, text="临界声速 Ccr = "),
            'v_star2': ttk.Label(result_frame, text="极限速度 Vm = ")
            
        }
        
        for i, (_, label) in enumerate(self.critical_labels.items()):
            label.grid(row=i, column=0, sticky="w", padx=5, pady=2)

    def calculate_critical(self):
        """计算临界参数"""
        try:
            gamma = float(self.entry_gamma2.get())
            Tz = float(self.entry_Tz.get())
            R = float(self.entry_R2.get())
            # 计算临界参数
            T_star = 2/(gamma+1)
            P_star = (2/(gamma+1))**(gamma/(gamma-1))
            rho_star = (2/(gamma+1))**(1/(gamma-1))
            c_star = math.sqrt(2*gamma*Tz*R/(gamma+1))
            v_star = math.sqrt(2*gamma*Tz*R/(gamma-1))
        
            
            # 更新结果
            self.critical_labels['T_star2'].config(text=f"临界温度比 Tcr/T = {T_star:.4f}")
            self.critical_labels['P_star2'].config(text=f"临界压力比 Pcr/P = {P_star:.4f}")
            self.critical_labels['rho_star2'].config(text=f"临界密度比 ρcr/ρ = {rho_star:.4f}")
            self.critical_labels['c_star2'].config(text=f"临界声速 Ccr = {c_star:.4f}")
            self.critical_labels['v_star2'].config(text=f"极限速度 Vm = {v_star:.4f}")
            
        except ValueError:
            messagebox.showerror("输入错误", "请输入有效的数字参数")

    def create_conversion_tab(self):
        """创建马赫数-速度系数转换标签页"""
        frame = self.conversion_frame
        
        # 输入参数框架
        input_frame = ttk.LabelFrame(frame, text="输入参数")
        input_frame.pack(pady=10, padx=10, fill="x")
        
        # 选择输入类型
        self.conversion_var = tk.StringVar(value="Ma")
        ttk.Radiobutton(input_frame, text="输入马赫数 Ma", 
                    variable=self.conversion_var, value="Ma").grid(row=0, column=0, padx=5, pady=5)
        ttk.Radiobutton(input_frame, text="输入速度系数 λ", 
                    variable=self.conversion_var, value="lambda").grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="输入值:").grid(row=1, column=0, padx=5, pady=5)
        self.entry_input = ttk.Entry(input_frame)
        self.entry_input.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="比热比 γ:").grid(row=2, column=0, padx=5, pady=5)
        self.entry_gamma3 = ttk.Entry(input_frame)
        self.entry_gamma3.insert(0, "1.4")
        self.entry_gamma3.grid(row=2, column=1, padx=5, pady=5)
        
        # 计算按钮
        btn_calc = ttk.Button(frame, text="转换计算", 
                            command=self.calculate_conversion)
        btn_calc.pack(pady=10)
        
        # 添加显示图像的按钮
        button_frame = ttk.Frame(frame)
        button_frame.pack(pady=10)
        
        btn_show_graphs = ttk.Button(button_frame, text="显示气动函数图象", 
                                    command=self.show_graphs)
        btn_show_graphs.grid(row=0, column=0, padx=5)
        
        # 添加新的显示q、y、z图象的按钮
        btn_show_qyz = ttk.Button(button_frame, text="显示q、y、z-λ图象", 
                                command=self.show_qyz_graphs)
        btn_show_qyz.grid(row=0, column=1, padx=5)
        
        # 结果显示框架
        result_frame = ttk.LabelFrame(frame, text="转换结果")
        result_frame.pack(pady=10, padx=10, fill="x")
        
        self.conversion_labels = {
            'Ma': ttk.Label(result_frame, text="马赫数 Ma = "),
            'lambda': ttk.Label(result_frame, text="速度系数 λ = "),
            'T_ratio1': ttk.Label(result_frame, text="τ=T*/T = "),
            'P_ratio1': ttk.Label(result_frame, text="π=P*/P = "),
            'rho_ratio1': ttk.Label(result_frame, text="ε=ρ*/ρ = "),
            'q_r1': ttk.Label(result_frame, text="q = "),
            'y_r1': ttk.Label(result_frame, text="y = "),
            'z_r1': ttk.Label(result_frame, text="z = "),
            'f_r1': ttk.Label(result_frame, text="f = "),
            'r_r1': ttk.Label(result_frame, text="r = ")
        }
        
        for i, (_, label) in enumerate(self.conversion_labels.items()):
            label.grid(row=i, column=0, sticky="w", padx=5, pady=1)

 
    def show_qyz_graphs(self):
        """显示q、y、z随λ变化的图像（使用双坐标轴）"""
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            
            gamma = float(self.entry_gamma3.get())
            lambda_max = math.sqrt((gamma + 1) / (gamma - 1))
            
            # 创建λ值数组（从0.01到接近λ_max）
            lambda_values = np.linspace(0.01, lambda_max * 0.99, 400)
            #lambda_values = np.linspace(0.01, 4, 400)
            
            # 计算对应的Ma值
            Ma_values = lambda_values * np.sqrt(2 / ((gamma + 1) - (gamma - 1) * lambda_values**2))
            
            # 计算q、y、z值
            term1 = lambda_values
            term2 = ((gamma + 1) / 2) ** (1 / (gamma - 1))
            inner = 1 - (gamma - 1) / (gamma + 1) * (lambda_values**2)
            term3 = inner ** (1 / (gamma - 1))
            q_values = term1 * term2 * term3
            
            # 计算π值用于y的计算
            pi_values = (1 + (gamma - 1)/2 * Ma_values**2) ** (-gamma / (gamma - 1))
            y_values = q_values / pi_values
            
            z_values = 1/lambda_values + lambda_values
            
            # 创建图形
            fig = plt.figure(figsize=(14, 6))
            plt.suptitle(f"气动函数随速度系数λ的变化 (γ={gamma})", fontsize=16)
            
            # q和y随λ变化图（使用双纵坐标）
            ax1 = plt.subplot(121)  # 1行2列的第1个位置
            
            # 绘制q曲线（使用左侧坐标轴）
            color = 'tab:blue'
            ax1.set_xlabel('速度系数 λ')
            ax1.set_ylabel('q(λ)', color=color)
            ax1.plot(lambda_values, q_values, color=color, label='q(λ)')
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.grid(True)
            
            # 创建第二个坐标轴共享相同的x轴
            ax2 = ax1.twinx()  
            color = 'tab:red'
            ax2.set_ylabel('y(λ)', color=color)
            ax2.plot(lambda_values, y_values, color=color, label='y(λ)')
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.set_ylim(0, 5)  
            
            # 添加图例
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')
            
            ax1.set_title('q(λ)和y(λ)随速度系数变化')
            
            # z随λ变化图（扩大横轴范围）
            ax3 = plt.subplot(122)  # 1行2列的第2个位置
            
            # 创建更宽的λ范围（从0到λ_max）
            extended_lambda = np.linspace(0.3, lambda_max * 0.99, 400)
            #extended_lambda = np.linspace(0.3, 4, 400)
            extended_z = 1/extended_lambda + extended_lambda
            
            ax3.plot(extended_lambda, extended_z, 'g-', label='z(λ)')
            ax3.set_xlabel('速度系数 λ')
            ax3.set_ylabel('z(λ)')
            ax3.set_title('z(λ)随速度系数变化')
            ax3.legend()
            ax3.grid(True)
            
            # 添加垂直线标记λ=1处的最小值
            ax3.axvline(x=1, color='r', linestyle='--', alpha=0.5)
            ax3.annotate(f'最小值 z(1)={2:.2f}', xy=(1, 2), xytext=(1.1, 3),
                        arrowprops=dict(facecolor='black', arrowstyle='->'))
            
            # 设置横轴范围到接近λ_max
            ax3.set_xlim(0, lambda_max * 1.05)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.85)  # 为总标题留出空间
            
            # 在新窗口中显示图像
            graph_window = tk.Toplevel(self.master)
            graph_window.title("q、y、z随λ变化图像")
            graph_window.geometry("1200x600")
            
            canvas = FigureCanvasTkAgg(fig, master=graph_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # 添加关闭按钮
            btn_close = ttk.Button(graph_window, text="关闭", 
                                command=graph_window.destroy)
            btn_close.pack(pady=10)
            
        except Exception as e:
            messagebox.showerror("错误", f"生成图像时出错: {str(e)}")

    def calculate_conversion(self):
        """计算马赫数-速度系数转换"""
        try:
            input_value = float(self.entry_input.get())
            gamma = float(self.entry_gamma3.get())
            
            if self.conversion_var.get() == "Ma":
                # 已知Ma求λ
                Ma = input_value
                lambda_val = Ma * math.sqrt(
                    (gamma + 1) / (2 + (gamma - 1) * Ma ** 2))
            else:
                # 已知λ求Ma
                lambda_val = input_value
                Ma = lambda_val * math.sqrt( 
                    2 / ((gamma + 1) - (gamma - 1) * lambda_val ** 2))
  
            
            # 计算其他参数
            T_ratio1 = (1+(gamma-1)/2*Ma**2)**(-1)
            P_ratio1 = (1+(gamma-1)/2*Ma**2)**(-gamma/(gamma-1))
            rho_ratio1 = (1+(gamma-1)/2*Ma**2)**(-1/(gamma-1))
            term1 = lambda_val
            term2 = ((gamma + 1) / 2) ** (1 / (gamma - 1))
            inner = 1 - (gamma - 1) / (gamma + 1) * (lambda_val ** 2)
            term3 = inner ** (1 / (gamma - 1))
            q_r1 = term1 * term2 * term3
            y_r1 = q_r1/P_ratio1
            z_r1 = 1/lambda_val + lambda_val
            f_r1 = ((2/(gamma+1))**(1/(gamma-1)))*q_r1*z_r1
            r_r1 = P_ratio1/f_r1

            # 更新结果
            self.conversion_labels['Ma'].config(text=f"马赫数 Ma = {Ma:.4f}")
            self.conversion_labels['lambda'].config(text=f"速度系数 λ = {lambda_val:.4f}")
            self.conversion_labels['T_ratio1'].config(text=f" τ=T*/T = {T_ratio1:.4f}")
            self.conversion_labels['P_ratio1'].config(text=f" π=P*/P = {P_ratio1:.4f}")
            self.conversion_labels['rho_ratio1'].config(text=f"ε= ρ*/ρ = {rho_ratio1:.4f}")
            self.conversion_labels['q_r1'].config(text=f"q = {q_r1:.4f}")
            self.conversion_labels['y_r1'].config(text=f"y = {y_r1:.4f}")
            self.conversion_labels['z_r1'].config(text=f"z = {z_r1:.4f}")
            self.conversion_labels['f_r1'].config(text=f"f = {f_r1:.4f}")
            self.conversion_labels['r_r1'].config(text=f"r = {r_r1:.4f}")


        except ValueError:
            messagebox.showerror("输入错误", "请输入有效的数字参数")
        # 在calculate_conversion方法的适当位置添加如下代码
        self.lambda_val = lambda_val 

    def show_graphs(self):
        """根据条件生成并展示三个图表"""
        gamma = float(self.entry_gamma3.get())#传入参数
        lambda_max = math.sqrt((gamma + 1) / (gamma - 1))#计算渐近线
        ma_values = np.linspace(0.0, 10.0, 400)#调整lambda取值范围和步长
        lambda_values = [ma * math.sqrt((gamma + 1) / (2 + (gamma - 1) * ma ** 2)) for ma in ma_values]
        tau_values = [(1 + (gamma - 1)/2 * ma ** 2) ** (-1) for ma in ma_values]
        pi_values = [(1 + (gamma - 1)/2 * ma ** 2) ** (-gamma / (gamma - 1)) for ma in ma_values]
        epsilon_values = [(1 + (gamma - 1)/2 * ma ** 2) ** (-1 / (gamma - 1)) for ma in ma_values]

        # 创建一个2行2列的网格布局，但是第二行第二个位置不放置图像
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        axs[1, 0].axis('off')  # 关闭不需要的第四个子图

        # λ-Ma曲线
        axs[0, 0].plot(ma_values, lambda_values, label='λ', color='blue')
        axs[0, 0].axhline(y=lambda_max, color='r', linestyle='--', label=f'λ max ({lambda_max:.2f})')
        axs[0, 0].set_xlabel('Ma')
        axs[0, 0].set_ylabel('λ')
        axs[0, 0].set_title('马赫数与速度系数关系 (λ - Ma)')
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        # τ、π、ε-λ曲线
        axs[0, 1].plot(lambda_values, tau_values, label='τ', color='red')
        axs[0, 1].plot(lambda_values, pi_values, label='π', color='blue')
        axs[0, 1].plot(lambda_values, epsilon_values, label='ε', color='green')
        axs[0, 1].set_xlabel('λ')
        axs[0, 1].set_ylabel('值')
        axs[0, 1].set_title('气动函数随 λ 的变化')
        axs[0, 1].legend()
        axs[0, 1].grid(True)

        # τ、π、ε-Ma曲线
        axs[1, 1].plot(ma_values, tau_values, label='τ', color='red')
        axs[1, 1].plot(ma_values, pi_values, label='π', color='blue')
        axs[1, 1].plot(ma_values, epsilon_values, label='ε', color='green')
        axs[1, 1].set_xlabel('Ma')
        axs[1, 1].set_ylabel('值')
        axs[1, 1].set_title('气动函数随 Ma 的变化')
        axs[1, 1].legend()
        axs[1, 1].grid(True)

        plt.tight_layout()
        plt.show()

    def create_static_tab(self):
        """创建已知静参数计算滞止参数标签页"""
        frame = self.static_frame
        
        # 输入参数框架
        input_frame = ttk.LabelFrame(frame, text="输入参数")
        input_frame.pack(pady=10, padx=10, fill="x")
        
        ttk.Label(input_frame, text="静温 T (K):").grid(row=0, column=0, padx=5, pady=5)
        self.entry_T4 = ttk.Entry(input_frame)
        self.entry_T4.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="静压 P (Pa):").grid(row=1, column=0, padx=5, pady=5)
        self.entry_P4 = ttk.Entry(input_frame)
        self.entry_P4.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="静密度 ρ (kg/m³):").grid(row=2, column=0, padx=5, pady=5)
        self.entry_rho4 = ttk.Entry(input_frame)
        self.entry_rho4.grid(row=2, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="马赫数 Ma:").grid(row=3, column=0, padx=5, pady=5)
        self.entry_ma4 = ttk.Entry(input_frame)
        self.entry_ma4.grid(row=3, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="比热比 γ:").grid(row=4, column=0, padx=5, pady=5)
        self.entry_gamma4 = ttk.Entry(input_frame)
        self.entry_gamma4.insert(0, "1.4")
        self.entry_gamma4.grid(row=4, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="气体常数 R (J/kg·K):").grid(row=5, column=0, padx=5, pady=5)
        self.entry_R4 = ttk.Entry(input_frame)
        self.entry_R4.insert(0, "287")
        self.entry_R4.grid(row=5, column=1, padx=5, pady=5)
        
        # 计算按钮
        btn_calc = ttk.Button(frame, text="计算总参数", command=self.calculate_from_static)
        btn_calc.pack(pady=10)
        
        # 结果显示框架
        result_frame = ttk.LabelFrame(frame, text="计算结果")
        result_frame.pack(pady=10, padx=10, fill="x")

        self.static_labels = {
            'T0': ttk.Label(result_frame, text="总温 T* (K) = "),
            'P0': ttk.Label(result_frame, text="总压 P* (Pa) = "),
            'rho0': ttk.Label(result_frame, text="总密度 ρ* (kg/m³) = "),
            'a': ttk.Label(result_frame, text="当地声速 c (m/s) = "),
            'a0': ttk.Label(result_frame, text="滞止声速 c (m/s) = ")          
        }
        
        for i, (_, label) in enumerate(self.static_labels.items()):
            label.grid(row=i, column=0, sticky="w", padx=5, pady=2)
    
    def calculate_from_static(self):
        """根据静参数计算滞止参数"""
        try:
            T = float(self.entry_T4.get())
            P = float(self.entry_P4.get())
            rho = float(self.entry_rho4.get())
            Ma = float(self.entry_ma4.get())
            gamma = float(self.entry_gamma4.get())
            R = float(self.entry_R4.get())
            
            # 计算总参数
            T0 = T * (1 + (gamma - 1) / 2 * Ma ** 2)
            P0 = P * (1 + (gamma - 1) / 2 * Ma ** 2) ** (gamma / (gamma - 1))
            rho0 = rho * (1 + (gamma - 1) / 2 * Ma ** 2) ** (1 / (gamma - 1))
            
            # 计算声速
            a = math.sqrt(gamma * R * T)
            a0 = math.sqrt(gamma * R * T0)
            
            # 更新结果
            self.static_labels['T0'].config(text=f"总温 T* (K) = {T0:.2f}")
            self.static_labels['P0'].config(text=f"总压 P* (Pa) = {P0:.2f}")
            self.static_labels['rho0'].config(text=f"总密度 ρ* (kg/m³) = {rho0:.6f}")
            self.static_labels['a'].config(text=f"当地声速 c (m/s) = {a:.2f}")
            self.static_labels['a0'].config(text=f"滞止声速 c* (m/s) = {a0:.2f}")
            
        except ValueError:
            messagebox.showerror("输入错误", "请输入有效的数字参数")
    def create_total_tab(self):
    #"""创建已知总参数计算静参数标签页"""
        frame = self.total_frame
    
    # 输入参数框架
        input_frame = ttk.LabelFrame(frame, text="输入参数")
        input_frame.pack(pady=10, padx=10, fill="x")
    
        ttk.Label(input_frame, text="总温 T* (K):").grid(row=0, column=0, padx=5, pady=5)
        self.entry_T5 = ttk.Entry(input_frame)
        self.entry_T5.grid(row=0, column=1, padx=5, pady=5)
    
        ttk.Label(input_frame, text="总压 P* (Pa):").grid(row=1, column=0, padx=5, pady=5)
        self.entry_P5 = ttk.Entry(input_frame)
        self.entry_P5.grid(row=1, column=1, padx=5, pady=5)
    
        ttk.Label(input_frame, text="总密度 ρ* (kg/m³):").grid(row=2, column=0, padx=5, pady=5)
        self.entry_rho5 = ttk.Entry(input_frame)
        self.entry_rho5.grid(row=2, column=1, padx=5, pady=5)
    
        ttk.Label(input_frame, text="马赫数 Ma:").grid(row=3, column=0, padx=5, pady=5)
        self.entry_ma5 = ttk.Entry(input_frame)
        self.entry_ma5.grid(row=3, column=1, padx=5, pady=5)
    
        ttk.Label(input_frame, text="比热比 γ:").grid(row=4, column=0, padx=5, pady=5)
        self.entry_gamma5 = ttk.Entry(input_frame)
        self.entry_gamma5.insert(0, "1.4")
        self.entry_gamma5.grid(row=4, column=1, padx=5, pady=5)
    
        ttk.Label(input_frame, text="气体常数 R (J/kg·K):").grid(row=5, column=0, padx=5, pady=5)
        self.entry_R5 = ttk.Entry(input_frame)
        self.entry_R5.insert(0, "287")
        self.entry_R5.grid(row=5, column=1, padx=5, pady=5)
    
    # 计算按钮
        btn_calc = ttk.Button(frame, text="计算静参数", command=self.calculate_from_total)
        btn_calc.pack(pady=10)
       
    # 结果显示框架
        result_frame = ttk.LabelFrame(frame, text="计算结果")
        result_frame.pack(pady=10, padx=10, fill="x")

        self.total_labels = {
        'T': ttk.Label(result_frame, text="静温 T (K) = "),
        'P': ttk.Label(result_frame, text="静压 P (Pa) = "),
        'rho': ttk.Label(result_frame, text="静密度 ρ (kg/m³) = "),
        'a': ttk.Label(result_frame, text="当地声速 c (m/s) = "),
        'a0': ttk.Label(result_frame, text="滞止声速 c* (m/s) = ")          
        }
    
        for i, (_, label) in enumerate(self.total_labels.items()):
            label.grid(row=i, column=0, sticky="w", padx=5, pady=2)
    
    def calculate_from_total(self):
    #"""根据总参数计算静参数"""
        try:
            T0 = float(self.entry_T5.get())
            P0 = float(self.entry_P5.get())
            rho0 = float(self.entry_rho5.get())
            Ma = float(self.entry_ma5.get())
            gamma = float(self.entry_gamma5.get())
            R = float(self.entry_R5.get())
        
        # 计算滞止参数
            denominator = 1 + (gamma - 1) / 2 * Ma ** 2
          
        # 计算静参数
            T = T0 / denominator
            P = P0 / (denominator ** (gamma / (gamma - 1)))
            rho = rho0 / (denominator ** (1 / (gamma - 1)))
        
        # 计算声速
            a = math.sqrt(gamma * R * T)       # 当地声速
            a0 = math.sqrt(gamma * R * T0)     # 滞止声速
        
        # 更新结果
            self.total_labels['T'].config(text=f"静温 T (K) = {T:.2f}")
            self.total_labels['P'].config(text=f"静压 P (Pa) = {P:.2f}")
            self.total_labels['rho'].config(text=f"静密度 ρ (kg/m³) = {rho:.6f}")
            self.total_labels['a'].config(text=f"当地声速 c (m/s) = {a:.2f}")
            self.total_labels['a0'].config(text=f"滞止声速 c* (m/s) = {a0:.2f}")
        
        except ValueError:
            messagebox.showerror("输入错误", "请输入有效的数字参数")


class CombinedShockwaveWindow:
    def __init__(self, master):
        self.master = master
        master.title("综合激波参数计算")
        master.geometry("650x500")  # 调整尺寸以适应两个子窗口
        
        # 创建一个Notebook组件
        notebook = ttk.Notebook(master)
        
        # 创建各个子窗口对应的框架
        shockwave_frame = ttk.Frame(notebook)  # 斜激波参数计算1
        
        pressure_shockwave_frame = ttk.Frame(notebook)  # 斜激波参数计算2
        velocity_shockwave_frame = ttk.Frame(notebook)  # 正激波参数计算
        
        # 实例化各个子窗口并将它们的控件添加到对应的框架中
        self.shockwave_window = ShockwaveWindow(shockwave_frame)
        self.velocity_shockwave_window = VelocityShockwaveWindow(velocity_shockwave_frame)
        self.pressure_shockwave_window = ShockwaveByPressure(pressure_shockwave_frame)
        
        # 将这些框架作为标签页添加到notebook中
        notebook.add(shockwave_frame, text='斜激波参数计算1')
        
        notebook.add(pressure_shockwave_frame, text='斜激波参数计算2')
        notebook.add(velocity_shockwave_frame, text='正激波参数计算')
        
        # 布局notebook
        notebook.pack(fill="both", expand=True)


class ShockwaveWindow:
    def __init__(self, parent):
        # 接收父级容器而不是创建新的顶级窗口
        self.frame = ttk.Frame(parent)
        self.frame.pack(fill="both", expand=True)

        # 输入参数框架
        input_frame = ttk.LabelFrame(parent, text="输入参数")
        input_frame.pack(pady=10, padx=20, fill="x")

        ttk.Label(input_frame, text="来流马赫数 Ma1:").grid(row=0, column=0, padx=5, pady=5)
        self.entry_ma1 = ttk.Entry(input_frame)
        self.entry_ma1.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="转折角 θ (°):").grid(row=1, column=0, padx=5, pady=5)
        self.entry_theta = ttk.Entry(input_frame)
        self.entry_theta.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="比热比 γ:").grid(row=2, column=0, padx=5, pady=5)
        self.entry_gamma = ttk.Entry(input_frame)
        self.entry_gamma.insert(0, "1.4")
        self.entry_gamma.grid(row=2, column=1, padx=5, pady=5)

        # 计算按钮
        btn_calc = ttk.Button(parent, text="计算", command=self.calculate)
        btn_calc.pack(pady=10)

        # 添加显示图像按钮
        btn_plot = ttk.Button(parent, text="显示图像", command=self.plot_shock_angle_vs_mach)
        btn_plot.pack(pady=10)

        # 结果显示框架
        result_frame = ttk.LabelFrame(parent, text="激波参数结果")
        result_frame.pack(pady=10, padx=20, fill="both", expand=True)

        # 创建两个子框架分别显示弱波和强波结果
        self.weak_frame = ttk.LabelFrame(result_frame, text="弱激波解")
        self.weak_frame.pack(side="left", padx=10, pady=5, fill="both", expand=True)

        self.strong_frame = ttk.LabelFrame(result_frame, text="强激波解")
        self.strong_frame.pack(side="right", padx=10, pady=5, fill="both", expand=True)

        # 结果标签
        result_labels = [
            "激波角 β (°)",
            "波后马赫数 Ma2",
            "压力比 P2/P1",
            "温度比 T2/T1",
            "密度比 ρ2/ρ1",
            "总压比 P02/P01"
        ]

        # 初始化弱波结果标签
        self.weak_labels = []
        for i, text in enumerate(result_labels):
            label = ttk.Label(self.weak_frame, text=f"{text}: ")
            label.grid(row=i, column=0, sticky="w", padx=5, pady=2)
            self.weak_labels.append(label)

        # 初始化强波结果标签
        self.strong_labels = []
        for i, text in enumerate(result_labels):
            label = ttk.Label(self.strong_frame, text=f"{text}: ")
            label.grid(row=i, column=0, sticky="w", padx=5, pady=2)
            self.strong_labels.append(label)

    def calculate_beta(self, Ma1, theta_rad, gamma, is_weak=True):
        """数值求解θ-β-Ma方程（牛顿迭代法优化版）"""
        max_iter = 100
        tolerance = 1e-8  # 更高精度要求
        beta_min = math.asin(1 / Ma1)  # 马赫角（理论最小值）

        # 物理约束：强激波解上限设为80度（经验值），避免数值不稳定
        beta_max = math.radians(80) if is_weak else math.pi / 2 * 0.99

        # 优化初始猜测：基于经验公式
        beta_guess = beta_min * 1.2 if is_weak else math.pi / 2 * 0.99

        # 牛顿迭代过程
        beta = beta_guess
        for _ in range(max_iter):
            try:
                sin_beta = math.sin(beta)
                cos_beta = math.cos(beta)
                tan_beta = sin_beta / cos_beta  # 显式计算更稳定

                # 计算分子分母及其导数
                numerator = 2 * (Ma1 ** 2 * sin_beta ** 2 - 1)
                denominator = tan_beta * (Ma1 ** 2 * (gamma + math.cos(2 * beta)) + 2)

                # 检查分母有效性
                if abs(denominator) < 1e-12:
                    raise ZeroDivisionError

                f = (numerator / denominator) - math.tan(theta_rad)

                # 精确导数计算（使用商法则）
                d_num = 4 * Ma1 ** 2 * sin_beta * cos_beta
                d_den = ((1 / cos_beta ** 2) * (Ma1 ** 2 * (gamma + math.cos(2 * beta)) + 2)
                         + tan_beta * (-2 * Ma1 ** 2 * math.sin(2 * beta)))

                df = (d_num * denominator - numerator * d_den) / (denominator ** 2)

                # 迭代更新并约束范围
                beta -= f / df
                beta = max(beta_min, min(beta, beta_max))  # 强制物理约束

                if abs(f) < tolerance:
                    return beta

            except (ZeroDivisionError, ValueError):
                # 出现奇异点时重置猜测值
                beta = (beta + beta_max) / 2

        return None  # 未收敛时返回None

    def calculate_shock_params(self, Ma1, beta, gamma):
        """ 计算激波参数 """
        sin_beta = math.sin(beta)
        Ma1n = Ma1 * sin_beta
        # 斜激波关系式
        Ma2n = math.sqrt(((gamma - 1) * Ma1n ** 2 + 2) / (2 * gamma * Ma1n ** 2 - (gamma - 1)))
        P_ratio = (2 * gamma * Ma1n ** 2 - (gamma - 1)) / (gamma + 1)
        rho_ratio = (gamma + 1) * Ma1n ** 2 / ((gamma - 1) * Ma1n ** 2 + 2)
        T_ratio = P_ratio / rho_ratio
        # 总压比
        P0_ratio = (((gamma + 1) * Ma1n ** 2) / ((gamma - 1) * Ma1n ** 2 + 2)) ** (gamma / (gamma - 1)) * \
                   ((gamma + 1) / (2 * gamma * Ma1n ** 2 - (gamma - 1))) ** (1 / (gamma - 1))
        # 波后马赫数
        Ma2 = Ma2n / math.sin(beta - math.atan(2 * (Ma1n ** 2 - 1) / ((gamma + 1) * Ma1 ** 2 * math.tan(beta))))
        return {
            'beta': math.degrees(beta),
            'Ma2': Ma2,
            'P_ratio': P_ratio,
            'T_ratio': T_ratio,
            'rho_ratio': rho_ratio,
            'P0_ratio': P0_ratio
        }

    def calculate(self):
        try:
            Ma1 = float(self.entry_ma1.get())
            theta_deg = float(self.entry_theta.get())
            gamma = float(self.entry_gamma.get())

            if Ma1 <= 1:
                raise ValueError("来流马赫数必须大于1")
            if theta_deg <= 0 or theta_deg >= 90:
                raise ValueError("转折角应在0°到90°之间")

            theta_rad = math.radians(theta_deg)

            # 计算最大可能转折角
            theta_max = math.degrees(self.calculate_max_theta(Ma1, gamma))
            if theta_deg > theta_max:
                raise ValueError(f"转折角超过最大可能值 {theta_max:.2f}°")

            # 求解弱波和强波
            beta_weak = self.calculate_beta(Ma1, theta_rad, gamma, is_weak=True)
            beta_strong = self.calculate_beta(Ma1, theta_rad, gamma, is_weak=False)

            if not beta_weak or not beta_strong:
                raise ValueError("未能找到有效解")

            # 计算参数
            weak_params = self.calculate_shock_params(Ma1, beta_weak, gamma)
            strong_params = self.calculate_shock_params(Ma1, beta_strong, gamma)

            # 更新弱波结果
            self.weak_labels[0].config(text=f"激波角 β (°): {weak_params['beta']:.2f}")
            self.weak_labels[1].config(text=f"波后马赫数 Ma2: {weak_params['Ma2']:.4f}")
            self.weak_labels[2].config(text=f"压力比 P2/P1: {weak_params['P_ratio']:.4f}")
            self.weak_labels[3].config(text=f"温度比 T2/T1: {weak_params['T_ratio']:.4f}")
            self.weak_labels[4].config(text=f"密度比 ρ2/ρ1: {weak_params['rho_ratio']:.4f}")
            self.weak_labels[5].config(text=f"总压比 P02/P01: {weak_params['P0_ratio']:.6f}")

            # 更新强波结果
            self.strong_labels[0].config(text=f"激波角 β (°): {strong_params['beta']:.2f}")
            self.strong_labels[1].config(text=f"波后马赫数 Ma2: {strong_params['Ma2']:.4f}")
            self.strong_labels[2].config(text=f"压力比 P2/P1: {strong_params['P_ratio']:.4f}")
            self.strong_labels[3].config(text=f"温度比 T2/T1: {strong_params['T_ratio']:.4f}")
            self.strong_labels[4].config(text=f"密度比 ρ2/ρ1: {strong_params['rho_ratio']:.4f}")
            self.strong_labels[5].config(text=f"总压比 P02/P01: {strong_params['P0_ratio']:.6f}")

        except ValueError as e:
            messagebox.showerror("输入错误", str(e))

    def calculate_max_theta(self, Ma1, gamma):
        """计算最大激波转折角"""
        theta_max = 0.0  # 初始为0，假设至少存在一个有效解

        # 遍历更密集的角度（步长0.1度），从89.9度到0.1度
        for beta_deg in range(899, 0, -1):
            beta = math.radians(beta_deg / 10)  # 转换为0.1度步长

            try:
                # 计算分子和分母
                sin_beta = math.sin(beta)
                ma_sq_sin_beta_sq = Ma1 ** 2 * sin_beta ** 2
                numerator = 2 * (ma_sq_sin_beta_sq - 1)

                cos_2beta = math.cos(2 * beta)
                denominator = math.tan(beta) * (Ma1 ** 2 * (gamma + cos_2beta) + 2)

                # 计算转折角并确保分母不为零
                theta = math.atan(numerator / denominator)

                # 只保留正角度且更大的值
                if theta > theta_max and theta > 0:
                    theta_max = theta

            except (ZeroDivisionError, ValueError):
                continue  # 跳过无效计算

        return theta_max

    def plot_shock_angle_vs_mach(self):
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            import numpy as np
        except ImportError:
            messagebox.showerror("错误", "需要安装matplotlib库")
            return
        
        try:
            # 获取用户输入参数
            theta_deg = float(self.entry_theta.get())
            gamma = float(self.entry_gamma.get())
            user_ma1 = float(self.entry_ma1.get())
            
            if theta_deg <= 0 or theta_deg >= 90:
                raise ValueError("转折角应在0°到90°之间")
            
            theta_rad = math.radians(theta_deg)
            
            # 创建新窗口
            plot_window = tk.Toplevel(self.frame.winfo_toplevel())
            plot_window.title(f"激波角随马赫数变化 (θ={theta_deg}°, γ={gamma})")
            plot_window.geometry("900x650")  # 稍微增大窗口尺寸
            
            fig, ax = plt.subplots(figsize=(9, 6))
            plt.subplots_adjust(top=0.92, bottom=0.12, left=0.1, right=0.95)  # 调整边距
            
            # 计算最小马赫数（脱体马赫数）
            min_ma1 = self.find_min_ma1(theta_deg, gamma)
            
            # 生成马赫数范围（从min_ma1到5.0）
            ma1_values = np.linspace(min_ma1 + 0.01, user_ma1*1.2, 100)
            weak_beta_values = []
            strong_beta_values = []
            
            # 计算每个马赫数对应的弱激波和强激波角
            for ma1 in ma1_values:
                # 弱激波解
                beta_weak = self.calculate_beta(ma1, theta_rad, gamma, is_weak=True)
                if beta_weak is not None:
                    weak_beta_values.append(math.degrees(beta_weak))
                else:
                    weak_beta_values.append(float('nan'))
                
                # 强激波解
                beta_strong = self.calculate_beta(ma1, theta_rad, gamma, is_weak=False)
                if beta_strong is not None:
                    strong_beta_values.append(math.degrees(beta_strong))
                else:
                    strong_beta_values.append(float('nan'))
            
            # 绘制弱激波曲线
            weak_line, = ax.plot(ma1_values, weak_beta_values, 'b-', linewidth=2)
            
            # 绘制强激波曲线
            strong_line, = ax.plot(ma1_values, strong_beta_values, 'r-', linewidth=2)
            
            # 标记最小马赫数点（脱体点）
            min_beta_weak = self.calculate_beta(min_ma1, theta_rad, gamma, is_weak=True)
            min_beta_strong = self.calculate_beta(min_ma1, theta_rad, gamma, is_weak=False)
            
            if min_beta_weak is not None and min_beta_strong is not None:
                min_beta_deg = (math.degrees(min_beta_weak) + math.degrees(min_beta_strong)) / 2
                detach_point = ax.plot(min_ma1, min_beta_deg, 'ro', markersize=8, zorder=5)[0]
                ax.annotate(f'脱体马赫数 ({min_ma1:.3f}, {min_beta_deg:.1f}°)',
                            xy=(min_ma1, min_beta_deg),
                            xytext=(min_ma1 + 0.3, min_beta_deg - 5),
                            arrowprops=dict(facecolor='red', shrink=0.05),
                            zorder=5)
            
            # 绘制用户输入马赫数的垂直线
            if min_ma1 <= user_ma1 <= user_ma1*1.2:
                ax.axvline(x=user_ma1, color='g', linestyle='--', linewidth=2, zorder=3)
                
                # 弱激波解标记
                beta_weak = self.calculate_beta(user_ma1, theta_rad, gamma, is_weak=True)
                if beta_weak is not None:
                    beta_weak_deg = math.degrees(beta_weak)
                    weak_point = ax.plot(user_ma1, beta_weak_deg, 'go', markersize=8, zorder=5)[0]
                    ax.annotate(f'弱波点 ({user_ma1}, {beta_weak_deg:.1f}°)',
                                xy=(user_ma1, beta_weak_deg),
                                xytext=(user_ma1 + 0.2, beta_weak_deg + 5),
                                arrowprops=dict(facecolor='green', shrink=0.05),
                                zorder=5)
                
                # 强激波解标记
                beta_strong = self.calculate_beta(user_ma1, theta_rad, gamma, is_weak=False)
                if beta_strong is not None:
                    beta_strong_deg = math.degrees(beta_strong)
                    strong_point = ax.plot(user_ma1, beta_strong_deg, 'mo', markersize=8, zorder=5)[0]
                    ax.annotate(f'强波点 ({user_ma1}, {beta_strong_deg:.1f}°)',
                                xy=(user_ma1, beta_strong_deg),
                                xytext=(user_ma1 + 0.2, beta_strong_deg - 5),
                                arrowprops=dict(facecolor='magenta', shrink=0.05),
                                zorder=5)
            
            # 设置图形属性
            ax.set_title(f"激波角随马赫数变化 (θ={theta_deg}°, γ={gamma})", fontsize=14, pad=15)
            ax.set_xlabel("来流马赫数 Ma1", fontsize=12)
            ax.set_ylabel("激波角 β (°)", fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # 优化图例位置和样式 - 放在右上角，使用半透明背景
            legend_items = []
            if 'weak_line' in locals():
                legend_items.append(weak_line)
            if 'strong_line' in locals():
                legend_items.append(strong_line)
            if 'detach_point' in locals():
                legend_items.append(detach_point)
            if 'weak_point' in locals() and 'strong_point' in locals():
                legend_items.append(weak_point)
                legend_items.append(strong_point)
            
            labels = ['弱激波解', '强激波解', '脱体点', '弱波点', '强波点']
            
            # 创建自定义图例，放在右上角外部
            ax.legend(legend_items[:2], labels[:2], 
                    loc='upper left', bbox_to_anchor=(0.01, 0.99),
                    framealpha=0.8, fontsize=10)
            
            # 在右下角添加额外图例说明点
            if 'detach_point' in locals() or 'weak_point' in locals() or 'strong_point' in locals():
                extra_legend = []
                extra_labels = []
                
                if 'detach_point' in locals():
                    extra_legend.append(detach_point)
                    extra_labels.append('脱体点')
                if 'weak_point' in locals():
                    extra_legend.append(weak_point)
                    extra_labels.append('弱波点')
                if 'strong_point' in locals():
                    extra_legend.append(strong_point)
                    extra_labels.append('强波点')
                    
                ax.legend(extra_legend, extra_labels,
                        loc='lower right', bbox_to_anchor=(0.99, 0.01),
                        framealpha=0.8, fontsize=10)
            
            # 添加理论说明
            plt.figtext(0.5, 0.02, 
                    f"注：当Ma1 < {min_ma1:.3f}时，激波脱体，不存在附着激波解",
                    ha="center", fontsize=10, style='italic')
            
            # 嵌入图形到Tkinter窗口
            canvas = FigureCanvasTkAgg(fig, master=plot_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except ValueError as e:
            messagebox.showerror("输入错误", str(e))
    
    def find_min_ma1(self, theta_deg, gamma, tolerance=1e-5):
        """使用二分法寻找最小马赫数（脱体马赫数）"""
        low = 1.0 + tolerance
        high = 10.0  # 设置足够大的上界
        
        # 检查转折角是否可能
        max_theta = math.degrees(self.calculate_max_theta(high, gamma))
        if theta_deg > max_theta:
            raise ValueError(f"转折角超过最大可能值 {max_theta:.2f}°")
        
        # 二分法搜索
        while high - low > tolerance:
            mid = (low + high) / 2
            max_theta_mid = math.degrees(self.calculate_max_theta(mid, gamma))
            
            if max_theta_mid >= theta_deg:
                high = mid
            else:
                low = mid
        
        return (low + high) / 2

    def open_shockwave(self):
        ShockwaveWindow(tk.Toplevel(self.master))


class VelocityShockwaveWindow:
    def __init__(self, parent):
        # 同样接收父级容器而不是创建新的顶级窗口
        self.frame = ttk.Frame(parent)
        self.frame.pack(fill="both", expand=True)

        # 输入参数框架
        input_frame = ttk.LabelFrame(parent, text="输入参数（国际单位制：Pa, kg/m³, m/s）")
        input_frame.pack(pady=10, padx=20, fill="x")

        input_params = [
            ("波前压强 P1:", "entry_p1"),
            ("波前密度 ρ1:", "entry_rho1"),
            ("波前速度 V1:", "entry_v1"),
            ("波后速度 V2:", "entry_v2")
        ]

        self.entries = {}
        for row, (text, name) in enumerate(input_params):
            ttk.Label(input_frame, text=text).grid(row=row, column=0, padx=5, pady=5, sticky="w")
            entry = ttk.Entry(input_frame)
            entry.grid(row=row, column=1, padx=5, pady=5)
            self.entries[name] = entry

        # 计算按钮
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="计算", command=self.calculate).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="清除", command=self.clear).pack(side=tk.LEFT, padx=5)

        # 结果显示框架
        self.result_frame = ttk.LabelFrame(parent, text="正激波计算结果")
        self.result_frame.pack(pady=10, padx=20, fill="x", expand=True)

        result_labels = [
            ("密度比", "ρ₂/ρ₁ ="),
            ("压强比", "P₂/P₁ ="),
            ("温度比", "T₂/T₁ ="),
            ("比热比", "γ ="),
            ("波前马赫数", "Ma₁ ="),
            ("波后马赫数", "Ma₂ =")
        ]

        self.result_vars = []
        for i, (desc, formula) in enumerate(result_labels):
            ttk.Label(self.result_frame, text=formula, font=('Arial', 10)).grid(row=i, column=0, padx=5, pady=2,
                                                                                sticky="w")
            var = tk.StringVar()
            ttk.Label(self.result_frame, textvariable=var, font=('Arial', 10, 'bold')).grid(row=i, column=1, padx=5,
                                                                                            pady=2, sticky="e")
            self.result_vars.append((desc, var))

    def validate_inputs(self):
        """ 输入验证和物理合理性检查 """
        try:
            p1 = float(self.entries["entry_p1"].get())
            v1 = float(self.entries["entry_v1"].get())
            rho1 = float(self.entries["entry_rho1"].get())
            v2 = float(self.entries["entry_v2"].get())

            if not all(x > 0 for x in [p1, v1, rho1, v2]):
                raise ValueError("所有输入值必须为正数")

            if v2 >= v1:
                raise ValueError("激波后速度必须小于波前速度")

            return p1, v1, rho1, v2

        except ValueError as e:
            messagebox.showerror("输入错误", str(e))
            raise

    def solve_gamma(self, p1, v1, rho1, v2):
        """ 牛顿迭代法求解比热比γ """

        def energy_equation(gamma):
            try:
                # 质量守恒求密度比
                rho_ratio = v1 / v2
                rho2 = rho1 * rho_ratio

                # 动量守恒求压强比
                p_ratio = (p1 + rho1 * v1 ** 2 - rho1 * v1 * v2) / p1
                p2 = p1 * p_ratio

                # 能量守恒方程
                h1 = gamma / (gamma - 1) * p1 / rho1
                h2 = gamma / (gamma - 1) * p2 / rho2
                return (h1 + 0.5 * v1 ** 2) - (h2 + 0.5 * v2 ** 2)
            except ZeroDivisionError:
                return float('inf')

        # 迭代参数设置
        gamma_guess = 1.4  # 空气的典型初始值
        tolerance = 1e-8
        max_iter = 100

        for _ in range(max_iter):
            f = energy_equation(gamma_guess)
            if abs(f) < tolerance:
                return gamma_guess
            # 数值微分计算导数
            h = 1e-6
            f_h = energy_equation(gamma_guess + h)
            df = (f_h - f) / h
            if abs(df) < 1e-12:
                break
            gamma_guess -= f / df
            gamma_guess = max(1.3, min(gamma_guess, 1.67))  # 物理合理范围

        raise ValueError("无法收敛，请检查输入参数是否满足正激波条件")

    def calculate_mach(self, gamma, p, rho, v):
        """ 计算马赫数 """
        speed_of_sound = math.sqrt(gamma * p / rho)
        return v / speed_of_sound

    def calculate(self):
        try:
            # 输入验证
            p1, v1, rho1, v2 = self.validate_inputs()

            # 基本守恒定律计算
            rho_ratio = v1 / v2
            p_ratio = (p1 + rho1 * v1 ** 2 - rho1 * v1 * v2) / p1
            T_ratio = p_ratio * (rho1 / (rho1 * rho_ratio))  # T = P/(ρR)

            # 数值求解比热比
            gamma = self.solve_gamma(p1, v1, rho1, v2)

            # 计算马赫数
            Ma1 = self.calculate_mach(gamma, p1, rho1, v1)
            p2 = p1 * p_ratio
            rho2 = rho1 * rho_ratio
            Ma2 = self.calculate_mach(gamma, p2, rho2, v2)

            # 结果格式化
            results = [
                f"{rho_ratio:.4f}",
                f"{p_ratio:.4f}",
                f"{T_ratio:.4f}",
                f"{gamma:.4f}",
                f"{Ma1:.4f}",
                f"{Ma2:.4f}"
            ]

            # 更新显示结果
            for (desc, var), value in zip(self.result_vars, results):
                var.set(f"{value}  ")

            # 后验验证
            if Ma1 < 1:
                messagebox.showwarning("警告", "波前马赫数小于1不符合正激波条件")
            if Ma2 > 1:
                messagebox.showwarning("警告", "波后马赫数大于1不符合正激波条件")

        except Exception as e:
            messagebox.showerror("计算错误", str(e))

    def clear(self):
        """ 清除所有输入和结果 """
        for entry in self.entries.values():
            entry.delete(0, tk.END)
        for _, var in self.result_vars:
            var.set("")

class ShockwaveByPressure:
    def __init__(self, parent):
        # 同样接收父级容器而不是创建新的顶级窗口
        self.frame = ttk.Frame(parent)
        self.frame.pack(fill="both", expand=True)

        # 输入参数框架
        input_frame = ttk.LabelFrame(parent, text="输入参数")
        input_frame.pack(pady=10, padx=20, fill="x")

        # 波前参数
        ttk.Label(input_frame, text="来流马赫数 Ma1:").grid(row=0, column=0, padx=5, pady=5)
        self.entry_ma1 = ttk.Entry(input_frame)
        self.entry_ma1.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="波后压强比 P2/P1:").grid(row=1, column=0, padx=5, pady=5)
        self.entry_p_ratio = ttk.Entry(input_frame)
        self.entry_p_ratio.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="比热比 γ:").grid(row=2, column=0, padx=5, pady=5)
        self.entry_gamma = ttk.Entry(input_frame)
        self.entry_gamma.insert(0, "1.4")
        self.entry_gamma.grid(row=2, column=1, padx=5, pady=5)

        # 计算按钮
        btn_calc = ttk.Button(parent, text="计算", command=self.calculate)
        btn_calc.pack(pady=10)

        # 结果显示框架
        result_frame = ttk.LabelFrame(parent, text="计算结果")
        result_frame.pack(pady=10, padx=20, fill="both", expand=True)

        # 结果标签
        result_labels = [
            "激波角 β (°)",
            "转折角 θ (°)",
            "波后马赫数 Ma2",
            "密度比 ρ2/ρ1",
            "温度比 T2/T1",
            "总压比 P02/P01"
        ]

        # 初始化结果标签
        self.result_labels = []
        for i, text in enumerate(result_labels):
            label = ttk.Label(result_frame, text=f"{text}: ")
            label.grid(row=i, column=0, sticky="w", padx=5, pady=2)
            self.result_labels.append(label)

    def calculate_beta_from_pressure(self, Ma1, p_ratio, gamma):
        """通过压强比直接计算激波角"""
        # 使用公式: P2/P1 = [2γ/(γ+1)] * (Ma1² sin²β) - (γ-1)/(γ+1)
        # 解出 sin²β
        numerator = p_ratio + 1.0*(gamma - 1) / (gamma + 1)
        denominator = (2.0* gamma / (gamma + 1)) * Ma1 * Ma1

        if denominator <= 0:
            raise ValueError("分母小于等于零，无法计算")

        sin2_beta = numerator / denominator

        # 检查sin²β是否在有效范围内
        if sin2_beta < 0 or sin2_beta > 1:
            raise ValueError(f"计算得到的sin²β={sin2_beta:.4f}超出[0,1]范围")

        sin_beta = math.sqrt(sin2_beta)
        beta = math.asin(sin_beta)
        return beta

    def calculate_theta(self, Ma1, beta, gamma):
        """计算转折角θ"""
        sin_beta = math.sin(beta)
        cos_beta = math.cos(beta)

        # 使用θ-β-Ma关系式
        numerator = 2.0 * ((Ma1 * sin_beta) ** 2 - 1) * cos_beta
        denominator = (Ma1 ** 2 * (gamma + math.cos(2 * beta)) + 2.0)* sin_beta

        if abs(denominator) < 1e-10:
            return math.pi / 2  # 当分母接近零时，θ≈90°

        tan_theta = numerator / denominator
        return math.atan(tan_theta)

    def calculate_shock_params(self, Ma1, beta, gamma):
        """计算激波参数"""
        sin_beta = math.sin(beta)
        Ma1n = Ma1 * sin_beta

        # 计算波后法向马赫数
        Ma2n_sq = ((gamma - 1) * Ma1n ** 2 + 2) / (2 * gamma * Ma1n ** 2 - (gamma - 1))
        Ma2n = math.sqrt(Ma2n_sq) if Ma2n_sq > 0 else 0

        # 计算转折角
        theta = self.calculate_theta(Ma1, beta, gamma)

        # 计算波后马赫数
        Ma2 = Ma2n / math.sin(beta - theta) if math.sin(beta - theta) != 0 else 0

        # 计算其他参数
        P_ratio = (2 * gamma * Ma1n ** 2 - (gamma - 1)) / (gamma + 1)
        rho_ratio = ((gamma + 1) * Ma1n ** 2) / ((gamma - 1) * Ma1n ** 2 + 2)
        T_ratio = P_ratio / rho_ratio

        # 计算总压比
        P0_ratio = (((gamma + 1) * Ma1n ** 2) / ((gamma - 1) * Ma1n ** 2 + 2)) ** (gamma / (gamma - 1)) * \
                   ((gamma + 1) / (2 * gamma * Ma1n ** 2 - (gamma - 1))) ** (1 / (gamma - 1))

        return {
            'beta': math.degrees(beta),
            'theta': math.degrees(theta),
            'Ma2': Ma2,
            'P_ratio': P_ratio,
            'T_ratio': T_ratio,
            'rho_ratio': rho_ratio,
            'P0_ratio': P0_ratio
        }

    def calculate(self):
        try:
            Ma1 = float(self.entry_ma1.get())
            p_ratio = float(self.entry_p_ratio.get())
            gamma = float(self.entry_gamma.get())

            if Ma1 <= 1:
                raise ValueError("来流马赫数必须大于1")
            if p_ratio <= 1:
                raise ValueError("波后压强比必须大于1")

            # 计算最大可能压强比（正激波）
            max_p_ratio = (2 * gamma * Ma1 ** 2 - (gamma - 1)) / (gamma + 1)
            if p_ratio > max_p_ratio:
                raise ValueError(f"压强比超过最大可能值 {max_p_ratio:.4f}")

            # 计算激波角
            beta = self.calculate_beta_from_pressure(Ma1, p_ratio, gamma)

            # 计算所有参数
            params = self.calculate_shock_params(Ma1, beta, gamma)

            # 更新结果
            self.result_labels[0].config(text=f"激波角 β (°): {params['beta']:.2f}")
            self.result_labels[1].config(text=f"转折角 θ (°): {params['theta']:.2f}")
            self.result_labels[2].config(text=f"波后马赫数 Ma2: {params['Ma2']:.4f}")
            self.result_labels[3].config(text=f"密度比 ρ2/ρ1: {params['rho_ratio']:.4f}")
            self.result_labels[4].config(text=f"温度比 T2/T1: {params['T_ratio']:.4f}")
            self.result_labels[5].config(text=f"总压比 P02/P01: {params['P0_ratio']:.6f}")

        except ValueError as e:
            messagebox.showerror("输入错误", str(e))


class ExpansionWindow:
    def __init__(self, master):
        self.master = master
        master.title("膨胀波计算")
        master.geometry("600x650")

        # 模式选择
        ttk.Label(master, text="选择计算案例:").pack(pady=5)
        self.mode_var = tk.StringVar()
        self.mode_combo = ttk.Combobox(master, textvariable=self.mode_var, state="readonly")
        self.mode_combo['values'] = [
            "已知 Ma1+θ → 求膨胀后参数",
            "已知 Ma2+θ → 求膨胀前参数",
            "已知 Ma1+Ma2 → 求 θ"
        ]
        self.mode_combo.current(0)
        self.mode_combo.pack(pady=5)
        self.mode_combo.bind("<<ComboboxSelected>>", self.update_input_state)

        # 输入参数框架
        input_frame = ttk.LabelFrame(master, text="输入参数")
        input_frame.pack(pady=10, padx=20, fill="x")

        ttk.Label(input_frame, text="波前马赫数 Ma1:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.entry_ma1 = ttk.Entry(input_frame)
        self.entry_ma1.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="波后马赫数 Ma2:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.entry_ma2 = ttk.Entry(input_frame)
        self.entry_ma2.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="比热比 γ:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.entry_gamma = ttk.Entry(input_frame)
        self.entry_gamma.insert(0, "1.4")
        self.entry_gamma.grid(row=2, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="转折角 θ (°):").grid(row=3, column=0, padx=5, pady=5, sticky="e")
        self.entry_theta = ttk.Entry(input_frame)
        self.entry_theta.grid(row=3, column=1, padx=5, pady=5)

        # 计算按钮
        btn_calc = ttk.Button(master, text="计算", command=self.calculate)
        btn_calc.pack(pady=10)

        # 结果显示框架
        result_frame = ttk.LabelFrame(master, text="计算结果")
        result_frame.pack(pady=10, padx=20, fill="x")

        self.labels = {
            'result': ttk.Label(result_frame, text="结果："),
            'p_ratio': ttk.Label(result_frame, text="压力比 P2/P1 = "),
            't_ratio': ttk.Label(result_frame, text="温度比 T2/T1 = "),
            'pm_angle': ttk.Label(result_frame, text="普朗特-迈耶角 ν = ")
        }

        for i, (_, label) in enumerate(self.labels.items()):
            label.grid(row=i, column=0, sticky="w", padx=5, pady=2)

        self.update_input_state()

    def prandtl_meyer(self, Ma, gamma):
        if Ma < 1:
            return 0
        nu = math.sqrt((gamma + 1) / (gamma - 1)) * math.atan(
            math.sqrt((gamma - 1) / (gamma + 1) * (Ma ** 2 - 1)))
        nu -= math.atan(math.sqrt(Ma ** 2 - 1))
        return math.degrees(nu)

    def inverse_prandtl_meyer(self, nu_target, gamma):
        Ma = 2.0
        tolerance = 1e-6
        for _ in range(100):
            nu = self.prandtl_meyer(Ma, gamma)
            error = nu - nu_target
            if abs(error) < tolerance:
                return Ma
            dnu = (self.prandtl_meyer(Ma + 0.001, gamma) - nu) / 0.001
            if abs(dnu) < 1e-8:
                break
            Ma -= error / dnu
        return Ma

    def update_input_state(self, event=None):
        mode = self.mode_var.get()
        if mode == "已知 Ma1+θ → 求膨胀后参数":
            self.entry_ma1.config(state="normal")
            self.entry_ma2.config(state="disabled")
            self.entry_theta.config(state="normal")
        elif mode == "已知 Ma2+θ → 求膨胀前参数":
            self.entry_ma1.config(state="disabled")
            self.entry_ma2.config(state="normal")
            self.entry_theta.config(state="normal")
        elif mode == "已知 Ma1+Ma2 → 求 θ":
            self.entry_ma1.config(state="normal")
            self.entry_ma2.config(state="normal")
            self.entry_theta.config(state="disabled")
        #else:  # 膨胀波相交/反射
        #    self.entry_ma1.config(state="normal")
        #    self.entry_ma2.config(state="disabled")  # 禁用 Ma2 输入
        #    self.entry_theta.config(state="normal")

    def calculate(self):
        try:
            gamma = float(self.entry_gamma.get())
            mode = self.mode_var.get()

            if mode == "已知 Ma1+θ → 求膨胀后参数":
                Ma1 = float(self.entry_ma1.get())
                theta = float(self.entry_theta.get())

                if Ma1 < 1:
                    messagebox.showerror("输入错误", "波前马赫数 Ma1 应该 ≥ 1")
                    return

                nu1 = self.prandtl_meyer(Ma1, gamma)
                nu2 = nu1 + theta
                Ma2 = self.inverse_prandtl_meyer(nu2, gamma)

            elif mode == "已知 Ma2+θ → 求膨胀前参数":
                Ma2 = float(self.entry_ma2.get())
                theta = float(self.entry_theta.get())

                if Ma2 < 1:
                    messagebox.showerror("输入错误", "波后马赫数 Ma2 应该 ≥ 1")
                    return

                nu2 = self.prandtl_meyer(Ma2, gamma)
                nu1 = nu2 - theta
                Ma1 = self.inverse_prandtl_meyer(nu1, gamma)

            elif mode == "已知 Ma1+Ma2 → 求 θ":
                Ma1 = float(self.entry_ma1.get())
                Ma2 = float(self.entry_ma2.get())

                if Ma1 < 1 or Ma2 < 1:
                    messagebox.showerror("输入错误", "波前和波后马赫数 Ma1, Ma2 应该 ≥ 1")
                    return

                nu1 = self.prandtl_meyer(Ma1, gamma)
                nu2 = self.prandtl_meyer(Ma2, gamma)
                theta = nu2 - nu1

            # elif mode == "膨胀波相交/反射":
            #     Ma1 = float(self.entry_ma1.get())
            #     theta = float(self.entry_theta.get())

            #     if Ma1 < 1:
            #         messagebox.showerror("输入错误", "波前马赫数 Ma1 应该 ≥ 1")
            #         return

            #     nu1 = self.prandtl_meyer(Ma1, gamma)
            #     nu2 = nu1 + theta
            #     Ma2 = self.inverse_prandtl_meyer(nu2, gamma)

            #     self.labels['result'].config(
            #         text=f"计算得到反射后马赫数 Ma2 ≈ {Ma2:.4f}"
            #     )
            #     self.labels['p_ratio'].config(text="")
            #     self.labels['t_ratio'].config(text="")
            #     self.labels['pm_angle'].config(text=f"ν1 = {nu1:.2f}°, ν2 = {nu2:.2f}°")
            #     return

            # 通用输出（非相交/反射模式）
            p_ratio = (1 + 0.5 * (gamma - 1) * Ma1 ** 2) ** (gamma / (gamma - 1)) / \
                      (1 + 0.5 * (gamma - 1) * Ma2 ** 2) ** (gamma / (gamma - 1))
            t_ratio = (1 + 0.5 * (gamma - 1) * Ma1 ** 2) / \
                      (1 + 0.5 * (gamma - 1) * Ma2 ** 2)

            self.labels['result'].config(text=f"Ma1 = {Ma1:.4f}, Ma2 = {Ma2:.4f}, θ = {theta:.2f}°")
            self.labels['p_ratio'].config(text=f"P2/P1 = {p_ratio:.4f}")
            self.labels['t_ratio'].config(text=f"T2/T1 = {t_ratio:.4f}")
            self.labels['pm_angle'].config(
                text=f"ν1 = {self.prandtl_meyer(Ma1, gamma):.2f}°, ν2 = {self.prandtl_meyer(Ma2, gamma):.2f}°")

        except ValueError:
            messagebox.showerror("输入错误", "请输入有效的数字")


class FlowWindow:
    """一维定常管流动计算器"""

    k: float = 1.4       # 比热比
    R: float = 287.0     # 气体常数 [J/(kg·K)]

    def __init__(self, master: tk.Tk) -> None:
        self.master = master
        master.title("一维定常管流动计算器 (pure-instance)")
        master.geometry("780x630")

        self.notebook = ttk.Notebook(master)
        self._build_tabs()
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)

    # ===== 通用气动函数 =====
    def crit_pressure_ratio(self) -> float:
        """p*/p0"""
        return (2 / (self.k + 1)) ** (self.k / (self.k - 1))

    def isen_T_ratio(self, M: float) -> float:
        """T/T0"""
        return 1 / (1 + (self.k - 1) / 2 * M**2)

    def isen_p_ratio(self, M: float) -> float:
        """p/p0"""
        return (1 + (self.k - 1) / 2 * M**2) ** (-self.k / (self.k - 1))

    def isen_area_ratio(self, M: float) -> float:
        """A/A*"""
        term = (2 / (self.k + 1)) * (1 + (self.k - 1) / 2 * M**2)
        return 1 / M * term ** ((self.k + 1) / (2 * (self.k - 1)))

    def mach_from_p_ratio(self, p_ratio: float) -> float:
        """由 p/p0 求亚音速 M"""
        if not 0 < p_ratio < 1:
            raise ValueError("p/p0 必须位于 (0,1)")
        term = p_ratio ** ((self.k - 1) / self.k)
        return math.sqrt(2 / (self.k - 1) * (1 / term - 1))

    def solve_mach_from_area(
        self, area_ratio: float, supersonic: bool,
        tol: float = 1e-6, max_iter: int = 100
    ) -> float:
        """由 A/A* 求 M (牛顿迭代)"""
        if area_ratio < 1:
            raise ValueError("A/A* ≥ 1")
        M = 2.0 if supersonic else 0.2
        for _ in range(max_iter):
            f = self.isen_area_ratio(M) - area_ratio
            dM = 1e-6
            df = (self.isen_area_ratio(M + dM) - area_ratio - f) / dM
            M_new = M - f / df
            if abs(M_new - M) < tol:
                return M_new
            M = M_new
        raise RuntimeError("Mach 迭代未收敛")


   
    # ===== Fanno =====
    def fanno_func(self, M: float) -> float:
        """4fL*/D"""
        term1 = (1 - M**2) / (self.k * M**2)
        term2 = (self.k + 1) / (2 * self.k) * math.log(
            ((self.k + 1) * M**2) / (2 + (self.k - 1) * M**2)
        )
        return term1 + term2

    def solve_fanno_exit_M(self, M1: float, fl_by_D: float,
                           tol: float = 1e-6) -> float:
        """给定 4fL/D 求出口 M"""
        low, high = 1e-6, 1 - 1e-6
        for _ in range(100):
            mid = 0.5 * (low + high)
            f_mid = self.fanno_func(mid) - fl_by_D
            if abs(f_mid) < tol:
                return mid
            if (self.fanno_func(low) - fl_by_D) * f_mid < 0:
                high = mid
            else:
                low = mid
        return mid

    # ===== Rayleigh (Heat-Addition) =====

    def ray_static_ratios(self, M: float) -> tuple[float, float]:
        """返回 Rayleigh 流中 p/p* 和 rho/rho*"""
        k = self.k
        p_pstar     = (k + 1) / (1 + k*M**2)
        rho_rhost   = (1 + k*M**2) / ((k + 1)*M**2)
        return p_pstar, rho_rhost

    def ray_total_pressure_ratio(self, M: float) -> float:
        """返回 Rayleigh 流中 p0/p0*"""
        k = self.k
        term1 = (k+1)/(1 + k*M**2)
        term2 = (2/(k+1)*(1 + (k-1)/2 * M**2))**(k/(k-1))
        return term1 * term2

    # ===== Rayleigh (Heat-Addition) =====
    def ray_T0_T0star(self, M: float) -> float:
        """修正：添加公式说明"""
        k = self.k
        numerator = 2 * (k + 1) * M**2
        denominator = (1 + k * M**2)**2
        return numerator / denominator * (1 + (k - 1) / 2 * M**2)

    def solve_rayleigh_exit_M(self, M1: float, T01: float, q: float) -> float:
        """修正：完整壅塞处理 + 动态Cp"""
        # 1. 动态计算比热容
        Cp = self.k * self.R / (self.k - 1)  # 使用气体参数计算       
        # 2. 计算临界总温 T0*
        T0_star = T01 / self.ray_T0_T0star(M1)    
        # 3. 计算临界加热量
        q_cr = Cp * (T0_star - T01)  
        # 4. 壅塞检查
        if q > q_cr:
            return 1.0  # 壅塞状态，出口M=1
        # 5. 计算出口总温
        T02 = T01 + q / Cp
        target = T02 / T0_star
        # 6. 分情况求解
        def residual(M):
            return self.ray_T0_T0star(M) - target
        
        # 亚音速入口 (0 < M1 < 1)
        if M1 < 1:
            try:
                # 尝试亚音速解 (0.01 < M < 0.99)
                return brentq(residual, 0.01, 0.99)
            except ValueError:
                # 尝试超音速解 (1.01 < M < 5)
                return brentq(residual, 1.01, 5)
        
        # 超音速入口 (M1 > 1)
        else:
            try:
                # 尝试超音速解
                return brentq(residual, 1.01, 5)
            except ValueError:
                # 尝试亚音速解
                return brentq(residual, 0.01, 0.99)

    # ===== 正常激波 =====
    def normal_shock_p_ratio(self, M1: float) -> float:
        """p₂/p₁"""
        return 1 + (2 * self.k) / (self.k + 1) * (M1**2 - 1)

    def normal_shock_p0_ratio(self, M1: float) -> float:
        """p0₂/p0₁"""
        k = self.k
        term1 = ((k + 1) * M1**2) / ((k - 1) * M1**2 + 2)
        term2 = (k + 1) / (2 * k * M1**2 - (k - 1))
        return term1 ** (k / (k - 1)) * term2 ** (1 / (k - 1))

    def solve_M1_from_sigma(self, sigma: float,
                            tol: float = 1e-6) -> float:
        """由 σ = p0₂/p0₁ 求激波前 M1"""
        low, high = 1.01, 5.0
        for _ in range(100):
            mid = 0.5 * (low + high)
            sigma_mid = self.normal_shock_p0_ratio(mid)
            if abs(sigma_mid - sigma) < tol:
                return mid
            if sigma_mid < sigma:
                high = mid
            else:
                low = mid
        return (low + high) * 0.5

    # ===== UI 构建 =====
    def _build_tabs(self) -> None:
        self.frames: dict[str, ttk.Frame] = {}
        for key, title in [
            ("conv", "收缩喷管"),
            ("laval", "拉瓦尔喷管"),
            ("fanno", "摩擦管流"),
            ("ray", "换热管流"),
            ("mass", "加质管流"),
        ]:
            frame = ttk.Frame(self.notebook)
            self.frames[key] = frame
            self.notebook.add(frame, text=title)

        self._build_conv_tab()
        self._build_laval_tab()
        self._build_fanno_tab()
        self._build_ray_tab()
        self._build_mass_tab()

    # -------- 1) 收缩喷管 --------
    def _build_conv_tab(self) -> None:
        f = self.frames["conv"]
        lf = ttk.LabelFrame(f, text="输入参数")
        lf.pack(fill="x", padx=10, pady=6)

        self.conv_entry: dict[str, ttk.Entry] = {}
        for i, (txt, key) in enumerate([
            ("入口总压 P0 (Pa):", "P0"),
            ("入口总温 T0 (K):", "T0"),
            ("出口面积 A (m²):", "A"),
            ("出口背压 Pb (Pa):", "Pb"),
        ]):
            ttk.Label(lf, text=txt).grid(row=i, column=0,
                                         sticky="e", padx=5, pady=4)
            e = ttk.Entry(lf)
            e.grid(row=i, column=1, sticky="w", padx=4, pady=4)
            self.conv_entry[key] = e

        ttk.Button(f, text="计算", command=self._calc_conv).pack(pady=6)

        resf = ttk.LabelFrame(f, text="计算结果")
        resf.pack(fill="x", padx=10, pady=6)
        self.conv_res = {
            "马赫": ttk.Label(resf, text="出口马赫数 = "),
            "流速": ttk.Label(resf, text="出口流速 (m/s) = "),
            "质量流量": ttk.Label(resf, text="质量流量 (kg/s) = "),
            "静压": ttk.Label(resf, text="出口静压 (Pa) = "),
        }
        for i, lbl in enumerate(self.conv_res.values()):
            lbl.grid(row=i, column=0, sticky="w", padx=6, pady=4)

    def _calc_conv(self) -> None:
        """收缩喷管计算"""
        try:
            P0 = float(self.conv_entry["P0"].get())
            T0 = float(self.conv_entry["T0"].get())
            A = float(self.conv_entry["A"].get())
            Pb = float(self.conv_entry["Pb"].get())
        except ValueError:
            messagebox.showerror("输入错误", "所有输入必须是数字!")
            return

        if any(v <= 0 for v in (P0, T0, A, Pb)):
            messagebox.showerror("输入错误", "所有输入必须为正值!")
            return

        try:
            P_crit = P0 * self.crit_pressure_ratio()
            if Pb <= P_crit:          # 壅塞
                Me = 1.0
                Te = T0 * 2 / (self.k + 1)
                Pe = P_crit
            else:                     # 非壅塞
                Pe = Pb
                Me = self.mach_from_p_ratio(Pe / P0)
                Te = T0 * self.isen_T_ratio(Me)

            rho_e = Pe / (self.R * Te)
            Ve = Me * math.sqrt(self.k * self.R * Te)
            mdot = rho_e * A * Ve

            self.conv_res["马赫"].config(text=f"出口马赫数 = {Me:.4f}")
            self.conv_res["流速"].config(text=f"出口流速 (m/s) = {Ve:.2f}")
            self.conv_res["质量流量"].config(text=f"质量流量 (kg/s) = {mdot:.4f}")
            self.conv_res["静压"].config(text=f"出口静压 (Pa) = {Pe:.2f}")
        except Exception as err:
            messagebox.showerror("计算错误", str(err))

    # -------- 2) 拉瓦尔喷管 --------
    def _build_laval_tab(self) -> None:
        f = self.frames["laval"]
        lf = ttk.LabelFrame(f, text="输入参数")
        lf.pack(fill="x", padx=10, pady=6)

        self.laval_entry: dict[str, ttk.Entry] = {}
        for i, (txt, key) in enumerate([
            ("入口总压 P0 (Pa):", "P0"),
            ("入口总温 T0 (K):", "T0"),
            ("喉部面积 A* (m²):", "At"),
            ("出口面积 Ae (m²):", "Ae"),
            ("出口背压 Pb (Pa):", "Pb"),
        ]):
            ttk.Label(lf, text=txt).grid(row=i, column=0,
                                         sticky="e", padx=5, pady=4)
            e = ttk.Entry(lf)
            e.grid(row=i, column=1, sticky="w", padx=4, pady=4)
            self.laval_entry[key] = e

        #ttk.Button(f, text="计算", command=self._calc_laval).pack(pady=6)

    # 在计算按钮下方添加绘图按钮
        ttk.Button(f, text="计算", command=self._calc_laval).pack(pady=6)
        ttk.Button(f, text="绘制压强分布", command=self._plot_pressure_distribution).pack(pady=6)  # 新增按钮

        resf = ttk.LabelFrame(f, text="计算结果")
        resf.pack(fill="x", padx=10, pady=6)
        self.laval_res = {
            "regime": ttk.Label(resf, text="工作状态 = "),
            "马赫": ttk.Label(resf, text="出口马赫数 = "),
            "质量流量": ttk.Label(resf, text="质量流量 (kg/s) = "),
            "静压": ttk.Label(resf, text="出口静压 (Pa) = "),
            "流速": ttk.Label(resf, text="出口流速 (m/s) = "),
            "shock_pos": ttk.Label(resf, text="激波位置面积 (m²) = "),
        }
        for i, lbl in enumerate(self.laval_res.values()):
            lbl.grid(row=i, column=0, sticky="w", padx=6, pady=2)


    def _calc_laval(self) -> None:
        """拉瓦尔喷管计算"""
        try:
            P0 = float(self.laval_entry["P0"].get())
            T0 = float(self.laval_entry["T0"].get())
            At = float(self.laval_entry["At"].get())
            Ae = float(self.laval_entry["Ae"].get())
            Pb = float(self.laval_entry["Pb"].get())
        except ValueError:
            messagebox.showerror("输入错误", "所有输入必须是数字!")
            return

        if any(v <= 0 for v in (P0, T0, At, Ae, Pb)):
            messagebox.showerror("输入错误", "所有输入必须为正值!")
            return

        try:
            area_ratio = Ae / At
            # 等熵出口 Mach
            Me_sup = self.solve_mach_from_area(area_ratio, True)
            Me_sub = self.solve_mach_from_area(area_ratio, False)

            p1_p0 = self.isen_p_ratio(Me_sup)          # 喉后超音速
            p3_p0 = self.isen_p_ratio(Me_sub)          # 喉后亚音速
            p2_p1 = self.normal_shock_p_ratio(Me_sup)  # 正常激波
            p2_p0 = p2_p1 * p1_p0
            pb_p0 = Pb / P0

            # — 工作状态
            if pb_p0 > p3_p0:
                regime = "亚临界"
            elif abs(pb_p0 - p3_p0) < 1e-5:
                regime = "临界"
            elif pb_p0 > p2_p0:
                regime = "内部激波"
            elif abs(pb_p0 - p2_p0) < 1e-5:
                regime = "出口激波"
            elif pb_p0 > p1_p0:
                regime = "过膨胀"
            elif abs(pb_p0 - p1_p0) < 1e-5:
                regime = "完全膨胀"
            else:
                regime = "欠膨胀"

            # — 质量流率（壅塞）
            T_star = T0 * 2 / (self.k + 1)
            P_star = P0 * (2 / (self.k + 1)) ** (self.k / (self.k - 1))
            rho_star = P_star / (self.R * T_star)
            V_star = math.sqrt(self.k * self.R * T_star)
            mdot_max = rho_star * At * V_star

            Me = pe = ve = shock_pos = None
            mdot = mdot_max

            if regime == "亚临界":
                Me = math.sqrt(((pb_p0) ** ((1 - self.k) / self.k) - 1)
                               * 2 / (self.k - 1))
                term = Me * (1 + (self.k - 1) / 2 * Me**2) ** (
                    -(self.k + 1) / (2 * (self.k - 1)))
                mdot = (P0 * Ae / math.sqrt(T0)) * math.sqrt(self.k / self.R) * term
                pe = Pb
            elif regime == "临界":
                Me = Me_sub
                pe = Pb
            elif regime == "内部激波":
                mdot = mdot_max
                C = mdot * math.sqrt(T0) / (Ae * Pb * math.sqrt(self.k / self.R))
                a, b, c = (self.k - 1) / 2, 1.0, -C**2
                Me = math.sqrt((-b + math.sqrt(b**2 - 4 * a * c)) / (2 * a))
                p0e = Pb * (1 + (self.k - 1) / 2 * Me**2) ** (self.k / (self.k - 1))
                sigma = p0e / P0
                M1 = self.solve_M1_from_sigma(sigma)
                qM1 = self.isen_area_ratio(M1)
                shock_pos = At * qM1
                pe = Pb
            elif regime == "出口激波":
                Me = Me_sup
                pe = P0 * p1_p0
            else:  # 过膨胀 / 完全膨胀 / 欠膨胀
                Me = Me_sup
                pe = P0 * p1_p0

            if Me is not None:
                Te = T0 / (1 + (self.k - 1) / 2 * Me**2)
                ve = Me * math.sqrt(self.k * self.R * Te)

            # — 结果显示
            self.laval_res["regime"].config(text=f"工作状态 = {regime}")
            self.laval_res["马赫"].config(
                text=f"出口马赫数 = {Me:.4f}" if Me else "出口马赫数 = N/A")
            self.laval_res["质量流量"].config(text=f"质量流量 (kg/s) = {mdot:.6f}")
            self.laval_res["静压"].config(
                text=f"出口静压 (Pa) = {pe:.2f}" if pe else "出口静压 (Pa) = N/A")
            self.laval_res["流速"].config(
                text=f"出口流速 (m/s) = {ve:.2f}" if ve else "出口流速 (m/s) = N/A")
            if shock_pos:
                self.laval_res["shock_pos"].config(
                    text=f"激波位置面积 (m²) = {shock_pos:.6e}")
            else:
                self.laval_res["shock_pos"].config(text="激波位置面积 = N/A")
        except Exception as err:
            messagebox.showerror("计算错误", str(err))

    def _plot_pressure_distribution(self) -> None:
        """绘制静压随喷管位置的变化曲线（包含出口后调整段）"""
        try:
            # ===== 1. 获取输入参数 =====
            P0 = float(self.laval_entry["P0"].get())  # 总压 (Pa)
            T0 = float(self.laval_entry["T0"].get())  # 总温 (K)
            At = float(self.laval_entry["At"].get())  # 喉部面积 (m²)
            Ae = float(self.laval_entry["Ae"].get())  # 出口面积 (m²)
            Pb = float(self.laval_entry["Pb"].get())  # 背压 (Pa)
            
            # ===== 2. 计算关键参数 =====
            area_ratio = Ae / At
            Me_sup = self.solve_mach_from_area(area_ratio, True)   # 超音速出口马赫数
            Me_sub = self.solve_mach_from_area(area_ratio, False)  # 亚音速出口马赫数
            
            p1_p0 = self.isen_p_ratio(Me_sup)       # 完全膨胀压强比
            p3_p0 = self.isen_p_ratio(Me_sub)       # 亚临界压强比
            p2_p1 = self.normal_shock_p_ratio(Me_sup) # 正激波压强比
            p2_p0 = p2_p1 * p1_p0                  # 出口激波压强比
            
            pb_p0 = Pb / P0                         # 背压比
            crit_ratio = self.crit_pressure_ratio()  # 临界压力比
            P_crit = P0 * crit_ratio                # 临界压力 (Pa)

            # ===== 3. 确定喷管工作状态 =====
            if pb_p0 > p3_p0:
                regime = "亚临界"
            elif abs(pb_p0 - p3_p0) < 1e-5:
                regime = "临界"
            elif pb_p0 > p2_p0:
                regime = "内部激波"
            elif abs(pb_p0 - p2_p0) < 1e-5:
                regime = "出口激波"
            elif pb_p0 > p1_p0:
                regime = "过膨胀"
            elif abs(pb_p0 - p1_p0) < 1e-5:
                regime = "完全膨胀"
            else:
                regime = "欠膨胀"

            # ===== 4. 创建喷管几何模型 =====
            num_points = 200  # 增加点数使曲线更平滑
            x_nozzle = np.linspace(0, 1, num_points)  # 喷管内位置 (0=入口, 0.5=喉部, 1=出口)
            
            # 收缩段 (0-0.5)
            conv_points = num_points // 2
            x_conv = x_nozzle[:conv_points]
            A_conv = At * (1 + 2*(0.5 - x_conv)**2)  # 二次曲线收缩
            
            # 扩张段 (0.5-1)
            exp_points = num_points - conv_points
            x_exp = x_nozzle[conv_points:]
            A_exp = At + (Ae - At) * (x_exp - 0.5)/0.5  # 线性扩张
            
            # 组合喷管面积分布
            A_x = np.concatenate([A_conv, A_exp])
            area_ratios = A_x / At  # 面积比分布

           
            # ===== 5. 计算喷管内流动参数 =====
            M_x = np.zeros(num_points)  # 马赫数分布
            p_x = np.zeros(num_points)   # 静压分布 (Pa)

            if regime in ["亚临界"]:
                # 亚临界状态整个喷管都是亚音速
                γ = self.k
                # 计算出口马赫数
                Me = math.sqrt(((P0/Pb)**((γ-1)/γ) - 1) * 2/(γ-1))
                Te = T0 / (1 + (γ-1)/2 * Me**2)
                ρe = Pb / (self.R * Te)
                Ve = Me * math.sqrt(γ * self.R * Te)
                mdot = ρe * Ae * Ve
                
                # 计算临界面积A*
                term = (2/(γ+1))**((γ+1)/(2*(γ-1)))
                A_star = mdot * math.sqrt(T0) / (P0 * math.sqrt(γ/self.R) * term)
                
                # 用A*计算所有截面的亚音速流动
                for i in range(num_points):
                    ar_real = A_x[i] / A_star
                    M_x[i] = self.solve_mach_from_area(ar_real, False)  # 始终亚音速解
                    p_x[i] = P0 * self.isen_p_ratio(M_x[i])

            else:
                # 收缩段总是亚音速 (0-0.5)
                for i in range(conv_points):
                    if area_ratios[i] > 1:
                        M_x[i] = self.solve_mach_from_area(area_ratios[i], False)
                    else:
                        M_x[i] = 0.1  # 入口处小马赫数
                    p_x[i] = P0 * self.isen_p_ratio(M_x[i])
                
                # 喉部 (x=0.5) 总是音速
                M_x[conv_points] = 1.0
                p_x[conv_points] = P_crit
                        
            # ===== 6. 处理扩张段不同流动状态 =====
            if regime in ["临界"]:
                # 整个扩张段亚音速
                for i in range(conv_points+1, num_points):
                    M_x[i] = self.solve_mach_from_area(area_ratios[i], False)
                    p_x[i] = P0 * self.isen_p_ratio(M_x[i])
            elif regime in ["亚临界"]:
                # 整个扩张段亚音速
                for i in range(num_points):
                    ar_real = A_x[i] / A_star
                    M_x[i] = self.solve_mach_from_area(ar_real, False)  # 始终亚音速解
                    p_x[i] = P0 * self.isen_p_ratio(M_x[i])
                # p_total[:num_points] = p_x[:num_points]
                # M_total[:num_points] = M_x[:num_points]
                    
            elif regime == "内部激波":
                # 计算激波位置
                p0e = Pb * (1 + (self.k - 1)/2 * Me_sub**2)**(self.k/(self.k-1))
                sigma = p0e / P0  # 总压恢复系数
                M1 = self.solve_M1_from_sigma(sigma)  # 激波前马赫数
                
                # 计算激波位置对应的面积比
                A_shock_ratio = self.isen_area_ratio(M1)
                A_shock = At * A_shock_ratio
                
                # 确保激波在喷管内 (限制在喉部之后，出口之前)
                A_shock = min(A_shock, Ae * 0.99)
                A_shock = max(A_shock, At * 1.01)
                
                # 在扩张段寻找激波位置索引
                shock_idx = conv_points + np.argmin(np.abs(A_x[conv_points:] - A_shock))
                shock_idx = max(conv_points+1, min(shock_idx, num_points-2))
                
                # 激波前超音速段 (喉部到激波位置)
                for i in range(conv_points+1, shock_idx+1):
                    M_x[i] = self.solve_mach_from_area(area_ratios[i], True)
                    p_x[i] = P0 * self.isen_p_ratio(M_x[i])
                
                # 计算激波后参数
                M2 = self.normal_shock_mach2(M1)  # 激波后马赫数
                p02 = p0e  # 激波后总压
                
                # 激波位置 (使用激波后总压)
                M_x[shock_idx] = M2
                p_x[shock_idx] = p02 * self.isen_p_ratio(M2)
                
                # 激波后亚音速段 (激波位置到出口)
                for i in range(shock_idx+1, num_points):
                    M_x[i] = self.solve_mach_from_area(area_ratios[i], False)
                    p_x[i] = p02 * self.isen_p_ratio(M_x[i])
                    
            else:  # 超音速流动 (出口激波/过膨胀/欠膨胀/完全膨胀)
                for i in range(conv_points+1, num_points):
                    M_x[i] = self.solve_mach_from_area(area_ratios[i], True)
                    p_x[i] = P0 * self.isen_p_ratio(M_x[i])

            # 出口参数
            p_exit = p_x[-1]
            M_exit = M_x[-1]
            
            # ===== 7. 计算出口后调整段 =====
            adjust_points = 50  # 出口后点数
            total_points = num_points + adjust_points
            x_total = np.linspace(0, 2, total_points)  # 0-1喷管内，1-2喷管外
            p_total = np.zeros(total_points)
            M_total = np.zeros(total_points)
            
            # 复制喷管内数据
            p_total[:num_points] = p_x
            M_total[:num_points] = M_x
            
            # 根据流动状态处理出口后调整
            if regime in ["亚临界", "临界", "完全膨胀"]:
                # 出口压强等于背压，不需要调整
                p_total[num_points:] = Pb
                M_total[num_points:] = M_exit
                
            elif regime in ["内部激波"]:
                # 平滑过渡到背压 (线性插值)
                for i in range(adjust_points):
                    t = i / (adjust_points - 1)
                    p_total[num_points + i] = p_exit + (Pb - p_exit) * t
                    M_total[num_points + i] = M_exit
            
            elif regime == "出口激波":
                # 计算激波后马赫数 (正确公式)
                numerator = (self.k - 1) * M_exit**2 + 2
                denominator = 2 * self.k * M_exit**2 - (self.k - 1)
                M_after_shock = math.sqrt(numerator / denominator)
                
                # 设置出口后所有点
                p_total[num_points:] = Pb
                M_total[num_points:] = M_after_shock  # 整个调整段保持恒定亚音速
                    
            else:  # 过膨胀或欠膨胀
                # 指数衰减调整
                adjustment_speed = 0.3 if regime == "过膨胀" else 0.1
                x_adjust = np.linspace(1, 2, adjust_points)
                delta_p = p_exit - Pb
                
                # 过膨胀: 压强从低到高上升至背压
                # 欠膨胀: 压强从高到低下降至背压
                if regime == "过膨胀":
                    p_adjust = Pb + delta_p * np.exp(-adjustment_speed * (x_adjust - 1) * 10)
                else:  # 欠膨胀
                    p_adjust = Pb + delta_p * (1 - np.exp(-adjustment_speed * (x_adjust - 1) * 5))
                
                # 马赫数调整
                M_adjust = np.zeros(adjust_points)
                if regime == "过膨胀":
                    # 过膨胀: 压缩波系，马赫数下降
                    M_adjust = M_exit * np.exp(-0.2 * (x_adjust - 1) * 2)
                    
                else:
                    # 欠膨胀: 膨胀波系，马赫数上升
                    #M_adjust = M_exit + (Me_sup - M_exit) * (1 - np.exp(-adjustment_speed * (x_adjust - 1) * 6))
                    M_adjust = M_exit + (Me_sup * 1.2 - M_exit) * (1 - np.exp(-0.4 * (x_adjust - 1) * 6))
                
                p_total[num_points:] = p_adjust
                M_total[num_points:] = M_adjust

            # ===== 8. 绘制图形 =====
            fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                        gridspec_kw={'height_ratios': [2, 1]})
            
            # 压强分布图
            ax1.plot(x_total, p_total, 'b-', linewidth=2, label=f'静压分布 ({regime}状态)')
            ax1.axhline(y=P_crit, color='r', linestyle='--', label=f'临界压力 ({P_crit/1000:.1f} kPa)')
            ax1.axhline(y=Pb, color='g', linestyle='-.', label=f'背压 ({Pb/1000:.1f} kPa)')
            ax1.axvline(x=1.6, color='k', linestyle='--', alpha=0.7, label='喷管出口')
            ax1.scatter([0.8], [p_total[conv_points]], color='r', s=100, zorder=5, label='喉部 (M=1)')
            ax1.scatter([1.6], [p_exit], color='b', s=80, zorder=5, label='喷管出口')
            ax1.scatter([2.0], [Pb], color='g', s=80, zorder=5, label='远下游背压')
            
            # 内部激波特殊标注
            if regime == "内部激波":
                shock_x = x_total[shock_idx]
                shock_p = p_total[shock_idx]
                ax1.scatter([shock_x], [shock_p], color='m', s=100, zorder=5, label='激波位置')
                ax1.axvline(x=shock_x, color='m', linestyle=':', alpha=0.7)
                
                # 计算并显示激波强度
                pressure_jump = p_x[shock_idx] / p_x[shock_idx-1]
                ax1.annotate(f'激波强度: {pressure_jump:.1f}倍',
                            (shock_x, (p_x[shock_idx] + p_x[shock_idx-1])/2),
                            ha='center', va='center', fontsize=9,
                            bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='m', lw=1))
            
            # 区域标注
            ax1.text(0.25, max(p_total)*0.8, "亚音速区", ha='center', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8))
            ax1.text(0.8, max(p_total)*0.8, "超音速区", ha='center', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8))
            ax1.text(1.8, max(p_total)*0.8, "出口后调整区", ha='center', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8))
            
            # 设置压强图属性
            ax1.set_title(f"拉瓦尔喷管静压分布 ({regime}状态)")
            ax1.set_ylabel("静压 (Pa)")
            ax1.set_ylim(min(p_total.min(), Pb, P_crit)*0.8, max(p_total.max(), Pb, P0)*1.2)
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.legend(loc='best')
            
            # 添加kPa坐标轴
            ax2 = ax1.twinx()
            ax2.set_ylabel('静压 (kPa)', color='purple')
            ax2.tick_params(axis='y', labelcolor='purple')
            
            # 马赫数分布图
            ax3.plot(x_total, M_total, 'r-', linewidth=1.5, label='马赫数')
            ax3.axvline(x=1.6, color='k', linestyle='--', alpha=0.7)
            ax3.axhline(y=1.0, color='gray', linestyle=':', alpha=0.7, label='音速')
            ax3.scatter([0.8], [1.0], color='r', s=60, zorder=5)
            ax3.scatter([1.6], [M_exit], color='b', s=50, zorder=5)
            
            # 内部激波马赫数标注
            if regime == "内部激波":
                ax3.scatter([shock_x], [M_total[shock_idx]], color='m', s=80, zorder=5)
                ax3.annotate(f'M={M_x[shock_idx-1]:.2f}→{M_x[shock_idx]:.2f}',
                            (shock_x, (M_x[shock_idx] + M_x[shock_idx-1])/2),
                            ha='center', va='bottom')
            
            # 设置马赫数图属性
            ax3.set_xlabel("归一化位置 (0=入口, 0.8=喉部, 1.6=出口, 2=下游)")
            ax3.set_ylabel("马赫数")
            ax3.set_ylim(0, max(M_total)*1.2)
            ax3.grid(True, linestyle='--', alpha=0.5)
            ax3.legend(loc='best')
            
            plt.tight_layout()
            plt.show()

        except Exception as err:
            messagebox.showerror("绘图错误", f"无法生成压强分布图: {str(err)}")

    # ===== 新增辅助函数 =====
    def normal_shock_mach2(self, M1):
        """计算正激波后马赫数"""
        k = self.k
        numerator = (k-1)*M1**2 + 2
        denominator = 2*k*M1**2 - (k-1)
        return math.sqrt(numerator / denominator)


    # -------- 3) Fanno --------
    def _build_fanno_tab(self) -> None:
        f = self.frames["fanno"]
        lf = ttk.LabelFrame(f, text="输入参数")
        lf.pack(fill="x", padx=10, pady=6)

        self.fanno_entry: dict[str, ttk.Entry] = {}
        for i, (txt, key) in enumerate([
            ("管长 L (m):", "L"),
            ("直径 D (m):", "D"),
            ("摩擦系数 f:", "f"),
            ("入口马赫数 M1:", "M1"),
        ]):
            ttk.Label(lf, text=txt).grid(row=i, column=0,
                                         sticky="e", padx=5, pady=4)
            e = ttk.Entry(lf)
            e.grid(row=i, column=1, sticky="w", padx=4, pady=4)
            self.fanno_entry[key] = e

        #ttk.Button(f, text="计算", command=self._calc_fanno).pack(pady=6)
        btn_frame = ttk.Frame(f)
        btn_frame.pack(pady=6)
        ttk.Button(btn_frame, text="计算", command=self._calc_fanno).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="绘制范诺线", command=self._plot_fanno_line).pack(side="left", padx=5)

        resf = ttk.LabelFrame(f, text="计算结果")
        resf.pack(fill="x", padx=10, pady=6)
        self.fanno_res = {
            "出口马赫数": ttk.Label(resf, text="出口马赫数 = "),
            "4fL*/D_max": ttk.Label(resf, text="4fL*/D_max = "),
            "最大管长": ttk.Label(resf, text="最大管长 L_max = "),  # 新增最大管长显示 
            "阻塞状态": ttk.Label(resf, text="阻塞状态 = "),
        }
        for i, lbl in enumerate(self.fanno_res.values()):
            lbl.grid(row=i, column=0, sticky="w", padx=6, pady=4)

    def fanno_func(self, M, gamma=1.4):
        """4fL*/D 计算"""
        if M <= 0 or M >= 1:
            return float('inf')
        term1 = (1 - M**2) / (gamma * M**2)
        term2 = ((gamma + 1) / (2 * gamma)) * math.log(
            ((gamma + 1) * M**2) / 
            (2 * (1 + (gamma - 1) * M**2 / 2))
        )
        return term1 + term2

    def solve_fanno_exit_M(self, M1, fl, gamma=1.4, max_iter=100, tol=1e-6):
        """求解Fanno流出口马赫数"""
        fl_star_M1 = self.fanno_func(M1, gamma)
        fl_star_M2 = fl_star_M1 - fl
        
        # fl_star_M2<0 表示阻塞状态
        if fl_star_M2 < 0:
            return 1.0
        
        # 牛顿迭代法求解M2 (0 < M2 < 1)
        low, high = 1e-5, 1.0 - 1e-5
        for _ in range(max_iter):
            M2 = (low + high) / 2
            f_current = self.fanno_func(M2, gamma)
            
            if abs(f_current - fl_star_M2) < tol:
                return M2
                
            if f_current > fl_star_M2:
                low = M2
            else:
                high = M2
                
        return (low + high) / 2

    def _calc_fanno(self) -> None:
        """Fanno流计算"""
        try:
            L = float(self.fanno_entry["L"].get())
            D = float(self.fanno_entry["D"].get())
            f_val = float(self.fanno_entry["f"].get())
            M1 = float(self.fanno_entry["M1"].get())
        except ValueError:
            messagebox.showerror("输入错误", "所有输入必须是数字!")
            return

        if any(v <= 0 for v in (L, D, f_val)) or not (0 < M1 < 1):
            messagebox.showerror("输入错误", "参数范围错误: L,D,f > 0, 0 < M1 < 1")
            return

        try:
            fl = 4 * f_val * L / D  
            fl_max = self.fanno_func(M1)
            
            if fl > fl_max:  # 阻塞状态
                self.fanno_res["阻塞状态"].config(text="阻塞 (Choked)", foreground="red")
                M2 = 1.0
            else:
                self.fanno_res["阻塞状态"].config(text="非阻塞 (Not Choked)", foreground="green")
                M2 = self.solve_fanno_exit_M(M1, fl)
                
            self.fanno_res["出口马赫数"].config(text=f"出口马赫数 = {M2:.4f}")
            self.fanno_res["4fL*/D_max"].config(text=f"4fL*/D_max = {fl_max:.4f}")

             # 新增最大管长的计算和显示
            L_max = (fl_max * D) / (4 * f_val)  # 最大管长计算
            self.fanno_res["最大管长"].config(text=f"最大管长 L_max = {L_max:.4f} m")
            
        except Exception as err:
            messagebox.showerror("计算错误", str(err))

    def _plot_fanno_line(self) -> None:
        """绘制范诺线（焓熵图），包括亚音速和超音速分支"""
        try:
            # 获取入口马赫数
            M1 = float(self.fanno_entry["M1"].get())
            
            if M1 <= 0:
                raise ValueError("入口马赫数必须大于0")
            
            # 气体常数
            gamma = self.k
            R = self.R
            Cp = gamma * R / (gamma - 1)  # 比定压热容
            
            # 创建马赫数数组（亚音速和超音速分支）
            M_sub = np.linspace(0.01, 0.99, 200)  # 亚音速分支
            M_sup = np.linspace(1.01, 5.0, 200)   # 超音速分支
            
            # 计算参考状态（临界状态 M=1）
            T_star = 1.0  # 无量纲参考温度
            p_star = 1.0  # 无量纲参考压力
            
            # 计算亚音速分支的熵增和温度比
            delta_s_sub = []
            T_ratios_sub = []
            
            for M in M_sub:
                # Fanno流温度比 T/T*
                T_ratio = (gamma + 1) / (2 + (gamma - 1) * M**2)
                T_ratios_sub.append(T_ratio)
                
                # Fanno流压力比 p/p*
                p_ratio = 1 / M * np.sqrt((gamma + 1) / (2 + (gamma - 1) * M**2))
                
                # 熵增 Δs = Cp*ln(T/T*) - R*ln(p/p*)
                ds = Cp * np.log(T_ratio) - R * np.log(p_ratio)
                delta_s_sub.append(ds)
            
            # 计算超音速分支的熵增和温度比
            delta_s_sup = []
            T_ratios_sup = []
            
            for M in M_sup:
                # Fanno流温度比 T/T*
                T_ratio = (gamma + 1) / (2 + (gamma - 1) * M**2)
                T_ratios_sup.append(T_ratio)
                
                # Fanno流压力比 p/p*
                p_ratio = 1 / M * np.sqrt((gamma + 1) / (2 + (gamma - 1) * M**2))
                
                # 熵增 Δs = Cp*ln(T/T*) - R*ln(p/p*)
                ds = Cp * np.log(T_ratio) - R * np.log(p_ratio)
                delta_s_sup.append(ds)
            
            # 计算临界点（M=1）
            T_ratio_crit = (gamma + 1) / (2 + (gamma - 1) * 1**2)
            p_ratio_crit = 1 / 1 * np.sqrt((gamma + 1) / (2 + (gamma - 1) * 1**2))
            ds_crit = Cp * np.log(T_ratio_crit) - R * np.log(p_ratio_crit)
            
            # 计算入口点
            T1_ratio = (gamma + 1) / (2 + (gamma - 1) * M1**2)
            p1_ratio = 1 / M1 * np.sqrt((gamma + 1) / (2 + (gamma - 1) * M1**2))
            ds1 = Cp * np.log(T1_ratio) - R * np.log(p1_ratio)
            
            # 创建图形
            plt.figure(figsize=(10, 7))
            
            # 绘制亚音速分支
            plt.plot(delta_s_sub, T_ratios_sub, 'b-', linewidth=1.5, label="亚音速分支")
            
            # 绘制超音速分支
            plt.plot(delta_s_sup, T_ratios_sup, 'r-', linewidth=1.5, label="超音速分支")
            
            # 标记临界点（M=1）
            plt.scatter([ds_crit], [T_ratio_crit], color='green', s=100, zorder=5, 
                    label="临界状态 (M=1)")
            plt.annotate('M=1.0', 
                        (ds_crit, T_ratio_crit), 
                        textcoords="offset points", 
                        xytext=(10, 10), 
                        ha='left',
                        fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.8))
            
            # 标记入口点
            color = 'blue' if M1 < 1 else 'red'
            plt.scatter([ds1], [T1_ratio], color=color, s=100, zorder=5, 
                    label=f"入口状态 (M={M1:.2f})")
            plt.annotate(f'M={M1:.2f}', 
                        (ds1, T1_ratio), 
                        textcoords="offset points", 
                        xytext=(10, -10) if M1 < 1 else (-60, 10), 
                        ha='left' if M1 < 1 else 'right',
                        fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.8))
            
            # 添加箭头指示流动方向
            if M1 < 1:
                # 亚音速流动方向（向M=1加速）
                idx = len(delta_s_sub) // 3
                plt.annotate('', 
                            xy=(delta_s_sub[idx], T_ratios_sub[idx]), 
                            xytext=(delta_s_sub[idx-10], T_ratios_sub[idx-10]),
                            arrowprops=dict(arrowstyle="->", color='blue', lw=1.5))
                plt.text(delta_s_sub[idx-15], T_ratios_sub[idx-15]-0.02, 
                        '流动方向', color='blue', fontsize=9)
            else:
                # 超音速流动方向（向M=1减速）
                idx = len(delta_s_sup) // 3
                plt.annotate('', 
                            xy=(delta_s_sup[idx], T_ratios_sup[idx]), 
                            xytext=(delta_s_sup[idx-10], T_ratios_sup[idx-10]),
                            arrowprops=dict(arrowstyle="->", color='red', lw=1.5))
                plt.text(delta_s_sup[idx-15], T_ratios_sup[idx-15]+0.02, 
                        '流动方向', color='red', fontsize=9)
            
            # 设置图形属性
            plt.title(f"范诺线 (Fanno Line) - 焓熵图 (γ={gamma:.2f})")
            plt.xlabel(r"熵增 $\Delta s$ [J/(kg·K)]")
            plt.ylabel(r"温度比 $T/T^*$")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(loc='best')
            
            # 添加第二Y轴显示焓值（h = Cp*T）
            ax2 = plt.gca().twinx()
            h_min = min(min(T_ratios_sub), min(T_ratios_sup)) * Cp
            h_max = max(max(T_ratios_sub), max(T_ratios_sup)) * Cp
            ax2.set_ylim(h_min, h_max)
            ax2.set_ylabel(r"焓 $h = C_p T$ [J/kg]")
            
            # 添加流动过程说明
            plt.figtext(0.5, 0.01, 
                    "注：范诺线表示绝热摩擦管流中总焓守恒的过程\n"
                    "临界点(M=1)处熵达到最大值，流动在临界点壅塞",
                    ha="center", fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.5", fc='white', alpha=0.7))
            
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.12)  # 为底部文本留出空间
            plt.show()
            
        except ValueError as ve:
            messagebox.showerror("输入错误", str(ve))
        except Exception as e:
            messagebox.showerror("绘图错误", f"绘制范诺线时出错: {str(e)}")


    # # -------- 4) Rayleigh --------

    # def _build_ray_tab(self):
    #     """创建换热管流计算标签页"""
    #     f = self.frames["ray"]
    #     # 输入参数框架
    #     input_frame = ttk.LabelFrame(f, text="输入参数")
    #     input_frame.pack(pady=10, padx=10, fill="x")
        
    #     # 创建输入字段
    #     ttk.Label(input_frame, text="进口马赫数 M1:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
    #     self.entry_M1_heat = ttk.Entry(input_frame)
    #     self.entry_M1_heat.grid(row=0, column=1, padx=5, pady=5)
        
    #     ttk.Label(input_frame, text="进口总温 T01 (K):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
    #     self.entry_T01_heat = ttk.Entry(input_frame)
    #     self.entry_T01_heat.grid(row=1, column=1, padx=5, pady=5)
        
    #     ttk.Label(input_frame, text="热交换量 q (kJ/kg):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
    #     self.entry_q_heat = ttk.Entry(input_frame)
    #     self.entry_q_heat.grid(row=2, column=1, padx=5, pady=5)
        
    #     # 计算按钮
    #     btn_calc = ttk.Button(f, text="计算", command=self.calculate_heat_transfer_flow)
    #     btn_calc.pack(pady=10)
        
    #     # 结果显示框架
    #     result_frame = ttk.LabelFrame(f, text="计算结果")
    #     result_frame.pack(pady=10, padx=10, fill="x")
        
    #     self.result_labels_heat = {
    #         'Mach2': ttk.Label(result_frame, text="出口马赫数 M2 = "),
    #         'T02': ttk.Label(result_frame, text="出口总温 T02 (K) = "),
    #         'PressureRatio': ttk.Label(result_frame, text="总压比 P02/P01 = ")
    #     }
        
    #     for i, (_, label) in enumerate(self.result_labels_heat.items()):
    #         label.grid(row=i, column=0, sticky="w", padx=10, pady=5)

    # def solve_rayleigh_mach(self, M1, T01, T02, gamma=1.4, Cp=1005):
    #     """求解瑞利流的出口马赫数"""
    #     # 定义目标函数
    #     def target(M2):
    #         term1 = (1 + gamma*M1**2)**2 * M2**2 * (1 + (gamma-1)/2*M2**2)
    #         term2 = (1 + gamma*M2**2)**2 * M1**2 * (1 + (gamma-1)/2*M1**2)
    #         T02_calc = T01 * term1 / term2
    #         return T02_calc - T02
        
    #     # 使用二分法求解
    #     M_low = 0.01
    #     M_high = 1.0 if M1 < 1.0 else 5.0  # 根据进口马赫数选择范围
        
    #     tolerance = 1e-6
    #     max_iter = 100
        
    #     for _ in range(max_iter):
    #         M_mid = (M_low + M_high) / 2
    #         f_mid = target(M_mid)
            
    #         if abs(f_mid) < tolerance:
    #             return M_mid
                
    #         if f_mid > 0:
    #             M_high = M_mid
    #         else:
    #             M_low = M_mid
        
    #     return (M_low + M_high) / 2

    # def calculate_heat_transfer_flow(self):
    #     try:
    #         # 获取输入参数
    #         M1 = float(self.entry_M1_heat.get())
    #         T01 = float(self.entry_T01_heat.get())
    #         q = float(self.entry_q_heat.get()) * 1000  # 转换为J/kg
    #         gamma = 1.4
    #         Cp = 1005  # 空气定压比热容 J/kg·K
            
    #         # 计算出口总温
    #         T02 = T01 + q / Cp
            
    #         # 计算出口马赫数
    #         M2 = self.solve_rayleigh_mach(M1, T01, T02, gamma, Cp)
            
    #         # 计算总压比
    #         P02_P01 = ( (1 + gamma*M1**2) / (1 + gamma*M2**2) ) * (
    #             (1 + (gamma-1)/2*M2**2) / (1 + (gamma-1)/2*M1**2) )**(gamma/(gamma-1))
            
    #         # 更新结果
    #         self.result_labels_heat['Mach2'].config(text=f"出口马赫数 M2 = {M2:.4f}")
    #         self.result_labels_heat['T02'].config(text=f"出口总温 T02 (K) = {T02:.2f}")
    #         self.result_labels_heat['PressureRatio'].config(text=f"总压比 P02/P01 = {P02_P01:.4f}")
            
    #     except ValueError:
    #         messagebox.showerror("输入错误", "请输入有效的数字参数")
    #     except Exception as e:
    #         messagebox.showerror("计算错误", f"计算过程中发生错误: {str(e)}")

    def _build_ray_tab(self) -> None:
        f = self.frames["ray"]
        lf = ttk.LabelFrame(f, text="输入参数")
        lf.pack(fill="x", padx=10, pady=6)
        
        self.ray_entry: dict[str, ttk.Entry] = {}
        params = [
            ("入口马赫数 M1:", "M1"),
            ("入口总温 T01 (K):", "T01"),
            ("热交换量 q (kJ/kg):", "q"),
        ]
        
        for i, (txt, key) in enumerate(params):
            ttk.Label(lf, text=txt).grid(row=i, column=0, sticky="e", padx=5, pady=4)
            e = ttk.Entry(lf)
            e.grid(row=i, column=1, sticky="w", padx=4, pady=4)
            self.ray_entry[key] = e
        
        btn_frame = ttk.Frame(f)
        btn_frame.pack(pady=6)
        ttk.Button(btn_frame, text="计算", command=self._calc_ray).pack(side="left", padx=5)
        
        resf = ttk.LabelFrame(f, text="计算结果")
        resf.pack(fill="x", padx=10, pady=6)
        
        self.ray_res = {
            "出口马赫数": ttk.Label(resf, text="出口马赫数 = "),
            "出口总温": ttk.Label(resf, text="出口总温 T02 (K) = "),
            "总压比": ttk.Label(resf, text="总压比 P02/P01 = "),
            "阻塞状态": ttk.Label(resf, text="阻塞状态 = "),
        }
        
        for i, lbl in enumerate(self.ray_res.values()):
            lbl.grid(row=i, column=0, sticky="w", padx=6, pady=4)

    def _calc_ray(self) -> None:
        """Rayleigh流计算（换热管流）"""
        try:
            # 获取输入参数
            M1 = float(self.ray_entry["M1"].get())
            T01 = float(self.ray_entry["T01"].get())
            q_kj = float(self.ray_entry["q"].get())
            q = q_kj * 1000  # 转换为J/kg
            
            # 验证输入范围
            if M1 <= 0:
                raise ValueError("入口马赫数必须大于0")
            if T01 <= 0:
                raise ValueError("入口总温必须大于0")
            
            # 计算出口参数
            M2 = self.solve_rayleigh_exit_M(M1, T01, q)
            Cp = self.k * self.R / (self.k - 1)
            T02 = T01 + q / Cp
            
            # 计算总压比 (Rayleigh流总压比公式)
            k = self.k
            term1 = (1 + k * M1**2) / (1 + k * M2**2)
            term2 = ((1 + (k-1)/2 * M2**2) / (1 + (k-1)/2 * M1**2)) ** (k/(k-1))
            p02_p01 = term1 * term2
            
            # 确定阻塞状态
            choked = "是" if M2 == 1.0 else "否"
            
            # 计算临界加热量（修正后的部分）
            T0_star_ratio = self.ray_T0_T0star(M1)
            T0_star = T01 / T0_star_ratio  # 修正后的临界总温计算
            q_critical = max(0.0, Cp * (T0_star - T01) / 1000)  # 确保非负
            
            # 更新结果
            self.ray_res["出口马赫数"].config(text=f"出口马赫数 = {M2:.4f}")
            self.ray_res["出口总温"].config(text=f"出口总温 T02 (K) = {T02:.2f}")
            self.ray_res["总压比"].config(text=f"总压比 P02/P01 = {p02_p01:.4f}")
            
            self.ray_res["阻塞状态"].config(text=f"阻塞状态 = {choked}")

            # 新增临界加热量显示
            if "临界加热量" not in self.ray_res:
                new_label = ttk.Label(self.ray_res["阻塞状态"].master, text="临界加热量 q_cr (kJ/kg) = ")
                new_label.grid(row=len(self.ray_res), column=0, sticky="w", padx=6, pady=4)
                self.ray_res["临界加热量"] = new_label

            self.ray_res["临界加热量"].config(text=f"临界加热量 q_cr (kJ/kg) = {q_critical:.4f}")
            
        except ValueError as ve:
            messagebox.showerror("输入错误", str(ve))
        except Exception as err:
            messagebox.showerror("计算错误", f"Rayleigh流计算错误: {str(err)}")

    def ray_T0_T0star(self, M: float) -> float:
        """计算 Rayleigh 流总温比 T0/T0*"""
        k = self.k
        numerator = 2 * (k + 1) * M**2
        denominator = (1 + k * M**2)**2
        return numerator / denominator * (1 + (k - 1) / 2 * M**2)

    # # -------- 5) 变流量加质管流 --------
    def _build_mass_tab(self):
        """创建变流量加质管流计算标签页"""
        f = self.frames["mass"]
        # 输入参数框架
        input_frame = ttk.LabelFrame(f, text="输入参数")
        input_frame.pack(pady=10, padx=10, fill="x")
        
        # 创建输入字段
        ttk.Label(input_frame, text="进口质量流量 ṁ1 (kg/s):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.entry_m1_mass = ttk.Entry(input_frame)
        self.entry_m1_mass.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="进口速度 V1 (m/s):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.entry_V1_mass = ttk.Entry(input_frame)
        self.entry_V1_mass.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="加质流量 Δṁ (kg/s):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.entry_m_add_mass = ttk.Entry(input_frame)
        self.entry_m_add_mass.grid(row=2, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="加质速度 V_add (m/s):").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.entry_V_add_mass = ttk.Entry(input_frame)
        self.entry_V_add_mass.grid(row=3, column=1, padx=5, pady=5)
        
        # 计算按钮
        btn_calc = ttk.Button(f, text="计算", command=self.calculate_mass_variation_flow)
        btn_calc.pack(pady=10)
        
        # 结果显示框架
        result_frame = ttk.LabelFrame(f, text="计算结果")
        result_frame.pack(pady=10, padx=10, fill="x")
        
        self.result_labels_mass = {
            'MassFlow2': ttk.Label(result_frame, text="出口质量流量 (kg/s) = "),
            'Velocity2': ttk.Label(result_frame, text="出口流速 (m/s) = "),
            'MomentumChange': ttk.Label(result_frame, text="动量变化率 (N) = ")
        }
        
        for i, (_, label) in enumerate(self.result_labels_mass.items()):
            label.grid(row=i, column=0, sticky="w", padx=10, pady=5)

    def calculate_mass_variation_flow(self):
        try:
            # 获取输入参数
            m_dot1 = float(self.entry_m1_mass.get())
            V1 = float(self.entry_V1_mass.get())
            m_dot_add = float(self.entry_m_add_mass.get())
            V_add = float(self.entry_V_add_mass.get())
            
            # 质量守恒
            m_dot2 = m_dot1 + m_dot_add
            
            # 动量守恒
            V2 = (m_dot1 * V1 + m_dot_add * V_add) / m_dot2
            
            # 动量变化率
            momentum_change = m_dot2 * V2 - m_dot1 * V1
            
            # 更新结果
            self.result_labels_mass['MassFlow2'].config(text=f"出口质量流量 (kg/s) = {m_dot2:.3f}")
            self.result_labels_mass['Velocity2'].config(text=f"出口流速 (m/s) = {V2:.2f}")
            self.result_labels_mass['MomentumChange'].config(text=f"动量变化率 (N) = {momentum_change:.2f}")
            
        except ValueError:
            messagebox.showerror("输入错误", "请输入有效的数字参数")
        except Exception as e:
            messagebox.showerror("计算错误", f"计算过程中发生错误: {str(e)}")
    







if __name__ == "__main__":
    root = tk.Tk()
    app = MainWindow(root)
    root.mainloop()