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
        combined_shockwave_window.master.geometry("1200x500+200+200")  # 设置初始位置和大小

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
        master.geometry("1200x500")  # 调整尺寸以适应两个子窗口

        # 创建标签框架分别放置斜激波和正激波的界面
        shockwave_frame = ttk.LabelFrame(master, text="斜激波参数计算1")
        shockwave_frame.pack(side="left", padx=10, pady=5, fill="both", expand=True)

        velocity_shockwave_frame = ttk.LabelFrame(master, text="正激波参数计算")
        velocity_shockwave_frame.pack(side="right", padx=10, pady=5, fill="both", expand=True)

        pressure_shockwave_frame = ttk.LabelFrame(master, text="斜激波参数计算2")
        pressure_shockwave_frame.pack(side="left", padx=10, pady=5, fill="both", expand=True)
        # 实例化ShockwaveWindow和VelocityShockwaveWindow并将它们的控件添加到对应的框架中
        self.shockwave_window = ShockwaveWindow(shockwave_frame)
        self.velocity_shockwave_window = VelocityShockwaveWindow(velocity_shockwave_frame)
        self.pressure_shockwave_window = ShockwaveByPressure(pressure_shockwave_frame)


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
            ma1_values = np.linspace(min_ma1 + 0.01, 5.0, 100)
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
            if min_ma1 <= user_ma1 <= 5.0:
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

class FlowWindow:#简易版
    def __init__(self, master):
        self.master = master
        master.title("一维定常管内流动计算器")
        master.geometry("700x550")
        
        # 创建选项卡
        self.notebook = ttk.Notebook(master)
        
        # 创建并添加各个喷管类型到notebook
        self.converging_nozzle_frame = ttk.Frame(self.notebook)
        self.create_converging_nozzle_tab()
        
        self.laval_nozzle_frame = ttk.Frame(self.notebook)
        self.create_laval_nozzle_tab()
        
        self.frictional_flow_frame = ttk.Frame(self.notebook)
        self.create_frictional_flow_tab()
        
        self.heat_transfer_flow_frame = ttk.Frame(self.notebook)
        self.create_heat_transfer_flow_tab()
        
        self.mass_variation_flow_frame = ttk.Frame(self.notebook)
        self.create_mass_variation_flow_tab()
        
        self.notebook.add(self.converging_nozzle_frame, text="收缩喷管")
        self.notebook.add(self.laval_nozzle_frame, text="拉瓦尔喷管")
        self.notebook.add(self.frictional_flow_frame, text="摩擦管流")
        self.notebook.add(self.heat_transfer_flow_frame, text="换热管流")
        self.notebook.add(self.mass_variation_flow_frame, text="变流量加质管流")
        
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)
        
        # 设置默认值用于测试
        self.set_default_values()

    def set_default_values(self):
        """设置默认输入值用于测试"""
        # 收缩喷管默认值
        self.entry_P0_conv.delete(0, tk.END)
        self.entry_P0_conv.insert(0, "500000")  # 500 kPa
        self.entry_T0_conv.delete(0, tk.END)
        self.entry_T0_conv.insert(0, "400")
        self.entry_A_out_conv.delete(0, tk.END)
        self.entry_A_out_conv.insert(0, "0.01")
        self.entry_Pb_conv.delete(0, tk.END)
        self.entry_Pb_conv.insert(0, "300000")  # 300 kPa
        
        # 拉瓦尔喷管默认值
        self.entry_P0_laval.delete(0, tk.END)
        self.entry_P0_laval.insert(0, "1000000")  # 1000 kPa
        self.entry_T0_laval.delete(0, tk.END)
        self.entry_T0_laval.insert(0, "500")
        self.entry_A_throat_laval.delete(0, tk.END)
        self.entry_A_throat_laval.insert(0, "0.005")
        self.entry_A_exit_laval.delete(0, tk.END)
        self.entry_A_exit_laval.insert(0, "0.01")
        
        # 摩擦管流默认值
        self.entry_L_fric.delete(0, tk.END)
        self.entry_L_fric.insert(0, "1")
        self.entry_D_fric.delete(0, tk.END)
        self.entry_D_fric.insert(0, "0.05")
        self.entry_f_fric.delete(0, tk.END)
        self.entry_f_fric.insert(0, "0.02")
        self.entry_M1_fric.delete(0, tk.END)
        self.entry_M1_fric.insert(0, "0.3")
        self.entry_P01_fric.delete(0, tk.END)
        self.entry_P01_fric.insert(0, "200000")  # 200 kPa
        self.entry_T01_fric.delete(0, tk.END)
        self.entry_T01_fric.insert(0, "300")
        
        # 换热管流默认值
        self.entry_M1_heat.delete(0, tk.END)
        self.entry_M1_heat.insert(0, "0.2")
        self.entry_T01_heat.delete(0, tk.END)
        self.entry_T01_heat.insert(0, "300")
        self.entry_q_heat.delete(0, tk.END)
        self.entry_q_heat.insert(0, "100")  # kJ/kg
        
        # 变流量加质管流默认值
        self.entry_m1_mass.delete(0, tk.END)
        self.entry_m1_mass.insert(0, "2")
        self.entry_V1_mass.delete(0, tk.END)
        self.entry_V1_mass.insert(0, "100")
        self.entry_m_add_mass.delete(0, tk.END)
        self.entry_m_add_mass.insert(0, "0.5")
        self.entry_V_add_mass.delete(0, tk.END)
        self.entry_V_add_mass.insert(0, "50")

    def create_converging_nozzle_tab(self):
        """创建收缩喷管计算标签页"""
        frame = self.converging_nozzle_frame
        # 输入参数框架
        input_frame = ttk.LabelFrame(frame, text="输入参数")
        input_frame.pack(pady=10, padx=10, fill="x")
        
        # 创建输入字段
        ttk.Label(input_frame, text="入口总压 P0 (Pa):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.entry_P0_conv = ttk.Entry(input_frame)
        self.entry_P0_conv.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="入口总温 T0 (K):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.entry_T0_conv = ttk.Entry(input_frame)
        self.entry_T0_conv.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="出口面积 A (m²):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.entry_A_out_conv = ttk.Entry(input_frame)
        self.entry_A_out_conv.grid(row=2, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="出口背压 Pb (Pa):").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.entry_Pb_conv = ttk.Entry(input_frame)
        self.entry_Pb_conv.grid(row=3, column=1, padx=5, pady=5)
        
        # 计算按钮
        btn_calc = ttk.Button(frame, text="计算", command=self.calculate_converging_nozzle)
        btn_calc.pack(pady=10)
        
        # 结果显示框架
        result_frame = ttk.LabelFrame(frame, text="计算结果")
        result_frame.pack(pady=10, padx=10, fill="x")
        
        self.result_labels_conv = {
            'Mach': ttk.Label(result_frame, text="出口马赫数 = "),
            'Velocity': ttk.Label(result_frame, text="出口流速 (m/s) = "),
            'MassFlow': ttk.Label(result_frame, text="质量流量 (kg/s) = "),
            'Pressure': ttk.Label(result_frame, text="出口静压 (Pa) = ")
        }
        
        for i, (_, label) in enumerate(self.result_labels_conv.items()):
            label.grid(row=i, column=0, sticky="w", padx=10, pady=5)

    def calculate_converging_nozzle(self):
        try:
            # 获取输入参数
            P0 = float(self.entry_P0_conv.get()) 
            T0 = float(self.entry_T0_conv.get())
            A_out = float(self.entry_A_out_conv.get())
            Pb = float(self.entry_Pb_conv.get())
            gamma = 1.4
            R = 287.0
            
            # 计算临界压力比
            crit_ratio = (2/(gamma+1))**(gamma/(gamma-1))
            P_crit = P0 * crit_ratio
            
            if Pb < P_crit:  # 阻塞流动
                Me = 1.0
                Pe = P_crit
                Te = T0 * 2/(gamma+1)
                rho_e = Pe/(R*Te)
                Ve = math.sqrt(gamma*R*Te)
            else:  # 亚声速流动
                Pe = Pb
                # 求解等熵流方程得到出口马赫数
                Me = math.sqrt((2/(gamma-1))*((P0/Pb)**((gamma-1)/gamma)-1))
                Te = T0/(1+(gamma-1)/2*Me**2)
                rho_e = Pe/(R*Te)
                Ve = Me * math.sqrt(gamma*R*Te)
            
            # 计算质量流量
            m_dot = rho_e * A_out * Ve
            
            # 更新结果
            self.result_labels_conv['Mach'].config(text=f"出口马赫数 = {Me:.4f}")
            self.result_labels_conv['Velocity'].config(text=f"出口流速 (m/s) = {Ve:.2f}")
            self.result_labels_conv['MassFlow'].config(text=f"质量流量 (kg/s) = {m_dot:.4f}")
            self.result_labels_conv['Pressure'].config(text=f"出口静压 (Pa) = {Pe:.2f}")
            
        except ValueError:
            messagebox.showerror("输入错误", "请输入有效的数字参数")
        except Exception as e:
            messagebox.showerror("计算错误", f"计算过程中发生错误: {str(e)}")

    def create_laval_nozzle_tab(self):
        """创建拉瓦尔喷管计算标签页"""
        frame = self.laval_nozzle_frame
        # 输入参数框架
        input_frame = ttk.LabelFrame(frame, text="输入参数")
        input_frame.pack(pady=10, padx=10, fill="x")
        
        # 创建输入字段
        ttk.Label(input_frame, text="入口总压 P0 (Pa):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.entry_P0_laval = ttk.Entry(input_frame)
        self.entry_P0_laval.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="入口总温 T0 (K):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.entry_T0_laval = ttk.Entry(input_frame)
        self.entry_T0_laval.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="喉部面积 A* (m²):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.entry_A_throat_laval = ttk.Entry(input_frame)
        self.entry_A_throat_laval.grid(row=2, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="出口面积 Ae (m²):").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.entry_A_exit_laval = ttk.Entry(input_frame)
        self.entry_A_exit_laval.grid(row=3, column=1, padx=5, pady=5)
        
        # 计算按钮
        btn_calc = ttk.Button(frame, text="计算", command=self.calculate_laval_nozzle)
        btn_calc.pack(pady=10)
        
        # 结果显示框架
        result_frame = ttk.LabelFrame(frame, text="计算结果")
        result_frame.pack(pady=10, padx=10, fill="x")
        
        self.result_labels_laval = {
            'Mach': ttk.Label(result_frame, text="出口马赫数 = "),
            'MassFlow': ttk.Label(result_frame, text="质量流量 (kg/s) = "),
            'Pressure': ttk.Label(result_frame, text="出口静压 (Pa) = "),
            'Velocity': ttk.Label(result_frame, text="出口流速 (m/s) = ")
        }
        
        for i, (_, label) in enumerate(self.result_labels_laval.items()):
            label.grid(row=i, column=0, sticky="w", padx=10, pady=5)

    def solve_area_ratio(self, area_ratio, gamma=1.4, subsonic=False):
        """求解等熵面积比方程，返回马赫数"""
        # 使用牛顿迭代法求解
        # 等熵面积比方程: A/A* = 1/M * [ (2/(γ+1)) * (1 + (γ-1)/2 * M^2) ]^((γ+1)/(2(γ-1)))
        
        # 定义函数和导数
        def f(M):
            return (1/M) * ((2/(gamma+1)) * (1 + (gamma-1)/2 * M**2))**((gamma+1)/(2*(gamma-1))) - area_ratio
        
        def df(M):
            term1 = ((2/(gamma+1)) * (1 + (gamma-1)/2 * M**2))**((gamma+1)/(2*(gamma-1)))
            term2 = (gamma+1)/(4*(gamma-1)) * (gamma-1)*M * (1 + (gamma-1)/2 * M**2)**-1
            return -1/(M**2) * term1 + (1/M) * term1 * term2
        
        # 选择初始值（亚音速或超音速）
        M0 = 0.5 if subsonic else 2.0
        
        # 牛顿迭代
        tolerance = 1e-6
        max_iter = 100
        for _ in range(max_iter):
            f_val = f(M0)
            df_val = df(M0)
            M1 = M0 - f_val/df_val
            
            if abs(M1 - M0) < tolerance:
                return M1
            
            M0 = M1
        
        return M0

    def calculate_laval_nozzle(self):
        try:
            # 获取输入参数
            P0 = float(self.entry_P0_laval.get())
            T0 = float(self.entry_T0_laval.get())
            A_throat = float(self.entry_A_throat_laval.get())
            A_exit = float(self.entry_A_exit_laval.get())
            gamma = 1.4
            R = 287.0
            
            # 计算临界参数（喉部）
            T_star = T0 * 2/(gamma+1)
            P_star = P0 * (2/(gamma+1))**(gamma/(gamma-1))
            rho_star = P_star/(R*T_star)
            V_star = math.sqrt(gamma*R*T_star)
            
            # 计算质量流量 (喉部达到声速)
            m_dot = rho_star * A_throat * V_star
            
            # 计算面积比
            area_ratio = A_exit / A_throat
            
            # 求解等熵面积比方程得到出口马赫数
            # 假设为超音速解
            Me = self.solve_area_ratio(area_ratio, gamma, subsonic=False)
            
            # 计算出口参数
            Te = T0 / (1 + (gamma-1)/2 * Me**2)
            Pe = P0 / (1 + (gamma-1)/2 * Me**2)**(gamma/(gamma-1))
            Ve = Me * math.sqrt(gamma*R*Te)
            
            # 更新结果
            self.result_labels_laval['Mach'].config(text=f"出口马赫数 = {Me:.4f}")
            self.result_labels_laval['MassFlow'].config(text=f"质量流量 (kg/s) = {m_dot:.4f}")
            self.result_labels_laval['Pressure'].config(text=f"出口静压 (Pa) = {Pe:.2f}")
            self.result_labels_laval['Velocity'].config(text=f"出口流速 (m/s) = {Ve:.2f}")
            
        except ValueError:
            messagebox.showerror("输入错误", "请输入有效的数字参数")
        except Exception as e:
            messagebox.showerror("计算错误", f"计算过程中发生错误: {str(e)}")

    def create_frictional_flow_tab(self):
        """创建摩擦管流计算标签页"""
        frame = self.frictional_flow_frame
        # 输入参数框架
        input_frame = ttk.LabelFrame(frame, text="输入参数")
        input_frame.pack(pady=10, padx=10, fill="x")
        
        # 创建输入字段
        ttk.Label(input_frame, text="管长 L (m):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.entry_L_fric = ttk.Entry(input_frame)
        self.entry_L_fric.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="直径 D (m):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.entry_D_fric = ttk.Entry(input_frame)
        self.entry_D_fric.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="摩擦系数 f:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.entry_f_fric = ttk.Entry(input_frame)
        self.entry_f_fric.grid(row=2, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="进口马赫数 M1:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.entry_M1_fric = ttk.Entry(input_frame)
        self.entry_M1_fric.grid(row=3, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="进口总压 P01 (Pa):").grid(row=4, column=0, padx=5, pady=5, sticky="w")
        self.entry_P01_fric = ttk.Entry(input_frame)
        self.entry_P01_fric.grid(row=4, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="进口总温 T01 (K):").grid(row=5, column=0, padx=5, pady=5, sticky="w")
        self.entry_T01_fric = ttk.Entry(input_frame)
        self.entry_T01_fric.grid(row=5, column=1, padx=5, pady=5)
        
        # 计算按钮
        btn_calc = ttk.Button(frame, text="计算", command=self.calculate_frictional_flow)
        btn_calc.pack(pady=10)
        
        # 结果显示框架
        result_frame = ttk.LabelFrame(frame, text="计算结果")
        result_frame.pack(pady=10, padx=10, fill="x")
        
        self.result_labels_fric = {
            'Mach2': ttk.Label(result_frame, text="出口马赫数 M2 = "),
            'PressureLoss': ttk.Label(result_frame, text="总压损失 (Pa) = "),
            'MaxLength': ttk.Label(result_frame, text="最大管长 (m) = ")
        }
        
        for i, (_, label) in enumerate(self.result_labels_fric.items()):
            label.grid(row=i, column=0, sticky="w", padx=10, pady=5)

    def fanno_flow_function(self, M, gamma=1.4):
        """范诺流函数，计算4fL*/D"""
        term1 = (1 - M**2) / (gamma * M**2)
        term2 = (gamma + 1) / (2 * gamma) * math.log(((gamma + 1) * M**2) / (2 + (gamma - 1) * M**2))
        return term1 + term2

    def solve_fanno_mach(self, M1, fl_actual, gamma=1.4):
        """求解摩擦管流的出口马赫数"""
        # 定义目标函数
        def target(M2):
            fl_max1 = self.fanno_flow_function(M1, gamma)
            fl_max2 = self.fanno_flow_function(M2, gamma)
            return fl_max1 - fl_max2 - fl_actual
        
        # 使用二分法求解
        M_low = 0.01
        M_high = 1.0  # 出口马赫数在亚音速范围内
        
        tolerance = 1e-6
        max_iter = 100
        
        for _ in range(max_iter):
            M_mid = (M_low + M_high) / 2
            f_mid = target(M_mid)
            
            if abs(f_mid) < tolerance:
                return M_mid
                
            if f_mid > 0:
                M_high = M_mid
            else:
                M_low = M_mid
        
        return (M_low + M_high) / 2

    def calculate_frictional_flow(self):
        try:
            # 获取输入参数
            L = float(self.entry_L_fric.get())
            D = float(self.entry_D_fric.get())
            f = float(self.entry_f_fric.get())
            M1 = float(self.entry_M1_fric.get())
            P01 = float(self.entry_P01_fric.get())
            T01 = float(self.entry_T01_fric.get())
            gamma = 1.4
            
            # 计算实际4fL/D
            fl_actual = 4 * f * L / D
            
            # 计算最大4fL*/D (对应M=1)
            fl_max = self.fanno_flow_function(M1, gamma)
            
            # 检查是否超过最大管长
            if fl_actual > fl_max:
                messagebox.showwarning("警告", "流动阻塞！超过最大管长")
                M2 = 1.0  # 阻塞时出口马赫数为1
            else:
                # 计算出口马赫数
                M2 = self.solve_fanno_mach(M1, fl_actual, gamma)
            
            # 计算总压比
            P02_P01 = (M1 / M2) * math.sqrt(
                (1 + 0.5*(gamma-1)*M2**2) / (1 + 0.5*(gamma-1)*M1**2))
            
            # 计算最大管长
            max_length = fl_max * D / (4 * f)
            
            # 更新结果
            self.result_labels_fric['Mach2'].config(text=f"出口马赫数 M2 = {M2:.4f}")
            self.result_labels_fric['PressureLoss'].config(text=f"总压损失 (Pa) = {P01*(1-P02_P01):.2f}")
            self.result_labels_fric['MaxLength'].config(text=f"最大管长 (m) = {max_length:.2f}")
            
        except ValueError:
            messagebox.showerror("输入错误", "请输入有效的数字参数")
        except Exception as e:
            messagebox.showerror("计算错误", f"计算过程中发生错误: {str(e)}")

    def create_heat_transfer_flow_tab(self):
        """创建换热管流计算标签页"""
        frame = self.heat_transfer_flow_frame
        # 输入参数框架
        input_frame = ttk.LabelFrame(frame, text="输入参数")
        input_frame.pack(pady=10, padx=10, fill="x")
        
        # 创建输入字段
        ttk.Label(input_frame, text="进口马赫数 M1:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.entry_M1_heat = ttk.Entry(input_frame)
        self.entry_M1_heat.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="进口总温 T01 (K):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.entry_T01_heat = ttk.Entry(input_frame)
        self.entry_T01_heat.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="热交换量 q (kJ/kg):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.entry_q_heat = ttk.Entry(input_frame)
        self.entry_q_heat.grid(row=2, column=1, padx=5, pady=5)
        
        # 计算按钮
        btn_calc = ttk.Button(frame, text="计算", command=self.calculate_heat_transfer_flow)
        btn_calc.pack(pady=10)
        
        # 结果显示框架
        result_frame = ttk.LabelFrame(frame, text="计算结果")
        result_frame.pack(pady=10, padx=10, fill="x")
        
        self.result_labels_heat = {
            'Mach2': ttk.Label(result_frame, text="出口马赫数 M2 = "),
            'T02': ttk.Label(result_frame, text="出口总温 T02 (K) = "),
            'PressureRatio': ttk.Label(result_frame, text="总压比 P02/P01 = ")
        }
        
        for i, (_, label) in enumerate(self.result_labels_heat.items()):
            label.grid(row=i, column=0, sticky="w", padx=10, pady=5)

    def solve_rayleigh_mach(self, M1, T01, T02, gamma=1.4, Cp=1005):
        """求解瑞利流的出口马赫数"""
        # 定义目标函数
        def target(M2):
            term1 = (1 + gamma*M1**2)**2 * M2**2 * (1 + (gamma-1)/2*M2**2)
            term2 = (1 + gamma*M2**2)**2 * M1**2 * (1 + (gamma-1)/2*M1**2)
            T02_calc = T01 * term1 / term2
            return T02_calc - T02
        
        # 使用二分法求解
        M_low = 0.01
        M_high = 1.0 if M1 < 1.0 else 5.0  # 根据进口马赫数选择范围
        
        tolerance = 1e-6
        max_iter = 100
        
        for _ in range(max_iter):
            M_mid = (M_low + M_high) / 2
            f_mid = target(M_mid)
            
            if abs(f_mid) < tolerance:
                return M_mid
                
            if f_mid > 0:
                M_high = M_mid
            else:
                M_low = M_mid
        
        return (M_low + M_high) / 2

    def calculate_heat_transfer_flow(self):
        try:
            # 获取输入参数
            M1 = float(self.entry_M1_heat.get())
            T01 = float(self.entry_T01_heat.get())
            q = float(self.entry_q_heat.get()) * 1000  # 转换为J/kg
            gamma = 1.4
            Cp = 1005  # 空气定压比热容 J/kg·K
            
            # 计算出口总温
            T02 = T01 + q / Cp
            
            # 计算出口马赫数
            M2 = self.solve_rayleigh_mach(M1, T01, T02, gamma, Cp)
            
            # 计算总压比
            P02_P01 = ( (1 + gamma*M1**2) / (1 + gamma*M2**2) ) * (
                (1 + (gamma-1)/2*M2**2) / (1 + (gamma-1)/2*M1**2) )**(gamma/(gamma-1))
            
            # 更新结果
            self.result_labels_heat['Mach2'].config(text=f"出口马赫数 M2 = {M2:.4f}")
            self.result_labels_heat['T02'].config(text=f"出口总温 T02 (K) = {T02:.2f}")
            self.result_labels_heat['PressureRatio'].config(text=f"总压比 P02/P01 = {P02_P01:.4f}")
            
        except ValueError:
            messagebox.showerror("输入错误", "请输入有效的数字参数")
        except Exception as e:
            messagebox.showerror("计算错误", f"计算过程中发生错误: {str(e)}")

    def create_mass_variation_flow_tab(self):
        """创建变流量加质管流计算标签页"""
        frame = self.mass_variation_flow_frame
        # 输入参数框架
        input_frame = ttk.LabelFrame(frame, text="输入参数")
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
        btn_calc = ttk.Button(frame, text="计算", command=self.calculate_mass_variation_flow)
        btn_calc.pack(pady=10)
        
        # 结果显示框架
        result_frame = ttk.LabelFrame(frame, text="计算结果")
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
            
            # 假设等截面管道，忽略摩擦和热交换
            m_dot2 = m_dot1 + m_dot_add
            
            # 动量守恒计算出口速度
            V2 = (m_dot1 * V1 + m_dot_add * V_add) / m_dot2
            
            # 计算动量变化率
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
