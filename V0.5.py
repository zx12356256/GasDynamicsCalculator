import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import os
import sys
import pygame

def resource_path(relative_path):
    """ 获取资源的绝对路径 """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class VideoCanvas(tk.Canvas):
    def __init__(self, master=None, video_source="background.mp4", audio_source="background_audio.mp3", width=1280, height=720):
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
            raise FileNotFoundError(f"未找到音频文件: {self.audio_source}. 如果你需要从视频中提取音频，请使用 moviepy 或其他工具.")

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
        master.title("气体动力参数计算 v1.0")
        master.geometry("1280x720")
        master.resizable(False, False)

        # 创建画布并设置视频背景（含音频）
        self.canvas = VideoCanvas(master, video_source="background.mp4", audio_source="background_audio.mp3", width=1280, height=720)
        self.canvas.pack(fill="both", expand=True)

        # 创建功能按钮和静音按钮
        button_style = {'font': ('微软雅黑', 14), 'width': 20, 'height': 2}
        btn1 = tk.Button(master, text="1. 滞止参数与气动函数", command=self.open_stagnation, **button_style)
        btn2 = tk.Button(master, text="2. 膨胀波计算", command=self.open_expansion, **button_style)
        btn3 = tk.Button(master, text="3. 激波计算", command=self.open_shockwave, **button_style)
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
        self.canvas.create_window(center_x + btn_width - 40, center_y + btn_height +80, window=btn1)
        self.canvas.create_window(center_x + btn_width + 250, center_y + btn_height + 80, window=btn2)
        self.canvas.create_window(center_x + btn_width - 40, center_y + btn_height + 180, window=btn3)
        self.canvas.create_window(center_x + btn_width + 250, center_y + btn_height + 180, window=btn4)
        
        # 放置静音按钮于右上角
        self.mute_button.place(x=1280-100, y=20)  # 假设按钮宽度大约为100px

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
        StagnationWindow(tk.Toplevel(self.master))

    def open_expansion(self):
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
            T_ratio = 1 + (gamma - 1) / 2 * Ma ** 2
            P_ratio = T_ratio ** (gamma / (gamma - 1))
            rho_ratio = P_ratio ** (1 / gamma)

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