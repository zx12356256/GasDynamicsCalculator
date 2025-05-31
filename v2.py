import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import os
import sys
import pygame
import math

def resource_path(relative_path):
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

        self.video_source = resource_path(video_source)
        self.audio_source = resource_path(audio_source)

        if not os.path.exists(self.video_source):
            raise FileNotFoundError(f"未找到视频文件: {self.video_source}")
        if not os.path.exists(self.audio_source):
            raise FileNotFoundError(f"未找到音频文件: {self.audio_source}")

        self.vid = cv2.VideoCapture(self.video_source)
        if not self.vid.isOpened():
            raise IOError(f"无法打开视频文件: {self.video_source}")

        pygame.mixer.init()
        pygame.mixer.music.load(self.audio_source)
        pygame.mixer.music.play(-1)

        self.play_video()

    def play_video(self):
        ret, frame = self.vid.read()
        if not ret:
            self.vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.vid.read()
        if ret:
            frame = cv2.resize(frame, (self.winfo_width(), self.winfo_height()))
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            self.photo = ImageTk.PhotoImage(image=img)
            self.create_image(0, 0, image=self.photo, anchor='nw')
        self.after(15, self.play_video)

    def stop(self):
        pygame.mixer.music.stop()
        self.vid.release()

class MainWindow:
    def __init__(self, master):
        self.master = master
        master.title("气体动力参数计算 v1.0")
        master.geometry("1280x720")
        master.resizable(False, False)

        screen_width = master.winfo_screenwidth()
        screen_height = master.winfo_screenheight()
        x_position = int((screen_width / 2) - (1280 / 2))
        y_position = int((screen_height / 2) - (720 / 2))
        master.geometry(f"1280x720+{x_position}+{y_position}")

        self.canvas = VideoCanvas(master)
        self.canvas.pack(fill="both", expand=True)

        button_style = {'font': ('微软雅黑', 14), 'width': 20, 'height': 2}
        btn1 = tk.Button(master, text="1. 滞止参数与气动函数", command=self.open_stagnation, **button_style)
        btn2 = tk.Button(master, text="2. 膨胀波计算", command=self.open_expansion, **button_style)
        btn3 = tk.Button(master, text="3. 激波计算", command=self.open_combined_shockwave, **button_style)
        btn4 = tk.Button(master, text="4. 一维定常管内流动", command=self.open_flow, **button_style)

        self.mute_button = tk.Button(master, text="静音", command=self.toggle_mute, **{'font': ('微软雅黑', 12)})
        self.is_muted = False

        btn_width = 200
        btn_height = 60
        center_x = 1280 // 2
        center_y = 720 // 2

        self.canvas.create_window(center_x + btn_width - 40, center_y + btn_height + 80, window=btn1)
        self.canvas.create_window(center_x + btn_width + 250, center_y + btn_height + 80, window=btn2)
        self.canvas.create_window(center_x + btn_width - 40, center_y + btn_height + 180, window=btn3)
        self.canvas.create_window(center_x + btn_width + 250, center_y + btn_height + 180, window=btn4)
        self.mute_button.place(x=1280 - 100, y=20)

    def toggle_mute(self):
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
        ExpansionWindow(tk.Toplevel(self.master))

    def open_combined_shockwave(self):
        ShockwaveWindow(tk.Toplevel(self.master))
        VelocityShockwaveWindow(tk.Toplevel(self.master))

    def open_flow(self):
        pass

# 其他窗口类应在此处继续添加（略）
