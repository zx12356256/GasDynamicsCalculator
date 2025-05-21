文件命名原则：main.py是主程序，所有修改都在此进行；v+数字，表示历史迭代版本（用以查看先前版本，不要做改动！！！）；.spec文件用于封装exe;

本地运行推荐使用VScode或者Pycharm
需安装：pip install tk pillow
pip install pyinstaller
pip install opencv-python
……其他库若报错请自行安装

背景视频MP4太大了传不上去，有需要我再单独发
封装exe命令：pyinstaller --onefile --noconfirm --windowed --add-data "background.jpg;." main.py
通常会存储在同文件夹的dist里面
