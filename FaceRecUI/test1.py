#
import csv
import glob
import os
import re
import shutil
import time
import warnings
from os import getcwd
import torch
import torch
import torch.nn as nn
import cv2
import dlib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QMovie
from PyQt5.QtWidgets import QFileDialog, QTableWidgetItem, QAbstractItemView
from PyQt5.QtCore import Qt
import datetime
from PyQt5.QtCore import QThread, pyqtSignal
import dlib
import cv2
from facenet_pytorch import MTCNN
from PIL import Image
import cv2
import numpy as np
import serial

mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')


def augment_image(img):
    flip = cv2.flip(img, 1)  # 镜像
    blur = cv2.GaussianBlur(img, (5, 5), 0)  # 数据镜像增强
    return [img, flip, blur]


class ModelLoader(QThread):
    models_loaded = pyqtSignal(object, object, object)  # 发射模型加载完成信号

    def run(self):
        # 初始化 MTCNN 检测器
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        mtcnn = MTCNN(keep_all=True, device=device)

        # dlib 的人脸关键点预测器和识别模型仍然可以使用
        predictor = dlib.shape_predictor('../data/data_dlib/shape_predictor_68_face_landmarks.dat')
        reco_model = dlib.face_recognition_model_v1("../data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

        # 抛出模型供其他函数使用
        self.models_loaded.emit(mtcnn, predictor, reco_model)


from FaceRecognition_UI import Ui_MainWindow

# 忽略警告
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings(action='ignore')


class Face_MainWindow(Ui_MainWindow):
    def __init__(self, MainWindow):
        self.path_face_dir = "../data/database_faces/"
        self.fontC = ImageFont.truetype("./Font/platech.ttf", 14, 0)

        self.cap_video = None  # 视频流对象

        self.path = getcwd()
        self.video_path = getcwd()

        self.timer_camera = QtCore.QTimer()  # 定时器
        self.timer_camera_load = QtCore.QTimer()  # 载入相机定时器
        self.timer_video = QtCore.QTimer()  # 视频定时器
        self.flag_timer = ""  # 用于标记正在进行的功能项（视频/摄像）

        self.CAM_NUM = 0  # 摄像头标号
        self.cap = cv2.VideoCapture(self.CAM_NUM)  # 屏幕画面对象
        self.cap_video = None  # 视频流对象

        self.current_image = None  # 当前的画面
        self.current_face = None  # 当前的人脸

        self.setupUi(MainWindow)  # 界面生成
        self.retranslateUi(MainWindow)  # 界面控件
        self.resetUi()
        self.slot_init()  # 槽函数

        self.count = 0
        self.count_face = 0
        self.col_row = []

        # 初始化检测器 - 使用MTCNN和dlib混合检测
        self.use_mtcnn = True  # 是否使用MTCNN
        self.mtcnn_detector = None
        self.detector = dlib.get_frontal_face_detector()  # 保留dlib作为备用
        self.predictor = dlib.shape_predictor('../data/data_dlib/shape_predictor_68_face_landmarks.dat')
        self.face_reco_model = dlib.face_recognition_model_v1("../data/data_dlib"
                                                              "/dlib_face_recognition_resnet_model_v1.dat")

        # 启动模型加载线程
        self.model_loader = ModelLoader()
        self.model_loader.models_loaded.connect(self.on_models_loaded)
        self.model_loader.start()

        # 性能统计
        self.mtcnn_detection_count = 0
        self.dlib_detection_count = 0
        self.total_detection_time = 0
        self.detection_count = 0
        self.face_feature_exist = []  # 用来存放所有录入人脸特征的数组
        self.face_name_exist = []  # 存储录入人脸名字

        # 用来存储上一帧和当前帧 ROI 的质心坐标
        self.last_centroid = []
        self.current_centroid = []

        # 用来存储上一帧和当前帧检测出目标的名字
        self.last_face_name = []
        self.current_face_name = []

        # 上一帧和当前帧中人脸数的计数器
        self.last_face_cnt = 0
        self.current_face_cnt = 0

        # 存储当前摄像头中捕获到的所有人脸的坐标名字
        self.current_face_position = []
        # 存储当前摄像头中捕获到的人脸特征
        self.current_face_feature = []

        self.reclassify_cnt = 0
        self.reclassify_interval = 20

        # 前后帧的距离
        self.last_current_distance = 0

        # 用来存放识别的距离
        self.current_face_distance = []

        self.exist_flag = None

    def on_models_loaded(self, mtcnn, predictor, reco_model):
        """模型加载完成的回调函数"""
        self.mtcnn_detector = mtcnn
        print("MTCNN模型加载完成，已启用混合检测模式")

    def detect_faces_hybrid(self, img):
        """混合人脸检测：优先使用MTCNN，失败时回退到dlib"""
        start_time = time.time()
        faces = []
        used_mtcnn = False

        if self.use_mtcnn and self.mtcnn_detector is not None:
            try:
                # 使用MTCNN检测
                img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                boxes, probs = self.mtcnn_detector.detect(img_pil)

                if boxes is not None and len(boxes) > 0:
                    # 过滤置信度较低的检测结果
                    valid_boxes = []
                    for i, (box, prob) in enumerate(zip(boxes, probs)):
                        if prob > 0.9:  # 置信度阈值
                            valid_boxes.append(box)

                    if valid_boxes:
                        # 转换MTCNN输出格式为dlib兼容格式
                        for box in valid_boxes:
                            left, top, right, bottom = box.astype(int)
                            # 边界检查
                            left = max(0, left)
                            top = max(0, top)
                            right = min(img.shape[1], right)
                            bottom = min(img.shape[0], bottom)

                            # 创建dlib.rectangle对象
                            rect = dlib.rectangle(left, top, right, bottom)
                            faces.append(rect)

                        # 如果MTCNN检测到人脸，直接返回
                        if faces:
                            used_mtcnn = True
                            self.mtcnn_detection_count += 1

            except Exception as e:
                print(f"MTCNN检测失败，回退到dlib: {e}")
                # 可选：记录错误到日志文件

        # 回退到dlib检测或MTCNN未检测到人脸时使用dlib
        if not faces:
            faces = self.detector(img, 0)
            if not used_mtcnn:
                self.dlib_detection_count += 1

        # 更新性能统计
        detection_time = time.time() - start_time
        self.total_detection_time += detection_time
        self.detection_count += 1

        return faces

    def toggle_detection_mode(self):
        """切换检测模式：MTCNN <-> dlib"""
        self.use_mtcnn = not self.use_mtcnn
        mode = "MTCNN混合模式" if self.use_mtcnn else "dlib传统模式"
        print(f"已切换到{mode}")
        return mode

    def get_detection_stats(self):
        """获取检测性能统计信息"""
        if self.detection_count == 0:
            return "暂无检测统计数据"

        avg_time = self.total_detection_time / self.detection_count
        mtcnn_ratio = (self.mtcnn_detection_count / self.detection_count) * 100
        dlib_ratio = (self.dlib_detection_count / self.detection_count) * 100

        stats = f"""检测性能统计:
总检测次数: {self.detection_count}
MTCNN使用次数: {self.mtcnn_detection_count} ({mtcnn_ratio:.1f}%)
dlib使用次数: {self.dlib_detection_count} ({dlib_ratio:.1f}%)
平均检测时间: {avg_time:.3f}秒
总检测时间: {self.total_detection_time:.3f}秒"""
        return stats

    def reset_detection_stats(self):
        """重置检测统计信息"""
        self.mtcnn_detection_count = 0
        self.dlib_detection_count = 0
        self.total_detection_time = 0
        self.detection_count = 0
        print("检测统计信息已重置")

    def resetUi(self):
        # 设置表格形式
        self.tableWidget_rec.horizontalHeader().setVisible(True)
        self.tableWidget_mana.horizontalHeader().setVisible(True)
        self.tableWidget_rec.setColumnWidth(0, 80)
        self.tableWidget_rec.setColumnWidth(1, 200)
        self.tableWidget_rec.setColumnWidth(2, 150)
        self.tableWidget_rec.setColumnWidth(3, 200)
        self.tableWidget_rec.setColumnWidth(4, 120)

        self.tableWidget_mana.setColumnWidth(0, 80)
        self.tableWidget_mana.setColumnWidth(1, 350)
        self.tableWidget_mana.setColumnWidth(2, 150)
        self.tableWidget_mana.setColumnWidth(3, 150)
        self.tableWidget_mana.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tabWidget.setCurrentIndex(0)
        self.tabWidget.setTabVisible(0, True)
        self.tabWidget.setTabVisible(1, False)
        self.tabWidget.setTabVisible(2, False)

        # 设置初始取图和录入按钮不可用
        self.toolButton_get_pic.setEnabled(False)
        self.toolButton_load_pic.setEnabled(False)
        self.toolButton_file_2.setEnabled(False)
        self.toolButton_camera_load.setEnabled(False)

        # 设置界面动画
        self.gif_movie()

    def gif_movie(self):
        # 设置界面动画
        gif = QMovie(':/newPrefix/images_test/face_rec.gif')
        self.label_display.setMovie(gif)
        self.label_display.setScaledContents(True)
        gif.start()

    def ini_value(self):
        self.face_feature_exist = []  # 用来存放所有录入人脸特征的数组
        self.face_name_exist = []  # 存储录入人脸名字

        # 用来存储上一帧和当前帧 ROI 的质心坐标
        self.last_centroid = []
        self.current_centroid = []

        # 用来存储上一帧和当前帧检测出目标的名字
        self.last_face_name = []
        self.current_face_name = []

        # 上一帧和当前帧中人脸数的计数器
        self.last_face_cnt = 0
        self.current_face_cnt = 0

        # 存储当前摄像头中捕获到的所有人脸的坐标名字
        self.current_face_position = []
        # 存储当前摄像头中捕获到的人脸特征
        self.current_face_feature = []

        self.reclassify_cnt = 0
        self.reclassify_interval = 20

        # 前后帧的距离
        self.last_current_distance = 0

        # 用来存放识别的距离
        self.current_face_distance = []

    def slot_init(self):
        self.toolButton_run_load.clicked.connect(self.change_size_load)
        self.toolButton_run_rec.clicked.connect(self.change_size_rec)
        self.toolButton_run_manage.clicked.connect(self.change_size_mana)
        self.toolButton_new_folder.clicked.connect(self.new_face_doing)
        self.toolButton_file_2.clicked.connect(self.choose_file)
        self.toolButton_get_pic.clicked.connect(self.get_img_doing)
        self.toolButton_load_pic.clicked.connect(self.load_img_doing)
        self.toolButton_file.clicked.connect(self.choose_rec_img)
        self.toolButton_runing.clicked.connect(self.run_rec)
        self.toolButton_camera.clicked.connect(self.button_open_camera_click)
        self.timer_camera.timeout.connect(self.show_camera)
        self.toolButton_video.clicked.connect(self.button_open_video_click)
        self.timer_video.timeout.connect(self.show_video)
        self.toolButton_camera_load.clicked.connect(self.button_open_camera_load)
        self.timer_camera_load.timeout.connect(self.show_camera_load)
        self.toolButton_mana_update.clicked.connect(self.do_update_face)
        self.tableWidget_mana.cellPressed.connect(self.table_review)
        self.toolButton_mana_delete.clicked.connect(self.delete_doing)

    def choose_rec_img(self):
        self.flag_timer = ""
        # 选择图片或视频文件后执行此槽函数
        self.timer_camera.stop()
        self.timer_video.stop()
        if self.cap:
            self.cap.release()
        if self.cap_video:
            self.cap_video.release()  # 释放视频画面帧
        # 清除UI上的label显示
        self.label_plate_result.setText("未知人脸")
        self.label_score_fps.setText("0")
        self.label_score_num.setText("0")
        self.label_score_dis.setText("0")
        # 清除文本框的显示内容
        self.textEdit_camera.setText("实时摄像已关闭")
        self.textEdit_camera.setStyleSheet("background-color: transparent;\n"
                                           "border-color: rgb(0, 170, 255);\n"
                                           "color: rgb(0, 170, 255);\n"
                                           "font: regular 12pt \"华为仿宋\";")
        self.textEdit_video.setText('实时视频已关闭')
        self.textEdit_video.setStyleSheet("background-color: transparent;\n"
                                          "border-color: rgb(0, 170, 255);\n"
                                          "color: rgb(0, 170, 255);\n"
                                          "font: regular 12pt \"华为仿宋\";")
        self.label_display.clear()
        # self.label_display.setStyleSheet("border-image: url(:/newPrefix/images_test/ini-image.png);")
        self.gif_movie()
        self.label_pic_newface.clear()
        # 使用文件选择对话框选择图片
        fileName_choose, filetype = QFileDialog.getOpenFileName(
            self.centralwidget, "选取图片文件",
            self.path,  # 起始路径
            "图片(*.jpg;*.jpeg;*.png)")  # 文件类型
        self.path = fileName_choose  # 保存路径
        if self.path != '':
            self.flag_timer = "image"
            self.textEdit_file.setText(self.path + '文件已选中')
            self.textEdit_file.setStyleSheet("background-color: transparent;\n"
                                             "border-color: rgb(0, 170, 255);\n"
                                             "color: rgb(0, 170, 255);\n"
                                             "font: regular 12pt \"华为仿宋\";")
            image = self.cv_imread(self.path)  # 读取选择的图片
            # image = cv2.imread("../LicensePlateRecognition/test3.jpeg")  # 读取选择的图片

            image = cv2.resize(image, (500, 500))  # 设定图像尺寸为显示界面大小

            if len(image.shape) < 3:
                self.path = ''
                self.label_display.setText("需要正常彩色图片，请重新选择！")
                self.label_display.setStyleSheet("border-image: url(:/newPrefix/images_test/ini-image.png);")
                self.gif_movie()

                return

            self.current_image = image.copy()
            show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # show = image.copy()
            showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
            a = QtGui.QPixmap.fromImage(showImage)
            self.label_display.setPixmap(a)
            self.label_display.setScaledContents(True)
            QtWidgets.QApplication.processEvents()

        else:
            # 选择取消，恢复界面状态
            self.flag_timer = ""
            self.textEdit_file.setText('图片文件未选中')
            self.textEdit_file.setStyleSheet("background-color: transparent;\n"
                                             "border-color: rgb(0, 170, 255);\n"
                                             "color: rgb(0, 170, 255);\n"
                                             "font: regular 12pt \"华为仿宋\";")

    def change_table(self, path, res, time_now, distance):
        # 更新表格记录
        self.count += 1  # 每识别出结果增加一条记录
        if self.count > 6:
            self.tableWidget_rec.setRowCount(self.count)
        newItem = QTableWidgetItem(str(self.count))  # 在表格中记录序号
        newItem.setTextAlignment(Qt.AlignCenter)
        self.tableWidget_rec.setItem(self.count - 1, 0, newItem)

        newItem = QTableWidgetItem(path)  # 在表格中记录车牌路径
        newItem.setTextAlignment(Qt.AlignVCenter)
        self.tableWidget_rec.setItem(self.count - 1, 1, newItem)

        newItem = QTableWidgetItem(res)  # 记录识别出的车牌在表格中
        newItem.setTextAlignment(Qt.AlignCenter)
        self.tableWidget_rec.setItem(self.count - 1, 2, newItem)
        self.tableWidget_rec.setCurrentItem(newItem)

        newItem = QTableWidgetItem(time_now)  # 记录识别出的车牌位置在表格中
        newItem.setTextAlignment(Qt.AlignCenter)
        self.tableWidget_rec.setItem(self.count - 1, 3, newItem)
        self.tableWidget_rec.setCurrentItem(newItem)

        newItem = QTableWidgetItem(str(round(distance, 4)))  # 记录识别出的车牌置信度在表格中
        newItem.setTextAlignment(Qt.AlignCenter)
        self.tableWidget_rec.setItem(self.count - 1, 4, newItem)
        self.tableWidget_rec.setCurrentItem(newItem)

    def button_open_camera_click(self):
        # self.count = 0
        # self.res_set = []
        if self.timer_video.isActive():
            self.timer_video.stop()
        self.flag_timer = ""
        if self.cap:
            self.cap.release()
        if self.cap_video:
            self.cap_video.release()  # 释放视频画面帧
        self.ini_value()

        if not self.timer_camera.isActive():  # 检查定时状态
            flag = self.cap.open(self.CAM_NUM)  # 检查相机状态
            if not flag:  # 相机打开失败提示
                QtWidgets.QMessageBox.warning(self.centralwidget, u"Warning",
                                              u"请检测相机与电脑是否连接正确！ ",
                                              buttons=QtWidgets.QMessageBox.Ok,
                                              defaultButton=QtWidgets.QMessageBox.Ok)
                self.flag_timer = ""
            else:
                self.textEdit_camera.setText("相机准备就绪")
                self.textEdit_camera.setStyleSheet("background-color: transparent;\n"
                                                   "border-color: rgb(0, 170, 255);\n"
                                                   "color: rgb(0, 170, 255);\n"
                                                   "font: regular 12pt \"华为仿宋\";")
                # 准备运行识别程序
                self.flag_timer = "camera"
                # 清除文本框的显示内容
                self.textEdit_video.setText("实时视频已关闭")
                self.textEdit_video.setStyleSheet("background-color: transparent;\n"
                                                  "border-color: rgb(0, 170, 255);\n"
                                                  "color: rgb(0, 170, 255);\n"
                                                  "font: regular 12pt \"华为仿宋\";")
                self.textEdit_file.setText('图片文件未选中')
                self.textEdit_file.setStyleSheet("background-color: transparent;\n"
                                                 "border-color: rgb(0, 170, 255);\n"
                                                 "color: rgb(0, 170, 255);\n"
                                                 "font: regular 12pt \"华为仿宋\";")
                # self.label_display.setText('正在启动识别系统...\n\nleading')
                QtWidgets.QApplication.processEvents()

                # 清除UI上的label显示
                self.label_plate_result.setText("未知人脸")
                self.label_score_fps.setText("0")
                self.label_score_num.setText("0")
                self.label_score_dis.setText("0")

        else:
            # 定时器未开启，界面回复初始状态
            self.flag_timer = ""
            self.timer_camera.stop()
            if self.cap:
                self.cap.release()
            self.label_display.clear()
            self.label_pic_newface.clear()
            self.textEdit_file.setText('文件未选中')
            self.textEdit_file.setStyleSheet("background-color: transparent;\n"
                                             "border-color: rgb(0, 170, 255);\n"
                                             "color: rgb(0, 170, 255);\n"
                                             "font: regular 12pt \"华为仿宋\";")
            self.textEdit_camera.setText('实时摄像已关闭')
            self.textEdit_camera.setStyleSheet("background-color: transparent;\n"
                                               "border-color: rgb(0, 170, 255);\n"
                                               "color: rgb(0, 170, 255);\n"
                                               "font: regular 12pt \"华为仿宋\";")
            self.textEdit_video.setText('实时视频已关闭')
            self.textEdit_video.setStyleSheet("background-color: transparent;\n"
                                              "border-color: rgb(0, 170, 255);\n"
                                              "color: rgb(0, 170, 255);\n"
                                              "font: regular 12pt \"华为仿宋\";")
            # self.label_display.setStyleSheet("border-image: url(:/newPrefix/images_test/ini-image.png);")
            self.gif_movie()

            # 清除UI上的label显示
            self.label_plate_result.setText("未知人脸")
            self.label_score_fps.setText("0")
            self.label_score_num.setText("0")
            self.label_score_dis.setText("0")

    def button_open_camera_load(self):
        # 用于点击录入窗口的摄像头按钮槽函数

        if self.timer_camera.isActive():
            self.timer_camera.stop()
        if self.timer_video.isActive():
            self.timer_video.stop()
        if self.cap:
            self.cap.release()
        if self.cap_video:
            self.cap_video.release()  # 释放视频画面帧
        self.label_display.clear()
        # self.label_display.setStyleSheet("border-image: url(:/newPrefix/images_test/ini-image.png);")
        self.gif_movie()

        self.label_pic_newface.clear()
        self.label_pic_org.clear()
        # self.count = 0

        if not self.timer_camera_load.isActive():  # 检查定时状态
            flag = self.cap.open(self.CAM_NUM)  # 检查相机状态
            if not flag:  # 相机打开失败提示
                QtWidgets.QMessageBox.warning(self.centralwidget, u"Warning",
                                              u"请检测相机与电脑是否连接正确！ ",
                                              buttons=QtWidgets.QMessageBox.Ok,
                                              defaultButton=QtWidgets.QMessageBox.Ok)
                self.flag_timer = ""
            else:
                self.textEdit_camera.setText("相机准备就绪")
                self.textEdit_camera.setStyleSheet("background-color: transparent;\n"
                                                   "border-color: rgb(0, 170, 255);\n"
                                                   "color: rgb(0, 170, 255);\n"
                                                   "font: regular 12pt \"华为仿宋\";")
                # 准备运行识别程序
                self.flag_timer = "camera_load"
                # 清除文本框的显示内容
                self.textEdit_video.setText("实时视频已关闭")
                self.textEdit_video.setStyleSheet("background-color: transparent;\n"
                                                  "border-color: rgb(0, 170, 255);\n"
                                                  "color: rgb(0, 170, 255);\n"
                                                  "font: regular 12pt \"华为仿宋\";")
                self.textEdit_file.setText('图片文件未选中')
                self.textEdit_file.setStyleSheet("background-color: transparent;\n"
                                                 "border-color: rgb(0, 170, 255);\n"
                                                 "color: rgb(0, 170, 255);\n"
                                                 "font: regular 12pt \"华为仿宋\";")
                # self.label_display.setText('正在启动识别系统...\n\nleading')
                QtWidgets.QApplication.processEvents()
                # 打开定时器
                self.timer_camera_load.start(30)
                # 清除UI上的label显示
                # self.label_plate_result.setText("未知人脸")
                # self.label_score_fps.setText("0")
                # self.label_score_num.setText("0")
                # self.label_score_dis.setText("0")

        else:
            # 定时器未开启，界面回复初始状态
            self.flag_timer = ""
            self.timer_camera_load.stop()
            if self.cap:
                self.cap.release()
            self.label_display.clear()
            self.label_pic_newface.clear()
            self.label_pic_org.clear()
            self.textEdit_file.setText('文件未选中')
            self.textEdit_file.setStyleSheet("background-color: transparent;\n"
                                             "border-color: rgb(0, 170, 255);\n"
                                             "color: rgb(0, 170, 255);\n"
                                             "font: regular 12pt \"华为仿宋\";")
            self.textEdit_camera.setText('实时摄像已关闭')
            self.textEdit_camera.setStyleSheet("background-color: transparent;\n"
                                               "border-color: rgb(0, 170, 255);\n"
                                               "color: rgb(0, 170, 255);\n"
                                               "font: regular 12pt \"华为仿宋\";")
            self.textEdit_video.setText('实时视频已关闭')
            self.textEdit_video.setStyleSheet("background-color: transparent;\n"
                                              "border-color: rgb(0, 170, 255);\n"
                                              "color: rgb(0, 170, 255);\n"
                                              "font: regular 12pt \"华为仿宋\";")
            # self.label_display.setStyleSheet("border-image: url(:/newPrefix/images_test/ini-image.png);")
            self.gif_movie()

            # 清除UI上的label显示
            self.label_plate_result.setText("未知人脸")
            self.label_score_fps.setText("0")
            self.label_score_num.setText("0")
            self.label_score_dis.setText("0")

    def button_open_video_click(self):
        if self.timer_camera.isActive():
            self.timer_camera.stop()
        if self.cap:
            self.cap.release()
        if self.cap_video:
            self.cap_video.release()  # 释放视频画面帧
        self.label_display.clear()
        # self.label_display.setStyleSheet("border-image: url(:/newPrefix/images_test/ini-image.png);")
        self.gif_movie()

        self.label_pic_newface.clear()
        self.ini_value()

        self.flag_timer = ""
        if not self.timer_video.isActive():  # 检查定时状态
            # 弹出文件选择框选择视频文件
            fileName_choose, filetype = QFileDialog.getOpenFileName(self.centralwidget, "选取视频文件",
                                                                    self.video_path,  # 起始路径
                                                                    "视频(*.mp4;*.avi)")  # 文件类型
            self.video_path = fileName_choose
            if fileName_choose != '':
                self.flag_timer = "video"
                self.textEdit_video.setText(fileName_choose + '文件已选中')
                self.textEdit_video.setStyleSheet("background-color: transparent;\n"
                                                  "border-color: rgb(0, 170, 255);\n"
                                                  "color: rgb(0, 170, 255);\n"
                                                  "font: regular 12pt \"华为仿宋\";")
                # self.label_display.setText('正在启动识别系统...\n\nleading')
                QtWidgets.QApplication.processEvents()

                try:  # 初始化视频流
                    self.cap_video = cv2.VideoCapture(fileName_choose)
                except:
                    print("[INFO] could not determine # of frames in video")
            else:
                self.textEdit_video.setText('视频文件未选中')
                self.textEdit_video.setStyleSheet("background-color: transparent;\n"
                                                  "border-color: rgb(0, 170, 255);\n"
                                                  "color: rgb(0, 170, 255);\n"
                                                  "font: regular 12pt \"华为仿宋\";")
            # 清除文本框的显示内容

            self.textEdit_camera.setText("实时摄像已关闭")
            self.textEdit_camera.setStyleSheet("background-color: transparent;\n"
                                               "border-color: rgb(0, 170, 255);\n"
                                               "color: rgb(0, 170, 255);\n"
                                               "font: regular 12pt \"华为仿宋\";")
            self.textEdit_file.setText('图片文件未选中')
            self.textEdit_file.setStyleSheet("background-color: transparent;\n"
                                             "border-color: rgb(0, 170, 255);\n"
                                             "color: rgb(0, 170, 255);\n"
                                             "font: regular 12pt \"华为仿宋\";")
            # self.label_display.setText('正在启动识别系统...\n\nleading')
            QtWidgets.QApplication.processEvents()

            # 清除UI上的label显示
            self.label_plate_result.setText("未知人脸")
            self.label_score_fps.setText("0")
            self.label_score_num.setText("0")
            self.label_score_dis.setText("0")

        else:
            # 定时器已开启，界面回复初始状态
            self.flag_timer = ""
            self.timer_video.stop()
            if self.cap:
                self.cap.release()
            self.label_display.clear()
            self.textEdit_file.setText('图片文件未选中')
            self.textEdit_file.setStyleSheet("background-color: transparent;\n"
                                             "border-color: rgb(0, 170, 255);\n"
                                             "color: rgb(0, 170, 255);\n"
                                             "font: regular 12pt \"华为仿宋\";")
            self.textEdit_camera.setText('实时摄像已关闭')
            self.textEdit_camera.setStyleSheet("background-color: transparent;\n"
                                               "border-color: rgb(0, 170, 255);\n"
                                               "color: rgb(0, 170, 255);\n"
                                               "font: regular 12pt \"华为仿宋\";")
            self.textEdit_video.setText('实时视频已关闭')
            self.textEdit_video.setStyleSheet("background-color: transparent;\n"
                                              "border-color: rgb(0, 170, 255);\n"
                                              "color: rgb(0, 170, 255);\n"
                                              "font: regular 12pt \"华为仿宋\";")
            # self.label_display.setStyleSheet("border-image: url(:/newPrefix/images_test/ini-image.png);")
            self.gif_movie()

            # 清除UI上的label显示
            self.label_plate_result.setText("未知人脸")
            self.label_score_fps.setText("0")
            self.label_score_num.setText("0")
            self.label_score_dis.setText("0")

    def show_camera(self):
        # 定时器槽函数，每隔一段时间执行
        start_time = time.time()
        flag, img_rd = self.cap.read()  # 获取画面
        if flag:
            # image = cv2.flip(image, 1)  # 左右翻转
            image = img_rd.copy()
            # 2. 检测人脸 / Detect faces for frame X (使用混合检测)
            faces = self.detect_faces_hybrid(img_rd)
            self.label_score_num.setText(str(len(faces)))

            # Update cnt for faces in frames
            self.last_face_cnt = self.current_face_cnt
            self.current_face_cnt = len(faces)

            # Update the face name list in last frame
            self.last_face_name = self.current_face_name[:]

            # update frame centroid list
            self.last_centroid = self.current_centroid
            self.current_centroid = []

            # 2.1. if cnt not changes
            if (self.current_face_cnt == self.last_face_cnt) and (
                    self.reclassify_cnt != self.reclassify_interval):

                self.current_face_position = []

                if "未知人脸" in self.current_face_name:
                    self.reclassify_cnt += 1

                if self.current_face_cnt != 0:
                    # 2.1.1 Get ROI positions
                    for k, d in enumerate(faces):
                        self.current_face_position.append(tuple(
                            [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))
                        self.current_centroid.append(
                            [int(faces[k].left() + faces[k].right()) / 2,
                             int(faces[k].top() + faces[k].bottom()) / 2])

                        # 计算矩形框大小 / Compute the size of rectangle box
                        y2 = d.right()
                        x2 = d.bottom()
                        y1 = d.left()
                        x1 = d.top()
                        # 判断人脸区域是否超出画面范围
                        if y2 > img_rd.shape[1]:
                            y2 = img_rd.shape[1]
                        elif x2 > img_rd.shape[0]:
                            x2 = img_rd.shape[0]
                        elif y1 < 0:
                            y1 = 0
                        elif x1 < 0:
                            x1 = 0

                        # 剪切出人脸
                        crop_face = img_rd[x1: x2, y1: y2]
                        self.current_face = crop_face
                        self.disp_face(crop_face)  # 在右侧label中显示检测出的人脸

                        rect = (d.left(), d.top(), d.right(), d.bottom())
                        name_lab = self.current_face_name[k] if self.current_face_name != [] else ""
                        image = self.drawRectBox(image, rect, name_lab)
                        self.label_plate_result.setText(name_lab)

                self.disp_img(image)  # 在画面中显示图像

                # Multi-faces in current frame, use centroid-tracker to track
                if self.current_face_cnt != 1:
                    self.centroid_tracker()

            # 2.2 If cnt of faces changes, 0->1 or 1->0 or ...
            else:
                self.current_face_position = []
                self.current_face_distance = []
                self.current_face_feature = []
                self.reclassify_cnt = 0

                # 2.2.1 Face cnt decreases: 1->0, 2->1, ...
                if self.current_face_cnt == 0:
                    # clear list of names and features
                    self.current_face_name = []
                # 2.2.2 Face cnt increase: 0->1, 0->2, ..., 1->2, ...
                else:
                    self.current_face_name = []
                    for i in range(len(faces)):
                        shape = self.predictor(img_rd, faces[i])
                        self.current_face_feature.append(
                            self.face_reco_model.compute_face_descriptor(img_rd, shape))
                        self.current_face_name.append("未知人脸")

                    # 2.2.2.1 遍历捕获到的图像中所有的人脸 / Traversal all the faces in the database
                    for k in range(len(faces)):
                        self.current_centroid.append(
                            [int(faces[k].left() + faces[k].right()) / 2,
                             int(faces[k].top() + faces[k].bottom()) / 2])

                        self.current_face_distance = []

                        # 2.2.2.2 每个捕获人脸的名字坐标 / Positions of faces captured
                        self.current_face_position.append(tuple(
                            [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))

                        # 2.2.2.3 对于某张人脸，遍历所有存储的人脸特征
                        # For every faces detected, compare the faces in the database
                        for i in range(len(self.face_feature_exist)):
                            # 如果 q 数据不为空
                            if str(self.face_feature_exist[i][0]) != '0.0':
                                e_distance_tmp = self.euclidean_distance(
                                    self.current_face_feature[k],
                                    self.face_feature_exist[i])
                                self.current_face_distance.append(e_distance_tmp)
                            else:
                                # 空数据 person_X
                                self.current_face_distance.append(999999999)

                        # 2.2.2.4 寻找出最小的欧式距离匹配 / Find the one with minimum e distance
                        min_dis = min(self.current_face_distance)
                        similar_person_num = self.current_face_distance.index(min_dis)

                        if min_dis < 0.4:
                            self.current_face_name[k] = self.face_name_exist[similar_person_num]
                        self.label_score_dis.setText(str(round(min_dis, 2)))
                        date_now = datetime.datetime.now().strftime('%m-%d_%H:%M:%S')
                        self.change_table(date_now + "_" + str(self.count), self.current_face_name[k], date_now,
                                          min_dis)

            end_time = time.time()
            if end_time == start_time:
                use_time = 1
            else:
                use_time = end_time - start_time
            fps_rec = int(1.0 / round(use_time, 3))
            self.label_score_fps.setText(str(fps_rec))

    def show_camera_load(self):
        # 定时器槽函数，每隔一段时间执行
        flag, img_rd = self.cap.read()  # 获取画面
        if flag:
            start_time = time.time()  # 计时

            self.current_image = img_rd.copy()
            image = img_rd.copy()
            face_name = self.lineEdit_face_name.text()
            # 使用混合人脸检测器进行人脸检测
            faces = self.detect_faces_hybrid(image)
            self.label_score_num.setText(str(len(faces)))

            if len(faces) != 0:
                # 矩形框 / Show the ROI of faces
                for k, d in enumerate(faces):
                    # 计算矩形框大小 / Compute the size of rectangle box
                    height = (d.bottom() - d.top())
                    width = (d.right() - d.left())
                    hh = int(height / 2)
                    ww = int(width / 2)
                    rect = (d.left() - ww, d.top() - hh, d.right() + ww, d.bottom() + hh)
                    image = self.drawRectBox(image, rect, "正在录入")

                    y2 = d.right() + ww
                    x2 = d.bottom() + hh
                    y1 = d.left() - ww
                    x1 = d.top() - hh
                    # 判断人脸区域是否超出画面范围
                    if y2 > img_rd.shape[1]:
                        y2 = img_rd.shape[1]
                    elif x2 > img_rd.shape[0]:
                        x2 = img_rd.shape[0]
                    elif y1 < 0:
                        y1 = 0
                    elif x1 < 0:
                        x1 = 0

                    # 剪切出人脸
                    crop_face = img_rd[x1: x2, y1: y2]
                    self.current_face = crop_face
                    self.disp_face(crop_face)
                    if self.current_face.any() and face_name != "请在此输入人脸名" and face_name != "":
                        # self.label_loadface.setText("检测到人脸区域，可点击取图按钮以保存！")
                        self.toolButton_get_pic.setEnabled(True)  # 存在人脸时可以取图
            else:
                self.toolButton_get_pic.setEnabled(False)  # 不存在人脸时不可取图

            self.disp_img(image)  # 在画面中显示图像
            end_time = time.time()
            if end_time == start_time:
                use_time = 1
            else:
                use_time = end_time - start_time
            fps_rec = int(1.0 / round(use_time, 3))
            self.label_score_fps.setText(str(fps_rec))  # 更新帧率
            self.label_plate_result.setText("正在录入")
            self.label_score_dis.setText("None")

    def show_video(self):
        # 视频定时器槽函数，每隔一段时间执行该函数
        start_time = time.time()
        flag, img_rd = self.cap_video.read()  # 获取画面
        if flag:
            # image = cv2.flip(image, 1)  # 左右翻转
            image = img_rd.copy()
            # 2. 检测人脸 / Detect faces for frame X (使用混合检测)
            faces = self.detect_faces_hybrid(img_rd)
            self.label_score_num.setText(str(len(faces)))

            # Update cnt for faces in frames
            self.last_face_cnt = self.current_face_cnt
            self.current_face_cnt = len(faces)

            # Update the face name list in last frame
            self.last_face_name = self.current_face_name[:]

            # update frame centroid list
            self.last_centroid = self.current_centroid
            self.current_centroid = []

            # 2.1. if cnt not changes
            if (self.current_face_cnt == self.last_face_cnt) and (
                    self.reclassify_cnt != self.reclassify_interval):

                self.current_face_position = []

                if "未知人脸" in self.current_face_name:
                    self.reclassify_cnt += 1

                if self.current_face_cnt != 0:
                    # 2.1.1 Get ROI positions
                    for k, d in enumerate(faces):
                        self.current_face_position.append(tuple(
                            [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))
                        self.current_centroid.append(
                            [int(faces[k].left() + faces[k].right()) / 2,
                             int(faces[k].top() + faces[k].bottom()) / 2])

                        # 计算矩形框大小 / Compute the size of rectangle box
                        # height = (d.bottom() - d.top())
                        # width = (d.right() - d.left())
                        # hh = int(height / 2)
                        # ww = int(width / 2)
                        # y2 = d.right() + ww
                        # x2 = d.bottom() + hh
                        # y1 = d.left() - ww
                        # x1 = d.top() - hh
                        y2 = d.right()
                        x2 = d.bottom()
                        y1 = d.left()
                        x1 = d.top()
                        # 判断人脸区域是否超出画面范围
                        if y2 > img_rd.shape[1]:
                            y2 = img_rd.shape[1]
                        elif x2 > img_rd.shape[0]:
                            x2 = img_rd.shape[0]
                        elif y1 < 0:
                            y1 = 0
                        elif x1 < 0:
                            x1 = 0

                        # 剪切出人脸
                        crop_face = img_rd[x1: x2, y1: y2]
                        self.current_face = crop_face
                        self.disp_face(crop_face)  # 在右侧label中显示检测出的人脸

                        rect = (d.left(), d.top(), d.right(), d.bottom())
                        name_lab = self.current_face_name[k] if self.current_face_name is not None else ""
                        image = self.drawRectBox(image, rect, name_lab)
                        self.label_plate_result.setText(name_lab)

                self.disp_img(image)  # 在画面中显示图像

                # Multi-faces in current frame, use centroid-tracker to track
                if self.current_face_cnt != 1:
                    self.centroid_tracker()

            # 2.2 If cnt of faces changes, 0->1 or 1->0 or ...
            else:
                self.current_face_position = []
                self.current_face_distance = []
                self.current_face_feature = []
                self.reclassify_cnt = 0

                # 2.2.1 Face cnt decreases: 1->0, 2->1, ...
                if self.current_face_cnt == 0:
                    # clear list of names and features
                    self.current_face_name = []
                # 2.2.2 Face cnt increase: 0->1, 0->2, ..., 1->2, ...
                else:
                    self.current_face_name = []
                    for i in range(len(faces)):
                        shape = self.predictor(img_rd, faces[i])
                        self.current_face_feature.append(
                            self.face_reco_model.compute_face_descriptor(img_rd, shape))
                        self.current_face_name.append("未知人脸")

                    # 2.2.2.1 遍历捕获到的图像中所有的人脸 / Traversal all the faces in the database
                    for k in range(len(faces)):
                        self.current_centroid.append(
                            [int(faces[k].left() + faces[k].right()) / 2,
                             int(faces[k].top() + faces[k].bottom()) / 2])

                        self.current_face_distance = []

                        # 2.2.2.2 每个捕获人脸的名字坐标 / Positions of faces captured
                        self.current_face_position.append(tuple(
                            [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))

                        # 2.2.2.3 对于某张人脸，遍历所有存储的人脸特征
                        # For every faces detected, compare the faces in the database
                        for i in range(len(self.face_feature_exist)):
                            # 如果 q 数据不为空
                            if str(self.face_feature_exist[i][0]) != '0.0':
                                e_distance_tmp = self.euclidean_distance(
                                    self.current_face_feature[k],
                                    self.face_feature_exist[i])
                                self.current_face_distance.append(e_distance_tmp)
                            else:
                                # 空数据 person_X
                                self.current_face_distance.append(999999999)

                        # 2.2.2.4 寻找出最小的欧式距离匹配 / Find the one with minimum e distance
                        min_dis = min(self.current_face_distance)
                        similar_person_num = self.current_face_distance.index(min_dis)

                        if min_dis < 0.6:
                            self.current_face_name[k] = self.face_name_exist[similar_person_num]
                        self.label_score_dis.setText(str(round(min_dis, 2)))
                        # 将识别记录到表格中
                        date_now = datetime.datetime.now().strftime('%m-%d_%H:%M:%S')
                        self.change_table(date_now + "_" + str(self.count), self.current_face_name[k], date_now,
                                          min_dis)

            end_time = time.time()
            if end_time - start_time == 0:
                use_time = 1
            else:
                use_time = end_time - start_time
            fps_rec = int(1.0 / round(use_time, 3))
            self.label_score_fps.setText(str(fps_rec))

    def centroid_tracker(self):
        for i in range(len(self.current_centroid)):
            distance_current_person = []
            # 计算不同对象间的距离
            for j in range(len(self.last_centroid)):
                self.last_current_distance = self.euclidean_distance(
                    self.current_centroid[i], self.last_centroid[j])

                distance_current_person.append(
                    self.last_current_distance)

            last_frame_num = distance_current_person.index(
                min(distance_current_person))
            self.current_face_name[i] = self.last_face_name[last_frame_num]

    def do_choose_file(self):
        # 选择图片识别时运行函数
        if self.path != '':
            self.label_display.clear()
            self.label_pic_newface.clear()
            QtWidgets.QApplication.processEvents()
            exist_flag = self.get_face_database()  # 获取已存在人脸的特征
            img_rd = self.cv_imread(self.path)  # 读取选择的图片

            # 使用混合人脸检测器进行人脸检测
            image = img_rd.copy()
            faces = self.detect_faces_hybrid(image)

            if len(faces) > 0:
                self.label_score_num.setText(str(len(faces)))
                # 矩形框 / Show the ROI of faces
                face_feature_list = []
                face_name_list = []
                face_position_list = []
                start_time = time.time()

                for k, d in enumerate(faces):
                    # 计算矩形框大小 / Compute the size of rectangle box
                    height = (d.bottom() - d.top())
                    width = (d.right() - d.left())
                    hh = int(height / 2)
                    ww = int(width / 2)

                    y2 = d.right() + ww
                    x2 = d.bottom() + hh
                    y1 = d.left() - ww
                    x1 = d.top() - hh
                    # 判断人脸区域是否超出画面范围
                    if y2 > img_rd.shape[1]:
                        y2 = img_rd.shape[1]
                    elif x2 > img_rd.shape[0]:
                        x2 = img_rd.shape[0]
                    elif y1 < 0:
                        y1 = 0
                    elif x1 < 0:
                        x1 = 0

                    # 剪切出人脸
                    crop_face = img_rd[x1: x2, y1: y2]
                    # 获取人脸特征
                    shape = self.predictor(img_rd, d)
                    face_feature_list.append(self.face_reco_model.compute_face_descriptor(img_rd, shape))

                    self.current_face = crop_face
                    self.disp_face(crop_face)  # 在右侧label中显示检测出的人脸

                if exist_flag:  # 获取已存在人脸的特征
                    for k in range(len(faces)):
                        # 初始化
                        face_name_list.append("未知人脸")

                        # 每个捕获人脸的名字坐标
                        face_position_list.append(tuple(
                            [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))

                        # 对于某张人脸，遍历所有存储的人脸特征
                        current_distance_list = []
                        for i in range(len(self.face_feature_exist)):
                            # 如果 person_X 数据不为空
                            if str(self.face_feature_exist[i][0]) != '0.0':
                                e_distance_tmp = self.euclidean_distance(face_feature_list[k],
                                                                         self.face_feature_exist[i])
                                current_distance_list.append(e_distance_tmp)
                            else:
                                # 空数据 person_X
                                current_distance_list.append(999999999)

                        # 寻找出最小的欧式距离匹配
                        min_dis = min(current_distance_list)
                        similar_person_num = current_distance_list.index(min_dis)
                        if min_dis < 0.4:
                            face_name_list[k] = self.face_name_exist[similar_person_num]
                        self.label_score_dis.setText(str(round(min_dis, 2)))

                        # 将识别记录到表格中
                        date_now = datetime.datetime.now().strftime('%m-%d_%H:%M:%S')
                        self.change_table(date_now + "_" + str(self.count), face_name_list[k], date_now,
                                          min_dis)

                end_time = time.time()
                fps_rec = int(1.0 / round((end_time - start_time), 3))
                self.label_score_fps.setText(str(fps_rec))

                for k, d in enumerate(faces):
                    # 计算矩形框大小 / Compute the size of rectangle box
                    # height = (d.bottom() - d.top())
                    # width = (d.right() - d.left())
                    # hh = int(height / 2)
                    # ww = int(width / 2)
                    rect = (d.left(), d.top(), d.right(), d.bottom())
                    image = self.drawRectBox(image, rect, face_name_list[k])
                    self.label_plate_result.setText(face_name_list[k])

                self.disp_img(image)  # 在画面中显示图像

            else:
                self.label_display.setText('未能检测到人脸，请重新选择！')
                # self.label_display.setStyleSheet("border-image: url(:/newPrefix/images_test/ini-image.png);")
                self.gif_movie()

        else:
            self.flag_timer = ""
            self.textEdit_file.setText('图片文件未选中或不符合！')
            self.textEdit_file.setStyleSheet("background-color: transparent;\n"
                                             "border-color: rgb(0, 170, 255);\n"
                                             "color: rgb(0, 170, 255);\n"
                                             "font: regular 12pt \"华为仿宋\";")

    def run_rec(self):
        # 点击开始运行按钮执行函数
        if self.flag_timer == "image":
            self.do_choose_file()

        elif self.flag_timer == "video":
            if not self.timer_video.isActive():
                # self.count = 0
                # self.tableWidget.clearContents()
                # 打开定时器
                self.exist_flag = self.get_face_database()
                self.timer_video.start(30)
            else:
                self.timer_video.stop()

        elif self.flag_timer == "camera":
            if not self.timer_camera.isActive():
                # self.count = 0
                # self.tableWidget.clearContents()
                QtWidgets.QApplication.processEvents()
                self.exist_flag = self.get_face_database()  # 获取已存在人脸的特征
                self.timer_camera.start(30)
            else:
                self.timer_camera.stop()
                self.flag_timer = ""
                if self.cap:
                    self.cap.release()
                self.textEdit_camera.setText("实时摄像已关闭")
                self.textEdit_camera.setStyleSheet("background-color: transparent;\n"
                                                   "border-color: rgb(0, 170, 255);\n"
                                                   "color: rgb(0, 170, 255);\n"
                                                   "font: regular 12pt \"华为仿宋\";")
                QtWidgets.QApplication.processEvents()

        else:
            self.textEdit_file.setText('图片文件未选中')
            self.textEdit_file.setStyleSheet("background-color: transparent;\n"
                                             "border-color: rgb(0, 170, 255);\n"
                                             "color: rgb(0, 170, 255);\n"
                                             "font: regular 12pt \"华为仿宋\";")
            self.textEdit_camera.setText("实时摄像已关闭")
            self.textEdit_camera.setStyleSheet("background-color: transparent;\n"
                                               "border-color: rgb(0, 170, 255);\n"
                                               "color: rgb(0, 170, 255);\n"
                                               "font: regular 12pt \"华为仿宋\";")
            # self.textEdit_model
            self.textEdit_video.setText('实时视频已关闭')
            self.textEdit_video.setStyleSheet("background-color: transparent;\n"
                                              "border-color: rgb(0, 170, 255);\n"
                                              "color: rgb(0, 170, 255);\n"
                                              "font: regular 12pt \"华为仿宋\";")
            self.label_display.clear()
            # self.label_display.setStyleSheet("border-image: url(:/newPrefix/images_test/ini-image.png);")
            self.gif_movie()

            self.label_pic_newface.clear()

            # self.count = 0
            # self.tableWidget.clearContents()
            # 清除UI上的label显示
            self.label_plate_result.setText("未知人脸")
            self.label_score_fps.setText("0")
            self.label_score_num.setText("0")
            self.label_score_dis.setText("0")

    def table_review(self, row, col):
        self.col_row = [row, col]

    def delete_doing(self):
        if self.col_row:

            # 弹出询问对话框
            msg = QtWidgets.QMessageBox.question(self.centralwidget, u"Warning",
                                                 u"确定删除该人脸数据吗?",
                                                 buttons=QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                                 defaultButton=QtWidgets.QMessageBox.No)
            if msg == QtWidgets.QMessageBox.Yes:
                # 获取当前行中的文件夹路径
                row, col = self.col_row
                r_path = self.tableWidget_mana.item(row, 1).text()

                # 删除该文件夹
                if os.path.exists(r_path):
                    shutil.rmtree(r_path)
                    self.label_mana_info.setText("已删除该人脸")
                    self.do_update_face()
                    self.label_mana_info.setText("开始重新录入")
                    person_list = os.listdir(self.path_face_dir)
                    with open("../data/features_all.csv", "w", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        for person in person_list:
                            features_list = []
                            photos_list = os.listdir(self.path_face_dir + "/" + person)
                            if photos_list:
                                for photo in photos_list:
                                    features_128D = self.extract_features(
                                        self.path_face_dir + "/" + person + "/" + photo)
                                    self.label_mana_info.setText(photo + "已录入！")
                                    QtWidgets.QApplication.processEvents()

                                    if features_128D == 0:
                                        continue
                                    else:
                                        features_list.append(features_128D)
                            if features_list:
                                features_mean = np.array(features_list).mean(axis=0)
                            else:
                                features_mean = np.zeros(128, dtype=int, order='C')
                            str_face = [person]
                            str_face.extend(list(features_mean))
                            writer.writerow(str_face)
                        self.label_mana_info.setText("已重新录入人脸！")

    def update_face(self):
        # 点击更新按钮后执行
        self.count_face = 0
        # 读取文件夹下存在的人脸数目
        person_list = os.listdir(self.path_face_dir)  # 人脸目录
        # 计算人脸数目
        num_faces = len(person_list)
        self.label_mana_face_num.setText(str(num_faces))

        timestamp = os.path.getmtime(self.path_face_dir)
        timeStruct = time.localtime(timestamp)
        timeStruct = time.strftime('%m-%d %H:%M:%S', timeStruct)
        self.label_mana_time.setText(timeStruct)

        # 将人脸信息显示在表格中
        for dir_face in person_list:
            face_name = dir_face
            dir_path = self.path_face_dir + dir_face + "/"
            # self.count_face += 1
            timestamp = os.path.getmtime(self.path_face_dir + dir_face)
            timeStruct = time.localtime(timestamp)
            timeStruct = time.strftime('%m-%d %H:%M:%S', timeStruct)
            self.change_table_mana(dir_path, face_name, timeStruct)

    def do_update_face(self):
        # 点击更新按钮后执行
        self.count_face = 0
        # 读取文件夹下存在的人脸数目
        person_list = os.listdir(self.path_face_dir)  # 人脸目录
        # 计算人脸数目
        num_faces = len(person_list)
        self.label_mana_face_num.setText(str(num_faces))

        timestamp = os.path.getmtime(self.path_face_dir)
        timeStruct = time.localtime(timestamp)
        timeStruct = time.strftime('%m-%d %H:%M:%S', timeStruct)
        self.label_mana_time.setText(timeStruct)

        # 将人脸信息显示在表格中
        for dir_face in person_list:
            face_name = dir_face
            dir_path = self.path_face_dir + dir_face + "/"
            # self.count_face += 1
            timestamp = os.path.getmtime(self.path_face_dir + dir_face)
            timeStruct = time.localtime(timestamp)
            timeStruct = time.strftime('%m-%d %H:%M:%S', timeStruct)
            self.change_table_mana(dir_path, face_name, timeStruct)
        self.label_mana_info.setText("人脸已更新")

    def change_table_mana(self, path, face_name, time_now):
        # 更新表格记录
        self.count_face += 1  # 每识别出结果增加一条记录
        if self.count_face >= 1:
            self.tableWidget_mana.setRowCount(self.count_face)
        newItem = QTableWidgetItem(str(self.count_face))  # 在表格中记录序号
        newItem.setTextAlignment(Qt.AlignCenter)
        self.tableWidget_mana.setItem(self.count_face - 1, 0, newItem)

        newItem = QTableWidgetItem(path)  # 在表格中记录人脸路径
        newItem.setTextAlignment(Qt.AlignVCenter)
        self.tableWidget_mana.setItem(self.count_face - 1, 1, newItem)

        newItem = QTableWidgetItem(face_name)  # 记录人脸名字在表格中
        newItem.setTextAlignment(Qt.AlignCenter)
        self.tableWidget_mana.setItem(self.count_face - 1, 2, newItem)
        self.tableWidget_mana.setCurrentItem(newItem)

        newItem = QTableWidgetItem(time_now)  # 记录修改时间在表格中
        newItem.setTextAlignment(Qt.AlignCenter)
        self.tableWidget_mana.setItem(self.count_face - 1, 3, newItem)
        self.tableWidget_mana.setCurrentItem(newItem)

    def load_img_doing(self):
        if self.timer_camera_load.isActive():
            self.timer_camera_load.stop()
        if self.cap:
            self.cap.release()

        person_list = os.listdir(self.path_face_dir)

        with open("../data/features_all.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for person in person_list:
                features_list = []
                photos_list = os.listdir(self.path_face_dir + "/" + person)
                if photos_list:
                    for photo in photos_list:
                        features_128D = self.extract_features(self.path_face_dir + "/" + person + "/" + photo)
                        self.label_loadface.setText("图片" + photo + "已录入！")
                        QtWidgets.QApplication.processEvents()

                        if features_128D == 0:
                            continue
                        else:
                            features_list.append(features_128D)
                if features_list:
                    features_mean = np.array(features_list).mean(axis=0)
                else:
                    features_mean = np.zeros(128, dtype=int, order='C')
                str_face = [person]
                str_face.extend(list(features_mean))
                writer.writerow(str_face)
            self.label_loadface.setText("已完成人脸录入！")

    @staticmethod
    def euclidean_distance(feature_1, feature_2):
        # 计算两个128D向量间的欧式距离
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    def extract_features(self, path_img):
        img_rd = self.cv_imread(path_img)
        faces = self.detect_faces_hybrid(img_rd)
        if len(faces) != 0:
            shape = self.predictor(img_rd, faces[0])
            face_descriptor = self.face_reco_model.compute_face_descriptor(img_rd, shape)
        else:
            face_descriptor = 0
        return face_descriptor

    def get_face_database(self):
        self.face_feature_exist = []
        self.face_name_exist = []
        if os.path.exists("../data/features_all.csv"):
            path_features_known_csv = "../data/features_all.csv"
            csv_rd = pd.read_csv(path_features_known_csv, header=None, encoding='gb2312')
            for i in range(csv_rd.shape[0]):
                features_someone_arr = []
                for j in range(1, 129):
                    if csv_rd.iloc[i][j] == '':
                        features_someone_arr.append('0')
                    else:
                        features_someone_arr.append(csv_rd.iloc[i][j])
                self.face_feature_exist.append(features_someone_arr)

                if csv_rd.iloc[i][0] == '':
                    self.face_name_exist.append("未知人脸")
                else:
                    self.face_name_exist.append(csv_rd.iloc[i][0])

            return 1
        else:
            return 0

    def get_img_doing(self):
        # 将识别出的人脸保存至文件夹中
        face_name = self.lineEdit_face_name.text()
        cur_path = self.path_face_dir + face_name + "/"
        files_path = glob.glob(pathname=cur_path + face_name + '_*.jpg')  # 获取当前文件夹下文件个数
        img_num = len(files_path)
        if self.current_face is not None:
            if self.current_face.any():
                img_num += 1
                cv2.imencode(".jpg", self.current_face)[1].tofile(cur_path + face_name + "_" + str(img_num) + ".jpg")
                self.label_loadface.setText(
                    "图片" + face_name + "_" + str(img_num) + ".jpg" + "已保存，请继续添加或录入！")

        files_path = glob.glob(pathname=cur_path + face_name + '_*.jpg')  # 获取当前文件夹下文件个数
        # 清除图像
        # self.label_display.clear()
        # self.label_pic_newface.clear()
        self.disp_load_face(self.current_face)

        self.current_face = None
        self.current_face = None
        self.toolButton_get_pic.setEnabled(False)

        if len(files_path) > 0:
            self.toolButton_load_pic.setEnabled(True)  # 文件夹下存在照片时可以录入

    def new_face_doing(self):
        self.toolButton_get_pic.setEnabled(False)
        self.toolButton_load_pic.setEnabled(False)
        # 清除图像
        self.label_display.clear()
        self.label_pic_newface.clear()
        self.current_face = None
        # 读取输入框的文本
        face_name = self.lineEdit_face_name.text()
        if face_name != "请在此输入人脸名" and face_name != "":

            # 新建文件夹
            if not os.path.isdir(self.path_face_dir + face_name):
                os.mkdir(self.path_face_dir + face_name)  # 不存在则创建新文件夹
                # 在界面中提示已经新建
                self.label_new_res.setText("输入的是新人名，已新建人脸文件夹！")
                self.label_loadface.setText("新建完成，请点击左侧按钮选择图片或摄像录入！")

            else:
                self.label_new_res.setText("该人名已存在，继续录入将写入更多照片！")
                self.label_loadface.setText("文件夹存在，请点击左侧按钮选择图片或摄像取图！")
                for dir_path, listdir, files in os.walk(self.path_face_dir + face_name):
                    for file in files:
                        ret = re.match(face_name + "_*", file)
                        if ret:
                            self.toolButton_load_pic.setEnabled(True)  # 文件夹下存在照片时可以录入
                            self.label_loadface.setText("文件夹下存在照片，可点击录入或取图！")

            if self.current_face is not None:
                if self.current_face.any():
                    self.toolButton_get_pic.setEnabled(True)  # 存在人脸时可以取图

            # 文件名已修改，可以开始选择图片或摄像按钮
            self.toolButton_file_2.setEnabled(True)
            self.toolButton_camera_load.setEnabled(True)
        else:
            self.label_new_res.setText("请在左下角文本框中输入人脸名字！")
            self.label_loadface.setText("请先输入要录入的人脸名字！")

    def choose_file(self):
        # 使用文件选择对话框选择图片
        fileName_choose, filetype = QFileDialog.getOpenFileName(
            self.centralwidget, "选取图片文件",
            self.path,  # 起始路径
            "图片(*.jpg;*.jpeg;*.png)")  # 文件类型
        self.path = fileName_choose  # 保存路径
        if self.path != '':
            img_rd = self.cv_imread(self.path)  # 读取选择的图片
            image = img_rd.copy()
            face_name = self.lineEdit_face_name.text()
            # 使用混合人脸检测器进行人脸检测
            faces = self.detect_faces_hybrid(image)
            if len(faces) != 0:
                # 矩形框 / Show the ROI of faces
                for k, d in enumerate(faces):
                    # 计算矩形框大小 / Compute the size of rectangle box
                    height = (d.bottom() - d.top())
                    width = (d.right() - d.left())
                    hh = int(height / 2)
                    ww = int(width / 2)
                    rect = (d.left() - ww, d.top() - hh, d.right() + ww, d.bottom() + hh)
                    image = self.drawRectBox(image, rect, "未知")

                    y2 = d.right() + ww
                    x2 = d.bottom() + hh
                    y1 = d.left() - ww
                    x1 = d.top() - hh
                    # 判断人脸区域是否超出画面范围
                    if y2 > img_rd.shape[1]:
                        y2 = img_rd.shape[1]
                    elif x2 > img_rd.shape[0]:
                        x2 = img_rd.shape[0]
                    elif y1 < 0:
                        y1 = 0
                    elif x1 < 0:
                        x1 = 0

                    # 剪切出人脸
                    crop_face = img_rd[x1: x2, y1: y2]
                    self.current_face = crop_face
                    self.disp_face(crop_face)

                    if self.current_face.any() and face_name != "请在此输入人脸名" and face_name != "":
                        self.label_loadface.setText("检测到人脸区域，可点击取图按钮以保存！")
                        self.toolButton_get_pic.setEnabled(True)  # 存在人脸时可以取图

                self.disp_img(image)  # 在画面中显示图像

    def disp_load_face(self, image):
        # self.label_pic_org.clear()
        image = cv2.resize(image, (500, 500))  # 设定图像尺寸为显示界面大小
        show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        a = QtGui.QPixmap.fromImage(showImage)
        self.label_pic_org.setPixmap(a)
        self.label_pic_org.setScaledContents(True)
        QtWidgets.QApplication.processEvents()

    def disp_img(self, image):
        # self.label_display.clear()
        image = cv2.resize(image, (500, 500))  # 设定图像尺寸为显示界面大小
        show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        a = QtGui.QPixmap.fromImage(showImage)
        self.label_display.setPixmap(a)
        self.label_display.setScaledContents(True)
        QtWidgets.QApplication.processEvents()

    def disp_face(self, image):
        self.label_pic_newface.clear()
        if image.any():
            image = cv2.resize(image, (200, 200))  # 设定图像尺寸为显示界面大小
            show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
            a = QtGui.QPixmap.fromImage(showImage)
            self.label_pic_newface.setPixmap(a)
            self.label_pic_newface.setScaledContents(True)
            QtWidgets.QApplication.processEvents()

    def drawRectBox(self, image, rect, addText):
        cv2.rectangle(image, (int(round(rect[0])), int(round(rect[1]))),
                      (int(round(rect[2])), int(round(rect[3]))),
                      (0, 0, 255), 2)
        cv2.rectangle(image, (int(rect[0] - 1), int(rect[1]) - 16), (int(rect[0] + 125), int(rect[1])), (0, 0, 255), -1,
                      cv2.LINE_AA)
        img = Image.fromarray(image)
        draw = ImageDraw.Draw(img)
        draw.text((int(rect[0] + 1), int(rect[1] - 16)), addText, (255, 255, 255), font=self.fontC)
        imagex = np.array(img)
        return imagex

    @staticmethod
    def cv_imread(filePath):
        # 读取图片
        # cv_img = cv2.imread(filePath)
        cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
        # imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
        # cv_img = cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
        if len(cv_img.shape) > 2:
            if cv_img.shape[2] > 3:
                cv_img = cv_img[:, :, :3]
        return cv_img

    def change_size_load(self):
        self.toolButton_run_load.setGeometry(26, 218, 280, 70)
        self.toolButton_run_rec.setGeometry(66, 324, 199, 49)
        self.toolButton_run_manage.setGeometry(66, 414, 199, 49)
        self.tabWidget.setCurrentIndex(1)
        self.tabWidget.setTabVisible(0, False)
        self.tabWidget.setTabVisible(1, True)
        self.tabWidget.setTabVisible(2, False)

        self.flag_timer = ""
        if self.timer_camera.isActive():
            self.timer_camera.stop()
        if self.timer_video.isActive():
            self.timer_video.stop()
        if self.cap:
            self.cap.release()
        if self.cap_video:
            self.cap_video.release()  # 释放视频画面帧

        QtWidgets.QApplication.processEvents()
        self.label_display.clear()
        self.gif_movie()

        self.label_pic_newface.clear()
        self.label_pic_org.clear()

    def change_size_rec(self):
        self.toolButton_run_load.setGeometry(66, 236, 199, 49)
        self.toolButton_run_rec.setGeometry(26, 312, 280, 70)
        self.toolButton_run_manage.setGeometry(66, 414, 199, 49)
        self.tabWidget.setCurrentIndex(0)
        self.tabWidget.setTabVisible(0, True)
        self.tabWidget.setTabVisible(1, False)
        self.tabWidget.setTabVisible(2, False)

        if self.timer_camera_load.isActive():
            self.timer_camera_load.stop()
        if self.cap:
            self.cap.release()

        QtWidgets.QApplication.processEvents()
        self.label_display.clear()
        self.gif_movie()

        self.label_pic_newface.clear()

        self.lineEdit_face_name.setText("请在此输入人脸名")
        self.label_new_res.setText("等待新建人脸文件夹")
        self.label_loadface.setText("等待点击以录入人脸")
        self.toolButton_get_pic.setEnabled(False)
        self.toolButton_load_pic.setEnabled(False)

    def change_size_mana(self):
        self.toolButton_run_load.setGeometry(66, 236, 199, 49)
        self.toolButton_run_rec.setGeometry(66, 324, 199, 49)
        self.toolButton_run_manage.setGeometry(26, 410, 280, 70)
        self.tabWidget.setCurrentIndex(2)
        self.tabWidget.setTabVisible(0, False)
        self.tabWidget.setTabVisible(1, False)
        self.tabWidget.setTabVisible(2, True)

        self.update_face()
