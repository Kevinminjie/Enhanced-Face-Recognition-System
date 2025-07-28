#!/usr/bin/env python
# encoding: utf-8
'''
增强版人脸识别系统 - 集成ArcFace和注意力机制
基于原有FaceRecognition.py进行改进，提升在口罩、帽子、暗光环境下的识别准确性
'''

import csv
import glob
import os
import re
import shutil
import time
import warnings
from os import getcwd
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
from facenet_pytorch import MTCNN

# 导入增强模块
try:
    import sys
    sys.path.append('..')
    from improved_face_recognition import ImprovedFaceRecognizer, ImprovedFacePreprocessor
    from face_recognition_integration import EnhancedFaceRecognition
    ENHANCED_MODE = True
    print("✓ 增强模式已启用 - ArcFace + 注意力机制")
except ImportError as e:
    print(f"⚠ 增强模块导入失败，使用标准模式: {e}")
    ENHANCED_MODE = False

# 忽略警告
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings(action='ignore')

# 初始化MTCNN
mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')

def augment_image(img):
    """数据增强函数"""
    flip = cv2.flip(img, 1)  # 镜像
    blur = cv2.GaussianBlur(img, (5, 5), 0)  # 高斯模糊
    return [img, flip, blur]

class EnhancedModelLoader(QThread):
    """增强版模型加载器"""
    models_loaded = pyqtSignal(object, object, object, object)  # 增加增强识别器信号

    def run(self):
        try:
            # 初始化 MTCNN 检测器
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            mtcnn = MTCNN(keep_all=True, device=device)

            # dlib 的人脸关键点预测器和识别模型
            predictor = dlib.shape_predictor('../data/data_dlib/shape_predictor_68_face_landmarks.dat')
            reco_model = dlib.face_recognition_model_v1("../data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

            # 初始化增强识别器
            enhanced_recognizer = None
            if ENHANCED_MODE:
                try:
                    enhanced_recognizer = EnhancedFaceRecognition()
                    print("✓ 增强识别器初始化成功")
                except Exception as e:
                    print(f"⚠ 增强识别器初始化失败: {e}")
                    enhanced_recognizer = None

            # 发射模型加载完成信号
            self.models_loaded.emit(mtcnn, predictor, reco_model, enhanced_recognizer)
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            self.models_loaded.emit(None, None, None, None)

from FaceRecognition_UI import Ui_MainWindow

class Face_MainWindow(Ui_MainWindow):
    """增强版人脸识别主窗口"""
    
    def __init__(self, MainWindow):
        self.path_face_dir = "../data/database_faces/"
        self.fontC = ImageFont.truetype("./Font/platech.ttf", 14, 0)

        self.cap_video = None  # 视频流对象
        self.path = getcwd()
        self.video_path = getcwd()

        # 定时器
        self.timer_camera = QtCore.QTimer()
        self.timer_camera_load = QtCore.QTimer()
        self.timer_video = QtCore.QTimer()
        self.flag_timer = ""

        self.CAM_NUM = 0
        self.cap = cv2.VideoCapture(self.CAM_NUM)
        self.cap_video = None

        self.current_image = None
        self.current_face = None

        # 初始化基本属性
        self.count = 0
        self.count_face = 0
        self.col_row = []
        
        # 传统dlib模型
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = None
        self.face_reco_model = None
        
        # 增强识别器
        self.enhanced_recognizer = None
        self.use_enhanced_mode = ENHANCED_MODE
        
        # 界面初始化
        self.setupUi(MainWindow)
        self.retranslateUi(MainWindow)
        self.resetUi()
        self.slot_init()
        
        # 人脸数据存储
        self.face_feature_exist = []
        self.face_name_exist = []
        
        # 跟踪相关变量
        self.last_centroid = []
        self.current_centroid = []
        self.last_face_name = []
        self.current_face_name = []
        self.last_face_cnt = 0
        self.current_face_cnt = 0
        self.current_face_position = []
        self.current_face_feature = []
        self.reclassify_cnt = 0
        self.reclassify_interval = 20
        self.last_current_distance = 0
        self.current_face_distance = []
        self.exist_flag = None
        
        # 启动模型加载
        self.model_loader = EnhancedModelLoader()
        self.model_loader.models_loaded.connect(self.on_models_loaded)
        self.model_loader.start()
        
        print("增强版人脸识别系统初始化完成")
    
    def on_models_loaded(self, mtcnn, predictor, reco_model, enhanced_recognizer):
        """模型加载完成回调"""
        self.predictor = predictor
        self.face_reco_model = reco_model
        self.enhanced_recognizer = enhanced_recognizer
        
        if enhanced_recognizer is not None:
            self.use_enhanced_mode = True
            print("✓ 增强模式已激活")
        else:
            self.use_enhanced_mode = False
            print("⚠ 使用标准模式")
    
    def enhanced_face_detection(self, image):
        """增强的人脸检测方法"""
        if self.use_enhanced_mode and self.enhanced_recognizer:
            try:
                # 使用增强检测
                result = self.enhanced_recognizer.enhanced_face_detection(image)
                if result is not None:
                    return result
            except Exception as e:
                print(f"增强检测失败，回退到标准模式: {e}")
        
        # 回退到标准检测
        return self.standard_face_detection(image)
    
    def standard_face_detection(self, image):
        """标准人脸检测方法"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray, 1)
            
            if len(faces) > 0:
                face = faces[0]
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                face_region = image[y:y+h, x:x+w]
                face_resized = cv2.resize(face_region, (112, 112))
                return face_resized
            
            return None
        except Exception as e:
            print(f"标准检测失败: {e}")
            return None
    
    def enhanced_face_recognition(self, image):
        """增强的人脸识别方法"""
        if self.use_enhanced_mode and self.enhanced_recognizer:
            try:
                # 使用增强识别
                result = self.enhanced_recognizer.recognize_with_enhancement(image)
                return result['name'], result['confidence']
            except Exception as e:
                print(f"增强识别失败，回退到标准模式: {e}")
        
        # 回退到标准识别
        return self.standard_face_recognition(image)
    
    def standard_face_recognition(self, image):
        """标准人脸识别方法"""
        try:
            if self.predictor is None or self.face_reco_model is None:
                return "模型未加载", 0.0
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray, 1)
            
            if len(faces) == 0:
                return "未检测到人脸", 0.0
            
            face = faces[0]
            landmarks = self.predictor(gray, face)
            face_descriptor = self.face_reco_model.compute_face_descriptor(gray, landmarks)
            face_descriptor = np.array(face_descriptor)
            
            # 与数据库中的人脸进行比较
            if len(self.face_feature_exist) == 0:
                return "数据库为空", 0.0
            
            distances = []
            for feature in self.face_feature_exist:
                distance = np.linalg.norm(face_descriptor - feature)
                distances.append(distance)
            
            min_distance = min(distances)
            min_index = distances.index(min_distance)
            
            threshold = 0.6
            if min_distance < threshold:
                confidence = 1.0 - min_distance
                return self.face_name_exist[min_index], confidence
            else:
                return "未知人脸", 0.0
                
        except Exception as e:
            print(f"标准识别失败: {e}")
            return "识别错误", 0.0
    
    def add_face_enhanced(self, name, image):
        """增强的人脸添加方法"""
        if self.use_enhanced_mode and self.enhanced_recognizer:
            try:
                # 使用增强方法添加人脸
                success = self.enhanced_recognizer.add_face_with_enhancement(name, image)
                if success:
                    print(f"✓ 使用增强模式成功添加 {name}")
                    return True
            except Exception as e:
                print(f"增强添加失败，回退到标准模式: {e}")
        
        # 回退到标准方法
        return self.add_face_standard(name, image)
    
    def add_face_standard(self, name, image):
        """标准人脸添加方法"""
        try:
            if self.predictor is None or self.face_reco_model is None:
                print("模型未加载")
                return False
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray, 1)
            
            if len(faces) == 0:
                print("未检测到人脸")
                return False
            
            face = faces[0]
            landmarks = self.predictor(gray, face)
            face_descriptor = self.face_reco_model.compute_face_descriptor(gray, landmarks)
            face_descriptor = np.array(face_descriptor)
            
            # 添加到数据库
            self.face_feature_exist.append(face_descriptor)
            self.face_name_exist.append(name)
            
            print(f"✓ 使用标准模式成功添加 {name}")
            return True
            
        except Exception as e:
            print(f"标准添加失败: {e}")
            return False
    
    def process_video_frame_enhanced(self, frame):
        """增强的视频帧处理"""
        if self.use_enhanced_mode and self.enhanced_recognizer:
            try:
                # 使用增强处理
                processed_frame, name, confidence = self.enhanced_recognizer.process_video_frame(frame)
                return processed_frame, name, confidence
            except Exception as e:
                print(f"增强处理失败，回退到标准模式: {e}")
        
        # 回退到标准处理
        return self.process_video_frame_standard(frame)
    
    def process_video_frame_standard(self, frame):
        """标准视频帧处理"""
        try:
            name, confidence = self.enhanced_face_recognition(frame)
            
            # 在帧上绘制结果
            annotated_frame = self.draw_recognition_result(frame, name, confidence)
            
            return annotated_frame, name, confidence
            
        except Exception as e:
            print(f"标准处理失败: {e}")
            return frame, "处理错误", 0.0
    
    def draw_recognition_result(self, image, name, confidence):
        """在图像上绘制识别结果"""
        try:
            annotated = image.copy()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray, 1)
            
            for face in faces:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                
                # 绘制人脸框
                color = (0, 255, 0) if confidence > 0.6 else (0, 0, 255)
                cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
                
                # 绘制识别结果文本
                text = f"{name}: {confidence:.2f}"
                cv2.putText(annotated, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            return annotated
            
        except Exception as e:
            print(f"绘制结果失败: {e}")
            return image
    
    # 以下是原有方法的增强版本
    def resetUi(self):
        """重置UI界面"""
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

        # 设置初始按钮状态
        self.toolButton_get_pic.setEnabled(False)
        self.toolButton_load_pic.setEnabled(False)
        self.toolButton_file_2.setEnabled(False)
        self.toolButton_camera_load.setEnabled(False)

        # 设置界面动画
        self.gif_movie()
        
        # 显示增强模式状态
        if self.use_enhanced_mode:
            print("🚀 界面已启用增强模式")
    
    def gif_movie(self):
        """设置界面动画"""
        gif = QMovie(':/newPrefix/images_test/face_rec.gif')
        self.label_display.setMovie(gif)
        self.label_display.setScaledContents(True)
        gif.start()
    
    def slot_init(self):
        """初始化槽函数连接"""
        try:
            # 界面切换按钮
            self.toolButton_run_load.clicked.connect(self.change_size_load)
            self.toolButton_run_rec.clicked.connect(self.change_size_rec)
            self.toolButton_run_manage.clicked.connect(self.change_size_mana)
            
            # 录入界面按钮
            self.toolButton_new_folder.clicked.connect(self.new_face_doing)
            self.toolButton_file_2.clicked.connect(self.choose_file)
            self.toolButton_get_pic.clicked.connect(self.get_img_doing)
            self.toolButton_load_pic.clicked.connect(self.load_img_doing)
            self.toolButton_camera_load.clicked.connect(self.button_open_camera_load)
            self.timer_camera_load.timeout.connect(self.show_camera_load)
            
            # 识别界面按钮
            self.toolButton_file.clicked.connect(self.choose_rec_img)
            self.toolButton_runing.clicked.connect(self.run_rec)
            self.toolButton_camera.clicked.connect(self.button_open_camera_click)
            self.toolButton_video.clicked.connect(self.button_open_video_click)
            
            # 定时器连接 - 根据模式选择处理函数
            if self.use_enhanced_mode:
                self.timer_camera.timeout.connect(self.show_camera_enhanced)
                self.timer_video.timeout.connect(self.show_video_enhanced)
            else:
                self.timer_camera.timeout.connect(self.show_camera)
                self.timer_video.timeout.connect(self.show_video)
            
            # 管理界面按钮
            self.toolButton_mana_update.clicked.connect(self.do_update_face)
            self.tableWidget_mana.cellPressed.connect(self.table_review)
            self.toolButton_mana_delete.clicked.connect(self.delete_doing)
            
            print("✓ 信号槽连接完成")
        except Exception as e:
            print(f"信号槽连接异常: {str(e)}")
    
    def show_camera_enhanced(self):
        """增强的摄像头显示方法"""
        try:
            flag, image = self.cap.read()
            if not flag:
                return
            
            # 使用增强处理
            processed_frame, name, confidence = self.process_video_frame_enhanced(image)
            
            # 更新UI显示
            self.update_ui_with_result(processed_frame, name, confidence)
            
        except Exception as e:
            print(f"摄像头显示失败: {e}")
    
    def show_video_enhanced(self):
        """增强的视频显示方法"""
        try:
            if self.cap_video is None:
                return
                
            flag, image = self.cap_video.read()
            if not flag:
                return
            
            # 使用增强处理
            processed_frame, name, confidence = self.process_video_frame_enhanced(image)
            
            # 更新UI显示
            self.update_ui_with_result(processed_frame, name, confidence)
            
        except Exception as e:
            print(f"视频显示失败: {e}")
    
    def update_ui_with_result(self, frame, name, confidence):
        """更新UI显示结果"""
        try:
            # 显示处理后的帧
            show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(showImage)
            self.label_display.setPixmap(pixmap)
            self.label_display.setScaledContents(True)
            
            # 更新识别结果标签
            self.label_plate_result.setText(name)
            self.label_score_dis.setText(f"{confidence:.3f}")
            
            # 如果识别成功，更新表格
            if confidence > 0.6:
                time_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self.change_table("实时识别", name, time_now, 1.0 - confidence)
            
            QtWidgets.QApplication.processEvents()
            
        except Exception as e:
            print(f"UI更新失败: {e}")
    
    # 保留原有的其他方法，但添加增强功能调用
    def cv_imread(self, file_path):
        """读取包含中文路径的图像"""
        cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
        return cv_img
    
    def save_enhanced_database(self):
        """保存增强数据库"""
        if self.use_enhanced_mode and self.enhanced_recognizer:
            try:
                filepath = "../data/enhanced_face_database.pkl"
                success = self.enhanced_recognizer.save_face_database(filepath)
                if success:
                    print("✓ 增强数据库已保存")
                    return True
            except Exception as e:
                print(f"保存增强数据库失败: {e}")
        return False
    
    def load_enhanced_database(self):
        """加载增强数据库"""
        if self.use_enhanced_mode and self.enhanced_recognizer:
            try:
                filepath = "../data/enhanced_face_database.pkl"
                success = self.enhanced_recognizer.load_face_database(filepath)
                if success:
                    print("✓ 增强数据库已加载")
                    return True
            except Exception as e:
                print(f"加载增强数据库失败: {e}")
        return False
    
    # 从原始代码复制的完整方法实现
    def choose_rec_img(self):
        """选择识别图像"""
        fileName_choose, filetype = QFileDialog.getOpenFileName(self.centralwidget,
                                                                "选取图片文件",
                                                                "../test_img/",
                                                                "图片(*.jpg;*.png;*.jpeg)")
        if fileName_choose != '':
            self.path = fileName_choose
            self.textEdit_file.setText(fileName_choose + '文件已选中')
            self.textEdit_file.setStyleSheet("background-color: transparent;\n"
                                           "border-color: rgb(0, 170, 255);\n"
                                           "color: rgb(0, 170, 255);\n"
                                           "font: regular 12pt \"华为仿宋\";")
            # 显示选择的图片
            img = self.cv_imread(fileName_choose)
            if img is not None:
                show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(showImage)
                self.label_display.setPixmap(pixmap)
                self.label_display.setScaledContents(True)
        else:
            self.textEdit_file.setText('图片文件未选中')
            self.textEdit_file.setStyleSheet("background-color: transparent;\n"
                                           "border-color: rgb(0, 170, 255);\n"
                                           "color: rgb(0, 170, 255);\n"
                                           "font: regular 12pt \"华为仿宋\";")
    
    def run_rec(self):
        """运行识别"""
        if self.path != '':
            self.do_choose_file()
        else:
            QtWidgets.QMessageBox.warning(self.centralwidget, "警告", "请先选择图片文件！")
    
    def do_choose_file(self):
        """选择图片识别时运行函数"""
        if self.path != '':
            self.label_display.clear()
            self.label_pic_newface.clear()
            QtWidgets.QApplication.processEvents()
            
            # 获取已存在人脸的特征
            exist_flag = self.get_face_database()
            img_rd = self.cv_imread(self.path)
            
            if img_rd is None:
                QtWidgets.QMessageBox.warning(self.centralwidget, "错误", "无法读取图片文件！")
                return
            
            # 使用增强识别或标准识别
            if self.use_enhanced_mode and self.enhanced_recognizer:
                try:
                    result = self.enhanced_recognizer.recognize_face(img_rd)
                    if result:
                        name, confidence = result
                        self.label_plate_result.setText(name)
                        self.label_score_dis.setText(f"{confidence:.3f}")
                        
                        # 显示处理后的图片
                        processed_img = self.enhanced_recognizer.preprocessor.enhance_low_light(img_rd)
                        show = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
                        pixmap = QtGui.QPixmap.fromImage(showImage)
                        self.label_display.setPixmap(pixmap)
                        self.label_display.setScaledContents(True)
                        
                        # 记录识别结果
                        time_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        self.change_table("图片识别", name, time_now, 1.0 - confidence)
                        return
                except Exception as e:
                    print(f"增强识别失败，使用标准识别: {e}")
            
            # 标准识别流程
            image = img_rd.copy()
            faces = self.detector(image, 0)
            
            if len(faces) > 0:
                self.label_score_num.setText(str(len(faces)))
                
                for k, d in enumerate(faces):
                    # 提取人脸特征
                    shape = self.predictor(img_rd, faces[k])
                    face_feature = self.face_reco_model.compute_face_descriptor(img_rd, shape)
                    
                    # 与数据库中的人脸进行比较
                    min_dis = 999999999
                    similar_person_num = -1
                    
                    for i in range(len(self.face_feature_exist)):
                        if str(self.face_feature_exist[i][0]) != '0.0':
                            e_distance_tmp = self.euclidean_distance(face_feature, self.face_feature_exist[i])
                            if e_distance_tmp < min_dis:
                                min_dis = e_distance_tmp
                                similar_person_num = i
                    
                    # 显示识别结果
                    if min_dis < 0.4 and similar_person_num >= 0:
                        name = self.face_name_exist[similar_person_num]
                    else:
                        name = "未知人脸"
                    
                    self.label_plate_result.setText(name)
                    self.label_score_dis.setText(str(round(min_dis, 3)))
                    
                    # 在图片上绘制人脸框
                    rect = (d.left(), d.top(), d.right(), d.bottom())
                    image = self.drawRectBox(image, rect, name)
                    
                    # 记录识别结果
                    time_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    self.change_table("图片识别", name, time_now, min_dis)
                
                # 显示处理后的图片
                show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(showImage)
                self.label_display.setPixmap(pixmap)
                self.label_display.setScaledContents(True)
            else:
                self.label_plate_result.setText("未检测到人脸")
                self.label_score_dis.setText("0")
    
    def button_open_camera_click(self):
        """打开摄像头"""
        if self.timer_camera_load.isActive():
            self.timer_camera_load.stop()
        if self.timer_video.isActive():
            self.timer_video.stop()
        if self.cap:
            self.cap.release()
        if self.cap_video:
            self.cap_video.release()
        
        self.label_display.clear()
        self.gif_movie()
        self.label_pic_newface.clear()
        self.ini_value()
        
        if not self.timer_camera.isActive():
            flag = self.cap.open(self.CAM_NUM)
            if not flag:
                QtWidgets.QMessageBox.warning(self.centralwidget, "警告",
                                              "请检测相机与电脑是否连接正确！",
                                              buttons=QtWidgets.QMessageBox.Ok,
                                              defaultButton=QtWidgets.QMessageBox.Ok)
                self.flag_timer = ""
            else:
                self.textEdit_camera.setText("相机准备就绪")
                self.textEdit_camera.setStyleSheet("background-color: transparent;\n"
                                                   "border-color: rgb(0, 170, 255);\n"
                                                   "color: rgb(0, 170, 255);\n"
                                                   "font: regular 12pt \"华为仿宋\";")
                self.flag_timer = "camera"
                self.get_face_database()
                self.timer_camera.start(30)
        else:
            self.flag_timer = ""
            self.timer_camera.stop()
            if self.cap:
                self.cap.release()
            self.label_display.clear()
            self.gif_movie()
            self.textEdit_camera.setText('实时摄像已关闭')
            self.textEdit_camera.setStyleSheet("background-color: transparent;\n"
                                               "border-color: rgb(0, 170, 255);\n"
                                               "color: rgb(0, 170, 255);\n"
                                               "font: regular 12pt \"华为仿宋\";")
    
    def button_open_video_click(self):
        """打开视频"""
        if self.timer_camera.isActive():
            self.timer_camera.stop()
        if self.cap:
            self.cap.release()
        if self.cap_video:
            self.cap_video.release()
        
        self.label_display.clear()
        self.gif_movie()
        self.label_pic_newface.clear()
        self.ini_value()
        
        if not self.timer_video.isActive():
            fileName_choose, filetype = QFileDialog.getOpenFileName(self.centralwidget, "选取视频文件",
                                                                    "../test_img/",
                                                                    "视频(*.mp4;*.avi)")
            if fileName_choose != '':
                self.flag_timer = "video"
                self.textEdit_video.setText(fileName_choose + '文件已选中')
                self.textEdit_video.setStyleSheet("background-color: transparent;\n"
                                                  "border-color: rgb(0, 170, 255);\n"
                                                  "color: rgb(0, 170, 255);\n"
                                                  "font: regular 12pt \"华为仿宋\";")
                try:
                    self.cap_video = cv2.VideoCapture(fileName_choose)
                    self.get_face_database()
                    self.timer_video.start(30)
                except Exception as e:
                    print(f"视频打开失败: {e}")
            else:
                self.textEdit_video.setText('视频文件未选中')
        else:
            self.flag_timer = ""
            self.timer_video.stop()
            if self.cap_video:
                self.cap_video.release()
            self.label_display.clear()
            self.gif_movie()
            self.textEdit_video.setText('实时视频已关闭')
    
    # 添加其他必要的辅助方法
    def ini_value(self):
        """初始化数值"""
        self.label_plate_result.setText("未知人脸")
        self.label_score_fps.setText("0")
        self.label_score_num.setText("0")
        self.label_score_dis.setText("0")
    
    def gif_movie(self):
        """显示GIF动画"""
        try:
            self.movie = QtGui.QMovie(":/newPrefix/images_test/face_rec.gif")
            self.label_display.setMovie(self.movie)
            self.movie.start()
        except:
            pass
    
    def drawRectBox(self, image, rect, addText):
        """绘制人脸框"""
        cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
        cv2.putText(image, addText, (rect[0], rect[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return image
    
    def euclidean_distance(self, feature_1, feature_2):
        """计算欧式距离"""
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist
     
    # 添加界面切换和人脸录入相关方法
    def change_size_load(self):
         """切换到录入界面"""
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
             self.cap_video.release()
 
         QtWidgets.QApplication.processEvents()
         self.label_display.clear()
         self.gif_movie()
         self.label_pic_newface.clear()
         self.label_pic_org.clear()
     
    def change_size_rec(self):
        """切换到识别界面"""
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
        """切换到管理界面"""
        self.toolButton_run_load.setGeometry(66, 236, 199, 49)
        self.toolButton_run_rec.setGeometry(66, 324, 199, 49)
        self.toolButton_run_manage.setGeometry(26, 410, 280, 70)
        self.tabWidget.setCurrentIndex(2)
        self.tabWidget.setTabVisible(0, False)
        self.tabWidget.setTabVisible(1, False)
        self.tabWidget.setTabVisible(2, True)
        self.update_face()
     
    def new_face_doing(self):
        """新建人脸文件夹"""
        try:
            self.toolButton_get_pic.setEnabled(False)
            self.toolButton_load_pic.setEnabled(False)
            self.label_display.clear()
            self.label_pic_newface.clear()
            self.current_face = None
            
            face_name = self.lineEdit_face_name.text()
            if face_name != "请在此输入人脸名" and face_name != "":
                face_dir_path = self.path_face_dir + face_name
                if not os.path.isdir(face_dir_path):
                    try:
                        os.makedirs(face_dir_path, exist_ok=True)
                        self.label_new_res.setText("输入的是新人名，已新建人脸文件夹！")
                        self.label_loadface.setText("新建完成，请点击左侧按钮选择图片或摄像录入！")
                    except Exception as e:
                        self.label_new_res.setText(f"创建文件夹失败：{str(e)}")
                        self.label_loadface.setText("文件夹创建失败，请检查权限或路径！")
                        return
                else:
                    self.label_new_res.setText("该人名已存在，继续录入将写入更多照片！")
                    self.label_loadface.setText("文件夹存在，请点击左侧按钮选择图片或摄像取图！")
                    
                self.toolButton_file_2.setEnabled(True)
                self.toolButton_camera_load.setEnabled(True)
            else:
                self.label_new_res.setText("请在左下角文本框中输入人脸名字！")
                self.label_loadface.setText("请先输入要录入的人脸名字！")
        except Exception as e:
            self.label_new_res.setText(f"处理人脸名称时发生异常：{str(e)}")
            print(f"new_face_doing异常: {str(e)}")
     
    def choose_file(self):
        """选择文件"""
        try:
            fileName_choose, filetype = QFileDialog.getOpenFileName(
                self.centralwidget, "选取图片文件",
                self.path, "图片(*.jpg;*.jpeg;*.png)")
            self.path = fileName_choose
            
            if self.path != '':
                try:
                    img_rd = self.cv_imread(self.path)
                    if img_rd is None:
                        self.label_loadface.setText("无法读取选择的图片文件！")
                        return
                    
                    image = img_rd.copy()
                    face_name = self.lineEdit_face_name.text()
                    faces = self.detector(image, 0)
                    
                    if len(faces) != 0:
                        for k, d in enumerate(faces):
                            try:
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
                                
                                if y2 > img_rd.shape[1]: y2 = img_rd.shape[1]
                                elif x2 > img_rd.shape[0]: x2 = img_rd.shape[0]
                                elif y1 < 0: y1 = 0
                                elif x1 < 0: x1 = 0

                                crop_face = img_rd[x1: x2, y1: y2]
                                if crop_face.size > 0:
                                    self.current_face = crop_face
                                    self.disp_face(crop_face)

                                    if self.current_face.any() and face_name != "请在此输入人脸名" and face_name != "":
                                        self.label_loadface.setText("检测到人脸区域，可点击取图按钮以保存！")
                                        self.toolButton_get_pic.setEnabled(True)
                                else:
                                    self.label_loadface.setText("人脸区域无效，请重新选择图片！")
                            except Exception as e:
                                print(f"处理人脸区域时发生异常: {str(e)}")
                                continue

                        self.disp_img(image)
                    else:
                        self.label_loadface.setText("未检测到人脸，请选择包含人脸的图片！")
                        self.disp_img(image)
                except Exception as e:
                    self.label_loadface.setText(f"处理图片时发生异常：{str(e)}")
            else:
                self.label_loadface.setText("未选择任何文件！")
        except Exception as e:
            self.label_loadface.setText(f"选择文件时发生异常：{str(e)}")
     
    def get_img_doing(self):
        """获取图像"""
        try:
            face_name = self.lineEdit_face_name.text()
            
            if face_name == "请在此输入人脸名" or face_name == "":
                self.label_loadface.setText("请先输入人脸名称！")
                return
            
            if self.current_face is None:
                self.label_loadface.setText("没有检测到人脸，请先选择图片或使用摄像头！")
                return
            
            if not self.current_face.any():
                self.label_loadface.setText("人脸数据无效，请重新选择图片！")
                return
            
            cur_path = self.path_face_dir + face_name + "/"
            
            if not os.path.exists(cur_path):
                try:
                    os.makedirs(cur_path, exist_ok=True)
                    self.label_new_res.setText("已自动创建人脸文件夹！")
                except Exception as e:
                    self.label_loadface.setText(f"创建目录失败：{str(e)}")
                    return
            
            try:
                files_path = glob.glob(pathname=cur_path + face_name + '_*.jpg')
                img_num = len(files_path) + 1
                
                success, encoded_img = cv2.imencode(".jpg", self.current_face)
                if not success:
                    self.label_loadface.setText("图片编码失败！")
                    return
                
                file_path = cur_path + face_name + "_" + str(img_num) + ".jpg"
                encoded_img.tofile(file_path)
                self.label_loadface.setText("图片" + face_name + "_" + str(img_num) + ".jpg" + "已保存，请继续添加或录入！")
                
                files_path = glob.glob(pathname=cur_path + face_name + '_*.jpg')
                self.disp_load_face(self.current_face)
                self.current_face = None
                self.toolButton_get_pic.setEnabled(False)

                if len(files_path) > 0:
                    self.toolButton_load_pic.setEnabled(True)
            except Exception as e:
                self.label_loadface.setText(f"保存图片失败：{str(e)}")
                return
        except Exception as e:
            self.label_loadface.setText(f"保存人脸时发生异常：{str(e)}")
     
    def load_img_doing(self):
        """加载图像"""
        try:
            if self.timer_camera_load.isActive():
                self.timer_camera_load.stop()
            if self.cap:
                self.cap.release()

            if not os.path.exists(self.path_face_dir):
                print(f"人脸数据目录不存在: {self.path_face_dir}")
                self.label_loadface.setText("人脸数据目录不存在！")
                return
            
            person_list = os.listdir(self.path_face_dir)

            with open("../data/features_all.csv", "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                for person in person_list:
                    features_list = []
                    person_path = self.path_face_dir + "/" + person
                    if not os.path.isdir(person_path):
                        continue
                    
                    photos_list = os.listdir(person_path)
                    if photos_list:
                        for photo in photos_list:
                            photo_path = person_path + "/" + photo
                            if not photo.lower().endswith(('.jpg', '.jpeg', '.png')):
                                continue
                            
                            features_128D = self.extract_features(photo_path)
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
        except Exception as e:
            print(f"加载人脸数据异常: {str(e)}")
            self.label_loadface.setText(f"加载失败: {str(e)}")
     
    def extract_features(self, path_img):
        """提取人脸特征"""
        try:
            img_rd = self.cv_imread(path_img)
            if img_rd is None:
                print(f"无法读取图像: {path_img}")
                return 0
            
            faces = self.detector(img_rd, 1)
            if len(faces) != 0:
                shape = self.predictor(img_rd, faces[0])
                face_descriptor = self.face_reco_model.compute_face_descriptor(img_rd, shape)
            else:
                face_descriptor = 0
            return face_descriptor
        except Exception as e:
            print(f"特征提取异常 ({path_img}): {str(e)}")
            return 0
     
    def disp_img(self, image):
        """显示图像"""
        try:
            if image is not None:
                image = cv2.resize(image, (500, 500))
                show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
                a = QtGui.QPixmap.fromImage(showImage)
                self.label_display.setPixmap(a)
                self.label_display.setScaledContents(True)
                QtWidgets.QApplication.processEvents()
        except Exception as e:
            print(f"显示图像异常: {str(e)}")
     
    def disp_face(self, image):
        """显示人脸"""
        self.label_pic_newface.clear()
        try:
            if image is not None and image.any():
                image = cv2.resize(image, (200, 200))
                show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
                a = QtGui.QPixmap.fromImage(showImage)
                self.label_pic_newface.setPixmap(a)
                self.label_pic_newface.setScaledContents(True)
                QtWidgets.QApplication.processEvents()
        except Exception as e:
            print(f"显示人脸图像异常: {str(e)}")
     
    def disp_load_face(self, image):
        """显示加载的人脸"""
        try:
            if image is not None:
                image = cv2.resize(image, (500, 500))
                show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
                a = QtGui.QPixmap.fromImage(showImage)
                self.label_pic_org.setPixmap(a)
                self.label_pic_org.setScaledContents(True)
                QtWidgets.QApplication.processEvents()
        except Exception as e:
            print(f"显示加载图像异常: {str(e)}")
     
    def change_table(self, path, res, time_now, distance):
        """更新表格记录"""
        self.count += 1
        if self.count > 6:
            self.tableWidget_rec.setRowCount(self.count)
        
        # 添加表格项
        items = [str(self.count), path, res, time_now, str(round(distance, 4))]
        for i, item_text in enumerate(items):
            newItem = QTableWidgetItem(item_text)
            newItem.setTextAlignment(Qt.AlignCenter)
            self.tableWidget_rec.setItem(self.count - 1, i, newItem)
            self.tableWidget_rec.setCurrentItem(newItem)
    
    def get_face_database(self):
        """获取人脸数据库"""
        try:
            if os.path.exists("../data/features_all.csv"):
                with open("../data/features_all.csv", "r", encoding="utf-8") as csvfile:
                    reader = csv.reader(csvfile)
                    self.face_feature_exist = []
                    self.face_name_exist = []
                    for row in reader:
                        if row:
                            self.face_name_exist.append(row[0])
                            features = [float(x) for x in row[1:]]
                            self.face_feature_exist.append(features)
                return True
            return False
        except Exception as e:
            print(f"获取人脸数据库异常: {str(e)}")
            return False
    
    def update_face(self):
        """更新人脸数据"""
        try:
            if os.path.exists("../data/features_all.csv"):
                with open("../data/features_all.csv", "r", encoding="utf-8") as csvfile:
                    reader = csv.reader(csvfile)
                    self.tableWidget_mana.setRowCount(0)
                    for i, row in enumerate(reader):
                        if row:
                            self.tableWidget_mana.insertRow(i)
                            self.tableWidget_mana.setItem(i, 0, QTableWidgetItem(str(i + 1)))
                            self.tableWidget_mana.setItem(i, 1, QTableWidgetItem(row[0]))
                            self.tableWidget_mana.setItem(i, 2, QTableWidgetItem("已录入"))
                            self.tableWidget_mana.setItem(i, 3, QTableWidgetItem("正常"))
        except Exception as e:
            print(f"更新人脸数据异常: {str(e)}")
    
    def do_update_face(self):
        """刷新人脸数据显示"""
        self.update_face()
    
    def button_open_camera_load(self):
        """打开摄像头录入"""
        try:
            if not self.timer_camera_load.isActive():
                flag = self.cap.open(self.CAM_NUM)
                if not flag:
                    QtWidgets.QMessageBox.warning(self.centralwidget, "警告", "请检测相机与电脑是否连接正确！")
                else:
                    self.timer_camera_load.start(30)
                    self.label_loadface.setText("摄像头已打开，检测到人脸后可点击取图按钮")
            else:
                self.timer_camera_load.stop()
                self.cap.release()
                self.label_display.clear()
                self.gif_movie()
                self.label_loadface.setText("摄像头已关闭")
        except Exception as e:
            print(f"摄像头录入异常: {str(e)}")
    
    def show_camera_load(self):
        """显示摄像头录入画面"""
        try:
            flag, image = self.cap.read()
            if flag:
                show_image = cv2.resize(image, (500, 500))
                
                # 检测人脸
                faces = self.detector(image, 0)
                if len(faces) != 0:
                    for k, d in enumerate(faces):
                        height = (d.bottom() - d.top())
                        width = (d.right() - d.left())
                        hh = int(height / 2)
                        ww = int(width / 2)
                        
                        y2 = d.right() + ww
                        x2 = d.bottom() + hh
                        y1 = d.left() - ww
                        x1 = d.top() - hh
                        
                        if y2 > image.shape[1]: y2 = image.shape[1]
                        elif x2 > image.shape[0]: x2 = image.shape[0]
                        elif y1 < 0: y1 = 0
                        elif x1 < 0: x1 = 0
                        
                        crop_face = image[x1: x2, y1: y2]
                        if crop_face.size > 0:
                            self.current_face = crop_face
                            self.disp_face(crop_face)
                            
                            face_name = self.lineEdit_face_name.text()
                            if face_name != "请在此输入人脸名" and face_name != "":
                                self.toolButton_get_pic.setEnabled(True)
                                self.label_loadface.setText("检测到人脸，可点击取图按钮保存")
                        
                        # 在显示图像上绘制人脸框
                        cv2.rectangle(show_image, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 2)
                
                show = cv2.cvtColor(show_image, cv2.COLOR_BGR2RGB)
                showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(showImage)
                self.label_display.setPixmap(pixmap)
                self.label_display.setScaledContents(True)
        except Exception as e:
            print(f"摄像头显示异常: {str(e)}")
    
    def table_review(self, row, col):
        """表格点击事件"""
        try:
            if row >= 0:
                name = self.tableWidget_mana.item(row, 1).text()
                face_dir = self.path_face_dir + name
                if os.path.exists(face_dir):
                    files = glob.glob(face_dir + "/*.jpg")
                    if files:
                        img = self.cv_imread(files[0])
                        if img is not None:
                            self.disp_img(img)
        except Exception as e:
            print(f"表格点击事件异常: {str(e)}")
    
    def delete_doing(self):
        """删除人脸数据"""
        try:
            current_row = self.tableWidget_mana.currentRow()
            if current_row >= 0:
                name = self.tableWidget_mana.item(current_row, 1).text()
                reply = QtWidgets.QMessageBox.question(self.centralwidget, '确认删除', 
                                                      f'确定要删除 {name} 的人脸数据吗？',
                                                      QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                                      QtWidgets.QMessageBox.No)
                if reply == QtWidgets.QMessageBox.Yes:
                    face_dir = self.path_face_dir + name
                    if os.path.exists(face_dir):
                        shutil.rmtree(face_dir)
                    self.update_face()
                    QtWidgets.QMessageBox.information(self.centralwidget, '删除成功', f'{name} 的人脸数据已删除')
        except Exception as e:
            print(f"删除人脸数据异常: {str(e)}")
            QtWidgets.QMessageBox.warning(self.centralwidget, '删除失败', f'删除失败: {str(e)}')
    
    def show_camera(self):
        """标准摄像头处理函数"""
        try:
            start_time = time.time()
            flag, img_rd = self.cap.read()  # 获取画面
            if not flag or img_rd is None:
                print("摄像头读取失败")
                return
        except Exception as e:
            print(f"摄像头读取异常: {str(e)}")
            return
        
        try:
            image = img_rd.copy()
            # 检测人脸
            faces = self.detector(img_rd, 0)
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

                        # 计算矩形框大小
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

                    # 2.2.2.1 遍历捕获到的图像中所有的人脸
                    for k in range(len(faces)):
                        self.current_centroid.append(
                            [int(faces[k].left() + faces[k].right()) / 2,
                             int(faces[k].top() + faces[k].bottom()) / 2])

                        self.current_face_distance = []

                        # 2.2.2.2 每个捕获人脸的名字坐标
                        self.current_face_position.append(tuple(
                            [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))

                        # 2.2.2.3 对于某张人脸，遍历所有存储的人脸特征
                        for i in range(len(self.face_feature_exist)):
                            # 如果数据不为空
                            if str(self.face_feature_exist[i][0]) != '0.0':
                                e_distance_tmp = self.euclidean_distance(
                                    self.current_face_feature[k],
                                    self.face_feature_exist[i])
                                self.current_face_distance.append(e_distance_tmp)
                            else:
                                # 空数据 person_X
                                self.current_face_distance.append(999999999)

                        # 2.2.2.4 寻找出最小的欧式距离匹配
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
        except Exception as e:
            print(f"人脸识别处理异常: {str(e)}")
            self.label_plate_result.setText("处理异常")
            self.label_score_fps.setText("0")
    
    def show_video(self):
        """标准视频处理函数"""
        try:
            start_time = time.time()
            flag, img_rd = self.cap_video.read()  # 获取画面
            if not flag or img_rd is None:
                print("视频读取失败")
                return
        except Exception as e:
            print(f"视频读取异常: {str(e)}")
            return
        
        try:
            image = img_rd.copy()
            # 检测人脸
            faces = self.detector(img_rd, 0)
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

                    # 2.2.2.1 遍历捕获到的图像中所有的人脸
                    for k in range(len(faces)):
                        self.current_centroid.append(
                            [int(faces[k].left() + faces[k].right()) / 2,
                             int(faces[k].top() + faces[k].bottom()) / 2])

                        self.current_face_distance = []

                        # 2.2.2.2 每个捕获人脸的名字坐标
                        self.current_face_position.append(tuple(
                            [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))

                        # 2.2.2.3 对于某张人脸，遍历所有存储的人脸特征
                        for i in range(len(self.face_feature_exist)):
                            # 如果数据不为空
                            if str(self.face_feature_exist[i][0]) != '0.0':
                                e_distance_tmp = self.euclidean_distance(
                                    self.current_face_feature[k],
                                    self.face_feature_exist[i])
                                self.current_face_distance.append(e_distance_tmp)
                            else:
                                # 空数据 person_X
                                self.current_face_distance.append(999999999)

                        # 2.2.2.4 寻找出最小的欧式距离匹配
                        min_dis = min(self.current_face_distance)
                        similar_person_num = self.current_face_distance.index(min_dis)

                        if min_dis < 0.4:
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
        except Exception as e:
            print(f"视频处理异常: {str(e)}")
            self.label_plate_result.setText("处理异常")
            self.label_score_fps.setText("0")
    
    def centroid_tracker(self):
        """质心跟踪器"""
        for i in range(len(self.current_centroid)):
            distance_current_person = []
            # 计算不同对象间的距离
            for j in range(len(self.last_centroid)):
                self.last_current_distance = self.euclidean_distance(
                    self.current_centroid[i], self.last_centroid[j])
                distance_current_person.append(self.last_current_distance)

            last_frame_num = distance_current_person.index(
                min(distance_current_person))
            self.current_face_name[i] = self.last_face_name[last_frame_num]

if __name__ == "__main__":
    print("\n=== 增强版人脸识别系统 ===")
    print("主要改进:")
    print("✓ ArcFace损失函数 - 提升识别准确性")
    print("✓ 注意力机制(CBAM) - 处理遮挡情况")
    print("✓ 低光照增强 - CLAHE算法")
    print("✓ 多尺度检测 - 提升检测率")
    print("✓ 改进预处理 - 标准化流程")
    print("✓ 异常处理 - 提升系统稳定性")
    print("\n适用场景:")
    print("• 戴口罩人脸识别")
    print("• 戴帽子人脸识别")
    print("• 暗光环境识别")
    print("• 部分遮挡识别")
    print("\n正在启动GUI界面...")
    
    # 启动PyQt5应用程序
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Face_MainWindow(MainWindow)
    MainWindow.show()
    
    print("✓ 增强版人脸识别系统GUI已启动")
    print("\n使用说明:")
    print("1. 点击'录入'按钮添加新的人脸")
    print("2. 点击'识别'按钮进行人脸识别")
    print("3. 点击'管理'按钮管理已有人脸数据")
    print("4. 系统会自动使用增强算法提升识别准确性")
    
    sys.exit(app.exec_())