#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时人脸检测程序
基于dlib和OpenCV实现实时摄像头人脸检测和识别
"""

import cv2
import dlib
import numpy as np
import pandas as pd
import os
import time
from PIL import Image, ImageDraw, ImageFont


class RealTimeFaceDetection:
    def __init__(self):
        # 初始化人脸检测器和识别模型
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('./data/data_dlib/shape_predictor_68_face_landmarks.dat')
        self.face_reco_model = dlib.face_recognition_model_v1(
            "./data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

        # 摄像头设置
        self.cap = None
        self.init_camera()

        # 人脸数据库
        self.face_feature_exist = []
        self.face_name_exist = []

        # 加载已有人脸数据
        self.load_face_database()

        # 中文字体设置（用于显示中文名字）
        try:
            self.font = ImageFont.truetype("./FaceRecUI/Font/platech.ttf", 20, 0)
        except:
            self.font = None
            print("警告：无法加载中文字体，将使用默认字体")

    def init_camera(self):
        """初始化摄像头，检测是否已经开启"""
        # 尝试创建摄像头对象
        test_cap = cv2.VideoCapture(0)

        if test_cap.isOpened():
            # 检查是否能读取帧
            ret, frame = test_cap.read()
            if ret:
                print("检测到摄像头已开启，直接使用")
                self.cap = test_cap
            else:
                print("摄像头已占用或无法读取，尝试重新初始化")
                test_cap.release()
                time.sleep(1)  # 等待一秒后重试
                self.cap = cv2.VideoCapture(0)
        else:
            print("摄像头未开启，正在打开摄像头")
            test_cap.release()
            self.cap = cv2.VideoCapture(0)

        # 设置摄像头参数
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            print("摄像头初始化成功")
        else:
            print("警告：摄像头初始化失败")

    def load_face_database(self):
        """加载人脸数据库"""
        if os.path.exists("./data/features_all.csv"):
            try:
                csv_rd = pd.read_csv("./data/features_all.csv", header=None, encoding='gb2312')
                print(f"成功加载人脸数据库，共有 {csv_rd.shape[0]} 个人脸")

                for i in range(csv_rd.shape[0]):
                    features_someone_arr = []
                    for j in range(1, 129):
                        if pd.isna(csv_rd.iloc[i][j]) or csv_rd.iloc[i][j] == '':
                            features_someone_arr.append(0.0)
                        else:
                            features_someone_arr.append(float(csv_rd.iloc[i][j]))
                    self.face_feature_exist.append(features_someone_arr)

                    if pd.isna(csv_rd.iloc[i][0]) or csv_rd.iloc[i][0] == '':
                        self.face_name_exist.append("未知人脸")
                    else:
                        self.face_name_exist.append(str(csv_rd.iloc[i][0]))

                print(f"加载的人脸名单: {self.face_name_exist}")
            except Exception as e:
                print(f"加载人脸数据库失败: {e}")
                print("将使用空数据库")
        else:
            print("人脸数据库文件不存在，将只进行人脸检测")

    def euclidean_distance(self, feature_1, feature_2):
        """计算两个128D向量间的欧式距离"""
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    def draw_rectangle_with_text(self, image, rect, text):
        """在图像上绘制矩形框和文字"""
        x1, y1, x2, y2 = rect

        # 绘制矩形框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 绘制文字背景
        cv2.rectangle(image, (x1, y1 - 30), (x1 + len(text) * 15, y1), (0, 255, 0), -1)

        # 如果有中文字体，使用PIL绘制中文
        if self.font and any('\u4e00' <= char <= '\u9fff' for char in text):
            # 转换为PIL图像
            img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            draw.text((x1 + 2, y1 - 28), text, (255, 255, 255), font=self.font)
            # 转换回OpenCV格式
            image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        else:
            # 使用OpenCV绘制英文
            cv2.putText(image, text, (x1 + 2, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return image

    def recognize_face(self, image, face_rect):
        """识别人脸"""
        if len(self.face_feature_exist) == 0:
            return "未知人脸", 999

        try:
            # 获取人脸特征
            shape = self.predictor(image, face_rect)
            face_descriptor = self.face_reco_model.compute_face_descriptor(image, shape)

            # 与数据库中的人脸进行比较
            min_distance = float('inf')
            best_match_name = "未知人脸"

            for i, known_face_feature in enumerate(self.face_feature_exist):
                if known_face_feature[0] != 0.0:  # 确保特征有效
                    distance = self.euclidean_distance(face_descriptor, known_face_feature)
                    if distance < min_distance:
                        min_distance = distance
                        if distance < 0.4:  # 阈值
                            best_match_name = self.face_name_exist[i]

            return best_match_name, min_distance
        except Exception as e:
            print(f"人脸识别错误: {e}")
            return "识别错误", 999

    def run(self):
        """运行实时人脸检测"""
        print("开始实时人脸检测...")
        print("正在识别人脸，请稍候...")

        if not self.cap.isOpened():
            print("错误：无法打开摄像头")
            return None

        # 设置缓冲区大小以减少延迟
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        frame_count = 0
        max_frames = 300  # 最多处理300帧，避免无限等待
        recognized_name = None  # 存储识别到的姓名

        while frame_count < max_frames:
            ret, frame = self.cap.read()
            if not ret:
                print("错误：无法读取摄像头画面")
                break

            frame_count += 1

            # 水平翻转图像（镜像效果）
            frame = cv2.flip(frame, 1)

            # 缩小图像进行检测以提高速度
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            faces = self.detector(small_frame, 0)

            # 将坐标放大回原尺寸
            scaled_faces = []
            for face in faces:
                scaled_face = dlib.rectangle(
                    face.left() * 2, face.top() * 2,
                    face.right() * 2, face.bottom() * 2
                )
                scaled_faces.append(scaled_face)

            # 处理检测到的每个人脸
            for face in scaled_faces:
                name, distance = self.recognize_face(frame, face)

                # 如果识别到有效人脸，立即返回结果
                if name != "未知人脸" and name != "识别错误" and distance < 0.4:
                    recognized_name = name
                    print(f"识别成功: {name} (相似度: {distance:.2f})")
                    # 清理资源并返回
                    self.cap.release()
                    cv2.destroyAllWindows()
                    return recognized_name

            # 每30帧显示一次进度
            if frame_count % 30 == 0:
                print(f"正在识别... 已处理 {frame_count} 帧")

        print("未能识别到有效人脸")

        # 清理资源
        self.cap.release()
        cv2.destroyAllWindows()
        print("程序已退出")
        return recognized_name


def main():
    """主函数"""
    try:
        face_detector = RealTimeFaceDetection()
        recognized_name = face_detector.run()
        return recognized_name
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        return None
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    name = main()
    if name:
        print(f"识别到的人员: {name}")
    else:
        print("未识别到有效人员")