#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强型实时人脸检测程序
基于dlib和OpenCV实现实时摄像头人脸检测和识别
集成自监督学习和环境反馈机制，提高遮挡和低光照条件下的识别准确率
"""

import cv2
import dlib
import numpy as np
import pandas as pd
import os
import time
import json
import pickle
from datetime import datetime
from collections import deque, defaultdict
from PIL import Image, ImageDraw, ImageFont
# from sklearn.metrics.pairwise import cosine_similarity  # 移除sklearn依赖
from scipy import ndimage
import threading
import queue

def cosine_similarity(X, Y=None):
    """计算余弦相似度，替代sklearn实现"""
    if Y is None:
        Y = X
    
    # 确保输入是2D数组
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)
    
    # 计算余弦相似度
    dot_product = np.dot(X, Y.T)
    norm_X = np.linalg.norm(X, axis=1, keepdims=True)
    norm_Y = np.linalg.norm(Y, axis=1, keepdims=True)
    
    # 避免除零
    norm_X = np.where(norm_X == 0, 1e-8, norm_X)
    norm_Y = np.where(norm_Y == 0, 1e-8, norm_Y)
    
    similarity = dot_product / (norm_X * norm_Y.T)
    return similarity

class RealTimeFaceDetection:
    def __init__(self):
        # 初始化人脸检测器和识别模型
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("./data/data_dlib/shape_predictor_68_face_landmarks.dat")
        self.face_reco_model = dlib.face_recognition_model_v1("./data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")
        
        # 摄像头设置
        self.cap = None
        self.init_camera()
        
        # 人脸数据库
        self.face_feature_exist = []
        self.face_name_exist = []
        
        # 环境反馈机制
        self.environmental_feedback = {
            'lighting_history': deque(maxlen=50),  # 光照历史
            'recognition_confidence': deque(maxlen=30),  # 识别置信度历史
            'face_quality_scores': deque(maxlen=20),  # 人脸质量分数
            'adaptive_threshold': 0.6,  # 自适应阈值
            'current_lighting_level': 0.5  # 当前光照水平
        }
        
        # 自监督学习机制
        self.self_supervised = {
            'feature_buffer': defaultdict(list),  # 特征缓冲区
            'confidence_weights': defaultdict(float),  # 置信度权重
            'temporal_features': deque(maxlen=10),  # 时序特征
            'learning_rate': 0.1,  # 学习率
            'update_threshold': 0.8  # 更新阈值
        }
        
        # 多尺度检测参数
        self.detection_scales = [0.5, 0.75, 1.0, 1.25]
        self.pyramid_levels = 3
        
        # 加载已有人脸数据
        self.load_face_database()
        self.load_environmental_model()
        
        # 中文字体设置（用于显示中文名字）
        try:
            self.font = ImageFont.truetype("./FaceRecUI/Font/platech.ttf", 20, 0)
        except:
            self.font = None
            print("警告：无法加载中文字体，将使用默认字体")
    
    def init_camera(self):
        """初始化摄像头，检测是否已经开启"""
        # 尝试创建摄像头对象
        test_cap = cv2.VideoCapture(1)
        
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
    
    def load_environmental_model(self):
        """加载环境模型"""
        model_path = "./data/environmental_model.pkl"
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    env_data = pickle.load(f)
                    self.environmental_feedback.update(env_data.get('feedback', {}))
                    self.self_supervised.update(env_data.get('self_supervised', {}))
                print("环境模型加载成功")
            except Exception as e:
                print(f"环境模型加载失败: {e}")
    
    def save_environmental_model(self):
        """保存环境模型"""
        model_path = "./data/environmental_model.pkl"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        try:
            env_data = {
                'feedback': dict(self.environmental_feedback),
                'self_supervised': dict(self.self_supervised),
                'timestamp': datetime.now().isoformat()
            }
            with open(model_path, 'wb') as f:
                pickle.dump(env_data, f)
        except Exception as e:
            print(f"环境模型保存失败: {e}")
    
    def enhance_image_for_low_light(self, image):
        """低光照图像增强"""
        # CLAHE (对比度限制自适应直方图均衡化)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 自适应CLAHE参数
        lighting_level = self.assess_lighting_condition(image)
        clip_limit = max(2.0, 4.0 - lighting_level * 2)
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # 伽马校正
        gamma = 1.2 if lighting_level < 0.3 else 1.0
        if gamma != 1.0:
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            enhanced = cv2.LUT(enhanced, table)
        
        return enhanced
    
    def assess_lighting_condition(self, image):
        """评估光照条件"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray) / 255.0
        
        # 计算对比度
        contrast = np.std(gray) / 255.0
        
        # 综合光照评分
        lighting_score = (mean_brightness + contrast) / 2.0
        
        # 更新光照历史
        self.environmental_feedback['lighting_history'].append(lighting_score)
        self.environmental_feedback['current_lighting_level'] = lighting_score
        
        return lighting_score
    
    def calculate_face_quality(self, image, face_rect):
        """计算人脸质量分数"""
        x1, y1, x2, y2 = face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()
        face_roi = image[y1:y2, x1:x2]
        
        if face_roi.size == 0:
            return 0.0
        
        # 计算清晰度 (拉普拉斯方差)
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        
        # 计算人脸大小分数
        face_area = (x2 - x1) * (y2 - y1)
        size_score = min(1.0, face_area / (100 * 100))  # 归一化到100x100像素
        
        # 计算姿态分数（基于人脸宽高比）
        aspect_ratio = (x2 - x1) / max(1, (y2 - y1))
        pose_score = 1.0 - abs(aspect_ratio - 1.0)  # 接近1.0的比例更好
        
        # 综合质量分数
        quality_score = (sharpness / 1000.0 * 0.5 + size_score * 0.3 + pose_score * 0.2)
        quality_score = min(1.0, quality_score)
        
        self.environmental_feedback['face_quality_scores'].append(quality_score)
        
        return quality_score
    
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
    
    def multi_scale_face_detection(self, image):
        """多尺度人脸检测"""
        all_faces = []
        
        for scale in self.detection_scales:
            if scale != 1.0:
                scaled_image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
            else:
                scaled_image = image
            
            # 检测人脸
            faces = self.detector(scaled_image, 1)
            
            # 将坐标缩放回原图尺寸
            for face in faces:
                if scale != 1.0:
                    scaled_face = dlib.rectangle(
                        int(face.left() / scale), int(face.top() / scale),
                        int(face.right() / scale), int(face.bottom() / scale)
                    )
                else:
                    scaled_face = face
                
                # 计算人脸质量
                quality = self.calculate_face_quality(image, scaled_face)
                all_faces.append((scaled_face, quality, scale))
        
        # 按质量排序，返回最佳人脸
        if all_faces:
            all_faces.sort(key=lambda x: x[1], reverse=True)
            return [face[0] for face in all_faces[:3]]  # 返回前3个最佳人脸
        
        return []
    
    def adaptive_threshold_update(self, confidence, success):
        """自适应阈值更新"""
        current_threshold = self.environmental_feedback['adaptive_threshold']
        
        # 基于识别成功率调整阈值
        if success:
            # 成功识别，可以适当提高阈值（更严格）
            adjustment = 0.01 * (1.0 - confidence)
        else:
            # 识别失败，降低阈值（更宽松）
            adjustment = -0.02
        
        # 考虑环境因素
        lighting_factor = self.environmental_feedback['current_lighting_level']
        if lighting_factor < 0.3:  # 低光照条件
            adjustment -= 0.01
        
        # 更新阈值
        new_threshold = current_threshold + adjustment
        self.environmental_feedback['adaptive_threshold'] = np.clip(new_threshold, 0.3, 0.8)
    
    def self_supervised_feature_update(self, name, feature_vector, confidence):
        """自监督特征更新"""
        if confidence > self.self_supervised['update_threshold']:
            # 高置信度特征，加入缓冲区
            self.self_supervised['feature_buffer'][name].append(feature_vector)
            
            # 保持缓冲区大小
            if len(self.self_supervised['feature_buffer'][name]) > 10:
                self.self_supervised['feature_buffer'][name].pop(0)
            
            # 更新置信度权重
            current_weight = self.self_supervised['confidence_weights'][name]
            learning_rate = self.self_supervised['learning_rate']
            self.self_supervised['confidence_weights'][name] = (
                current_weight * (1 - learning_rate) + confidence * learning_rate
            )
    
    def enhanced_feature_matching(self, query_feature, name_index):
        """增强特征匹配"""
        name = self.face_name_exist[name_index]
        base_feature = self.face_feature_exist[name_index]
        
        # 基础欧氏距离
        base_distance = self.euclidean_distance(query_feature, base_feature)
        
        # 如果有自监督学习的特征，进行融合
        if name in self.self_supervised['feature_buffer']:
            buffer_features = self.self_supervised['feature_buffer'][name]
            if buffer_features:
                # 计算与缓冲区特征的相似度
                similarities = []
                for buf_feature in buffer_features:
                    sim = 1.0 / (1.0 + self.euclidean_distance(query_feature, buf_feature))
                    similarities.append(sim)
                
                # 加权平均
                avg_similarity = np.mean(similarities)
                confidence_weight = self.self_supervised['confidence_weights'][name]
                
                # 融合距离
                enhanced_distance = base_distance * (1 - confidence_weight * 0.3) + \
                                  (1.0 - avg_similarity) * (confidence_weight * 0.3)
                
                return enhanced_distance
        
        return base_distance
    
    def recognize_face(self, image, face_rect):
        """增强人脸识别"""
        if len(self.face_feature_exist) == 0:
            return "未知人脸", 999, 0.0
        
        try:
            # 图像预处理和增强
            enhanced_image = self.enhance_image_for_low_light(image)
            
            # 获取人脸特征
            shape = self.predictor(enhanced_image, face_rect)
            face_descriptor = self.face_reco_model.compute_face_descriptor(enhanced_image, shape)
            face_feature_vector = list(face_descriptor)
            
            # 计算人脸质量
            face_quality = self.calculate_face_quality(image, face_rect)
            
            # 与数据库中的人脸进行增强匹配
            min_distance = float('inf')
            best_match_name = "未知人脸"
            best_match_index = -1
            
            for i, known_face_feature in enumerate(self.face_feature_exist):
                if known_face_feature[0] != 0.0:  # 确保特征有效
                    distance = self.enhanced_feature_matching(face_feature_vector, i)
                    if distance < min_distance:
                        min_distance = distance
                        best_match_index = i
            
            # 自适应阈值判断
            adaptive_threshold = self.environmental_feedback['adaptive_threshold']
            
            # 根据人脸质量调整阈值
            quality_adjusted_threshold = adaptive_threshold * (1.0 + (1.0 - face_quality) * 0.2)
            
            if min_distance < quality_adjusted_threshold and best_match_index >= 0:
                best_match_name = self.face_name_exist[best_match_index]
                confidence = 1.0 / (1.0 + min_distance)
                
                # 更新自监督学习
                self.self_supervised_feature_update(best_match_name, face_feature_vector, confidence)
                
                # 更新环境反馈
                self.environmental_feedback['recognition_confidence'].append(confidence)
                self.adaptive_threshold_update(confidence, True)
                
                return best_match_name, min_distance, confidence
            else:
                # 识别失败，更新环境反馈
                self.adaptive_threshold_update(0.0, False)
                return "未知人脸", min_distance, 0.0
            
        except Exception as e:
            print(f"人脸识别错误: {e}")
            return "识别错误", 999, 0.0
    
    def run(self):
        """运行增强实时人脸检测"""
        print("开始增强实时人脸检测...")
        print("正在识别人脸，请稍候...")
        
        if not self.cap.isOpened():
            print("错误：无法打开摄像头")
            return None
        
        # 设置缓冲区大小以减少延迟
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        frame_count = 0
        max_frames = 300  # 最多处理300帧，避免无限等待
        recognized_name = None  # 存储识别到的姓名
        recognition_history = deque(maxlen=5)  # 识别历史，用于稳定性
        
        while frame_count < max_frames:
            ret, frame = self.cap.read()
            if not ret:
                print("错误：无法读取摄像头画面")
                break
            
            frame_count += 1
            
            # 水平翻转图像（镜像效果）
            frame = cv2.flip(frame, 1)
            
            # 评估当前环境条件
            lighting_level = self.assess_lighting_condition(frame)
            
            # 使用多尺度检测
            detected_faces = self.multi_scale_face_detection(frame)
            
            if not detected_faces:
                # 如果多尺度检测失败，使用传统方法作为备选
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                faces = self.detector(small_frame, 0)
                
                detected_faces = []
                for face in faces:
                    scaled_face = dlib.rectangle(
                        face.left() * 2, face.top() * 2,
                        face.right() * 2, face.bottom() * 2
                    )
                    detected_faces.append(scaled_face)
            
            # 处理检测到的每个人脸
            best_recognition = None
            best_confidence = 0.0
            
            for face in detected_faces:
                name, distance, confidence = self.recognize_face(frame, face)
                
                # 选择最佳识别结果
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_recognition = (name, distance, confidence)
            
            # 如果有有效识别结果
            if best_recognition and best_recognition[0] not in ["未知人脸", "识别错误"]:
                name, distance, confidence = best_recognition
                
                # 添加到历史记录
                recognition_history.append((name, confidence))
                
                # 检查识别稳定性
                if len(recognition_history) >= 3:
                    # 检查最近3次识别是否一致
                    recent_names = [r[0] for r in list(recognition_history)[-3:]]
                    recent_confidences = [r[1] for r in list(recognition_history)[-3:]]
                    
                    if len(set(recent_names)) == 1 and np.mean(recent_confidences) > 0.7:
                        recognized_name = name
                        avg_confidence = np.mean(recent_confidences)
                        print(f"稳定识别成功: {name} (平均置信度: {avg_confidence:.3f})")
                        print(f"环境条件 - 光照水平: {lighting_level:.2f}, 自适应阈值: {self.environmental_feedback['adaptive_threshold']:.3f}")
                        
                        # 保存环境模型
                        self.save_environmental_model()
                        
                        # 清理资源并返回
                        self.cap.release()
                        cv2.destroyAllWindows()
                        return recognized_name
            
            # 每30帧显示一次进度和环境状态
            if frame_count % 30 == 0:
                print(f"正在识别... 已处理 {frame_count} 帧")
                print(f"当前环境 - 光照: {lighting_level:.2f}, 阈值: {self.environmental_feedback['adaptive_threshold']:.3f}")
                if recognition_history:
                    recent_conf = np.mean([r[1] for r in recognition_history])
                    print(f"最近识别置信度: {recent_conf:.3f}")
        
        print("未能识别到有效人脸")
        print(f"最终环境状态 - 光照: {self.environmental_feedback['current_lighting_level']:.2f}")
        print(f"自适应阈值: {self.environmental_feedback['adaptive_threshold']:.3f}")
        
        # 保存环境模型（即使识别失败也保存学习到的环境信息）
        self.save_environmental_model()
        
        # 清理资源
        self.cap.release()
        cv2.destroyAllWindows()
        print("程序已退出")
        return recognized_name

def face_run():
    """主函数"""
    try:
        face_detector = RealTimeFaceDetection()
        recognized_name = face_detector.run()
        name, PERSON, PERSON_id = recognized_name.split("-")
        return name, PERSON, PERSON_id
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        return None
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    name, PERSON, PERSON_id = face_run()
    if name:
        print(f"识别到的人员: {name}")
    else:
        print("未识别到有效人员")