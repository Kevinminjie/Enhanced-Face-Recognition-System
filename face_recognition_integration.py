#!/usr/bin/env python
# encoding: utf-8
'''
人脸识别系统集成脚本
将改进的ArcFace和注意力机制集成到现有的FaceRecognition.py中
'''

import sys
import os
import cv2
import numpy as np
from improved_face_recognition import ImprovedFaceRecognizer, ImprovedFacePreprocessor

class EnhancedFaceRecognition:
    """增强版人脸识别系统"""
    
    def __init__(self):
        # 初始化改进的人脸识别器
        self.improved_recognizer = ImprovedFaceRecognizer()
        self.preprocessor = ImprovedFacePreprocessor()
        
        # 兼容原有系统的变量
        self.current_face_dir = None
        self.face_features = {}
        
        print("增强版人脸识别系统初始化完成")
    
    def enhanced_face_detection(self, image):
        """增强的人脸检测，支持复杂条件"""
        try:
            if image is None:
                return None
            
            # 使用改进的预处理器检测人脸
            aligned_face = self.preprocessor.detect_and_align_face(image)
            
            if aligned_face is not None:
                return aligned_face
            
            # 如果改进方法失败，回退到原始方法
            return self._fallback_detection(image)
            
        except Exception as e:
            print(f"增强人脸检测失败: {e}")
            return None
    
    def _fallback_detection(self, image):
        """回退的人脸检测方法"""
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_region = image[y:y+h, x:x+w]
                face_resized = cv2.resize(face_region, (112, 112))
                return face_resized
            
            return None
        except Exception as e:
            print(f"回退检测方法失败: {e}")
            return None
    
    def enhanced_feature_extraction(self, face_image):
        """增强的特征提取"""
        try:
            # 使用改进的特征提取
            features = self.improved_recognizer.extract_features(face_image)
            return features
        except Exception as e:
            print(f"特征提取失败: {e}")
            return None
    
    def add_face_with_enhancement(self, name, image_path_or_array):
        """使用增强算法添加人脸"""
        try:
            # 处理输入（可以是路径或图像数组）
            if isinstance(image_path_or_array, str):
                image = cv2.imread(image_path_or_array)
            else:
                image = image_path_or_array
            
            if image is None:
                print(f"无法读取图像")
                return False
            
            # 检测人脸
            face = self.enhanced_face_detection(image)
            if face is None:
                print(f"未检测到人脸")
                return False
            
            # 添加到改进的识别器
            success = self.improved_recognizer.add_face_to_database(name, face)
            
            if success:
                # 同时保存到兼容格式
                features = self.enhanced_feature_extraction(face)
                if features is not None:
                    if name not in self.face_features:
                        self.face_features[name] = []
                    self.face_features[name].append(features)
                
                print(f"成功添加 {name} 的人脸特征（增强版）")
                return True
            
            return False
            
        except Exception as e:
            print(f"添加人脸失败: {e}")
            return False
    
    def recognize_with_enhancement(self, image, threshold=0.6):
        """使用增强算法识别人脸"""
        try:
            # 使用改进的识别器
            name, confidence, aligned_face = self.improved_recognizer.process_image(image)
            
            return {
                'name': name,
                'confidence': confidence,
                'aligned_face': aligned_face,
                'success': confidence >= threshold
            }
            
        except Exception as e:
            print(f"人脸识别失败: {e}")
            return {
                'name': 'Error',
                'confidence': 0.0,
                'aligned_face': None,
                'success': False
            }
    
    def process_video_frame(self, frame):
        """处理视频帧"""
        try:
            if frame is None:
                return None, "No Frame", 0.0
            
            # 增强的人脸检测
            face = self.enhanced_face_detection(frame)
            if face is None:
                return frame, "No Face Detected", 0.0
            
            # 识别人脸
            result = self.recognize_with_enhancement(frame)
            
            # 在原图上绘制结果
            annotated_frame = self._draw_recognition_result(frame, result)
            
            return annotated_frame, result['name'], result['confidence']
            
        except Exception as e:
            print(f"视频帧处理失败: {e}")
            return frame, "Error", 0.0
    
    def _draw_recognition_result(self, image, result):
        """在图像上绘制识别结果"""
        try:
            annotated = image.copy()
            
            # 检测人脸位置用于绘制框
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                # 绘制人脸框
                color = (0, 255, 0) if result['success'] else (0, 0, 255)
                cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
                
                # 绘制识别结果文本
                text = f"{result['name']}: {result['confidence']:.2f}"
                cv2.putText(annotated, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            return annotated
            
        except Exception as e:
            print(f"绘制结果失败: {e}")
            return image
    
    def batch_process_images(self, image_folder, output_folder=None):
        """批量处理图像"""
        try:
            if not os.path.exists(image_folder):
                print(f"文件夹不存在: {image_folder}")
                return
            
            if output_folder and not os.path.exists(output_folder):
                os.makedirs(output_folder)
            
            results = []
            
            for filename in os.listdir(image_folder):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_path = os.path.join(image_folder, filename)
                    image = cv2.imread(image_path)
                    
                    if image is not None:
                        result = self.recognize_with_enhancement(image)
                        results.append({
                            'filename': filename,
                            'name': result['name'],
                            'confidence': result['confidence']
                        })
                        
                        print(f"{filename}: {result['name']} ({result['confidence']:.3f})")
                        
                        # 保存处理后的图像
                        if output_folder:
                            annotated = self._draw_recognition_result(image, result)
                            output_path = os.path.join(output_folder, f"processed_{filename}")
                            cv2.imwrite(output_path, annotated)
            
            return results
            
        except Exception as e:
            print(f"批量处理失败: {e}")
            return []
    
    def save_face_database(self, filepath):
        """保存人脸数据库"""
        try:
            import pickle
            
            database = {
                'improved_db': self.improved_recognizer.face_database,
                'compatible_db': self.face_features
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(database, f)
            
            print(f"人脸数据库已保存到: {filepath}")
            return True
            
        except Exception as e:
            print(f"保存数据库失败: {e}")
            return False
    
    def load_face_database(self, filepath):
        """加载人脸数据库"""
        try:
            import pickle
            
            if not os.path.exists(filepath):
                print(f"数据库文件不存在: {filepath}")
                return False
            
            with open(filepath, 'rb') as f:
                database = pickle.load(f)
            
            if 'improved_db' in database:
                self.improved_recognizer.face_database = database['improved_db']
            
            if 'compatible_db' in database:
                self.face_features = database['compatible_db']
            
            print(f"人脸数据库已从 {filepath} 加载")
            return True
            
        except Exception as e:
            print(f"加载数据库失败: {e}")
            return False

# 使用示例和测试
if __name__ == "__main__":
    # 创建增强版人脸识别系统
    enhanced_system = EnhancedFaceRecognition()
    
    print("\n=== 增强版人脸识别系统 ===")
    print("主要功能:")
    print("1. enhanced_face_detection() - 增强的人脸检测")
    print("2. add_face_with_enhancement() - 添加人脸到数据库")
    print("3. recognize_with_enhancement() - 识别人脸")
    print("4. process_video_frame() - 处理视频帧")
    print("5. batch_process_images() - 批量处理图像")
    print("6. save/load_face_database() - 保存/加载数据库")
    
    print("\n=== 技术改进 ===")
    print("✓ ArcFace损失函数 - 提升识别准确性")
    print("✓ 注意力机制(CBAM) - 处理遮挡情况")
    print("✓ 低光照增强 - CLAHE算法")
    print("✓ 多尺度检测 - 提升检测率")
    print("✓ 改进预处理 - 标准化流程")
    print("✓ 异常处理 - 提升系统稳定性")
    
    print("\n系统已准备就绪，可以开始使用！")