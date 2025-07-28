#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
算法对比实验快速运行脚本

简化版本的算法对比实验，专注于核心性能指标的快速测试
"""

import cv2
import dlib
import numpy as np
import time
import os
import pandas as pd
from datetime import datetime

class QuickComparison:
    """
    快速算法对比类
    """
    
    def __init__(self):
        # 初始化检测器
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
        self.face_rec_model = dlib.face_recognition_model_v1('data/data_dlib/dlib_face_recognition_resnet_model_v1.dat')
        
        # 加载人脸数据库
        self.face_database = self.load_face_database()
        
        print(f"已加载 {len(self.face_database)} 个人脸数据")
    
    def load_face_database(self):
        """加载人脸数据库"""
        database = {}
        
        if os.path.exists('data/features_all.csv'):
            data = pd.read_csv('data/features_all.csv', header=None)
            for i in range(len(data)):
                name = data.iloc[i, 0]
                features = np.array(data.iloc[i, 1:129], dtype=np.float64)
                
                if name not in database:
                    database[name] = []
                database[name].append(features)
        
        return database
    
    def cosine_similarity(self, feature1, feature2):
        """计算余弦相似度"""
        dot_product = np.dot(feature1, feature2)
        norm1 = np.linalg.norm(feature1)
        norm2 = np.linalg.norm(feature2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot_product / (norm1 * norm2)
    
    def traditional_recognition(self, image, threshold=0.6):
        """传统人脸识别算法"""
        start_time = time.time()
        
        # 单一尺度检测
        faces = self.detector(image)
        
        if len(faces) == 0:
            return None, 0, time.time() - start_time
        
        # 取第一个人脸
        face = faces[0]
        landmarks = self.predictor(image, face)
        face_feature = np.array(self.face_rec_model.compute_face_descriptor(image, landmarks))
        
        # 简单匹配
        best_match = None
        best_similarity = 0
        
        for person_name, db_features in self.face_database.items():
            max_similarity = max([self.cosine_similarity(face_feature, db_feat) 
                                for db_feat in db_features])
            
            if max_similarity > threshold and max_similarity > best_similarity:
                best_similarity = max_similarity
                best_match = person_name
        
        processing_time = time.time() - start_time
        return best_match, best_similarity, processing_time
    
    def enhanced_recognition(self, image):
        """增强人脸识别算法（简化版）"""
        start_time = time.time()
        
        # 多尺度检测
        all_faces = []
        scales = [0.8, 1.0, 1.2]
        
        for scale in scales:
            height, width = image.shape[:2]
            new_height, new_width = int(height * scale), int(width * scale)
            scaled_image = cv2.resize(image, (new_width, new_height))
            
            faces = self.detector(scaled_image)
            for face in faces:
                scaled_face = dlib.rectangle(
                    int(face.left() / scale),
                    int(face.top() / scale),
                    int(face.right() / scale),
                    int(face.bottom() / scale)
                )
                all_faces.append(scaled_face)
        
        if not all_faces:
            return None, 0, time.time() - start_time
        
        # 取最大的人脸
        face = max(all_faces, key=lambda f: (f.right() - f.left()) * (f.bottom() - f.top()))
        
        # 图像增强（简化版）
        enhanced_image = self.simple_enhance(image)
        
        # 特征提取
        landmarks = self.predictor(enhanced_image, face)
        face_feature = np.array(self.face_rec_model.compute_face_descriptor(enhanced_image, landmarks))
        
        # 自适应阈值匹配
        best_match = None
        best_similarity = 0
        adaptive_threshold = 0.55  # 稍微降低阈值
        
        for person_name, db_features in self.face_database.items():
            max_similarity = max([self.cosine_similarity(face_feature, db_feat) 
                                for db_feat in db_features])
            
            if max_similarity > adaptive_threshold and max_similarity > best_similarity:
                best_similarity = max_similarity
                best_match = person_name
        
        processing_time = time.time() - start_time
        return best_match, best_similarity, processing_time
    
    def simple_enhance(self, image):
        """简单图像增强"""
        # 转换到LAB色彩空间
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # CLAHE增强
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # 重新组合
        enhanced_lab = cv2.merge([l, a, b])
        enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return enhanced_image
    
    def simulate_conditions(self, image, condition):
        """模拟不同环境条件"""
        if condition == 'low_light':
            return cv2.convertScaleAbs(image, alpha=0.4, beta=-30)
        elif condition == 'high_light':
            return cv2.convertScaleAbs(image, alpha=1.3, beta=30)
        elif condition == 'blur':
            return cv2.GaussianBlur(image, (5, 5), 1.0)
        else:
            return image
    
    def run_quick_test(self):
        """运行快速测试"""
        print("\n开始快速算法对比测试...")
        print("=" * 60)
        
        # 使用摄像头进行实时测试
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("无法打开摄像头")
            return
        
        # 测试条件
        conditions = ['normal', 'low_light', 'high_light']
        current_condition = 0
        
        # 统计数据
        stats = {
            'traditional': {'times': [], 'confidences': [], 'successes': 0, 'total': 0},
            'enhanced': {'times': [], 'confidences': [], 'successes': 0, 'total': 0}
        }
        
        print("实时对比测试 - 按键说明:")
        print("'q': 退出")
        print("'c': 切换测试条件")
        print("'t': 执行测试")
        print("'r': 显示统计报告")
        
        frame_count = 0
        test_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 应用当前测试条件
            test_frame = self.simulate_conditions(frame, conditions[current_condition])
            
            # 显示当前条件
            display_frame = test_frame.copy()
            cv2.putText(display_frame, f"Condition: {conditions[current_condition]}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display_frame, f"Tests: {test_count}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display_frame, "Press 't' to test, 'c' to change condition", 
                       (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Algorithm Comparison Test', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                current_condition = (current_condition + 1) % len(conditions)
                print(f"切换到测试条件: {conditions[current_condition]}")
            elif key == ord('t'):
                # 执行对比测试
                test_count += 1
                print(f"\n执行测试 #{test_count} - 条件: {conditions[current_condition]}")
                
                # 传统算法测试
                trad_result, trad_conf, trad_time = self.traditional_recognition(test_frame)
                stats['traditional']['times'].append(trad_time)
                stats['traditional']['total'] += 1
                if trad_result:
                    stats['traditional']['confidences'].append(trad_conf)
                    stats['traditional']['successes'] += 1
                
                # 增强算法测试
                enh_result, enh_conf, enh_time = self.enhanced_recognition(test_frame)
                stats['enhanced']['times'].append(enh_time)
                stats['enhanced']['total'] += 1
                if enh_result:
                    stats['enhanced']['confidences'].append(enh_conf)
                    stats['enhanced']['successes'] += 1
                
                print(f"传统算法: {trad_result} (置信度: {trad_conf:.3f}, 时间: {trad_time:.3f}s)")
                print(f"增强算法: {enh_result} (置信度: {enh_conf:.3f}, 时间: {enh_time:.3f}s)")
                
            elif key == ord('r'):
                # 显示统计报告
                self.show_quick_report(stats)
        
        cap.release()
        cv2.destroyAllWindows()
        
        # 最终报告
        if test_count > 0:
            print("\n最终测试报告:")
            self.show_quick_report(stats)
            self.save_quick_report(stats)
    
    def show_quick_report(self, stats):
        """显示快速报告"""
        print("\n" + "=" * 50)
        print("快速测试统计报告")
        print("=" * 50)
        
        for algo_name, data in stats.items():
            if data['total'] > 0:
                success_rate = data['successes'] / data['total'] * 100
                avg_time = np.mean(data['times']) if data['times'] else 0
                avg_conf = np.mean(data['confidences']) if data['confidences'] else 0
                
                print(f"\n{algo_name.upper()} 算法:")
                print(f"  成功率: {success_rate:.1f}% ({data['successes']}/{data['total']})")
                print(f"  平均处理时间: {avg_time:.4f}s")
                print(f"  平均置信度: {avg_conf:.3f}")
        
        # 对比分析
        if stats['traditional']['total'] > 0 and stats['enhanced']['total'] > 0:
            trad_success = stats['traditional']['successes'] / stats['traditional']['total'] * 100
            enh_success = stats['enhanced']['successes'] / stats['enhanced']['total'] * 100
            
            trad_time = np.mean(stats['traditional']['times'])
            enh_time = np.mean(stats['enhanced']['times'])
            
            print(f"\n对比结果:")
            print(f"  成功率提升: {enh_success - trad_success:.1f}%")
            print(f"  速度变化: {((trad_time - enh_time) / trad_time * 100):.1f}%")
    
    def save_quick_report(self, stats):
        """保存快速报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"quick_comparison_report_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("快速算法对比测试报告\n")
            f.write("=" * 40 + "\n")
            f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for algo_name, data in stats.items():
                if data['total'] > 0:
                    success_rate = data['successes'] / data['total'] * 100
                    avg_time = np.mean(data['times']) if data['times'] else 0
                    avg_conf = np.mean(data['confidences']) if data['confidences'] else 0
                    
                    f.write(f"{algo_name.upper()} 算法:\n")
                    f.write(f"  成功率: {success_rate:.1f}%\n")
                    f.write(f"  平均处理时间: {avg_time:.4f}s\n")
                    f.write(f"  平均置信度: {avg_conf:.3f}\n\n")
        
        print(f"\n报告已保存到: {report_file}")

def main():
    """主函数"""
    print("人脸识别算法快速对比测试")
    print("=" * 40)
    
    try:
        comparison = QuickComparison()
        comparison.run_quick_test()
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        print("请确保:")
        print("1. 摄像头正常工作")
        print("2. dlib模型文件存在")
        print("3. 人脸数据库已加载")

if __name__ == "__main__":
    main()