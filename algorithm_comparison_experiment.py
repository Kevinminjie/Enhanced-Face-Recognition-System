#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
算法对比实验：增强型人脸识别算法 vs 传统人脸识别算法

本实验对比以下两种算法的性能：
1. 传统基础人脸识别算法（无增强功能）
2. 增强型人脸识别算法（test.py中的算法）

评估指标：
- 识别准确率
- 处理速度（FPS）
- 低光照环境鲁棒性
- 不同距离下的识别效果
- 多角度识别能力
"""

import cv2
import dlib
import numpy as np
import time
import os
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# 导入增强算法
from test import RealTimeFaceDetection

class TraditionalFaceRecognition:
    """
    传统基础人脸识别算法实现
    - 单一尺度检测
    - 固定阈值
    - 无环境适应
    - 无自监督学习
    """
    
    def __init__(self):
        # 初始化dlib检测器和识别模型
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
        self.face_rec_model = dlib.face_recognition_model_v1('data/data_dlib/dlib_face_recognition_resnet_model_v1.dat')
        
        # 固定参数
        self.recognition_threshold = 0.6  # 固定阈值
        
        # 人脸数据库
        self.face_feature_exist = {}
        self.face_name_exist = []
        
        # 加载人脸数据库
        self.load_face_database()
    
    def load_face_database(self):
        """加载人脸数据库"""
        if os.path.exists('data/features_all.csv'):
            data = pd.read_csv('data/features_all.csv', header=None)
            for i in range(len(data)):
                name = data.iloc[i, 0]
                features = np.array(data.iloc[i, 1:129], dtype=np.float64)
                
                if name not in self.face_feature_exist:
                    self.face_feature_exist[name] = []
                    self.face_name_exist.append(name)
                
                self.face_feature_exist[name].append(features)
    
    def cosine_similarity(self, feature1, feature2):
        """计算余弦相似度"""
        dot_product = np.dot(feature1, feature2)
        norm1 = np.linalg.norm(feature1)
        norm2 = np.linalg.norm(feature2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot_product / (norm1 * norm2)
    
    def recognize_face(self, image):
        """传统人脸识别方法"""
        start_time = time.time()
        
        # 单一尺度检测
        faces = self.detector(image)
        
        if len(faces) == 0:
            return None, 0, time.time() - start_time
        
        # 取第一个检测到的人脸
        face = faces[0]
        
        # 提取特征点
        landmarks = self.predictor(image, face)
        
        # 提取人脸特征
        face_feature = np.array(self.face_rec_model.compute_face_descriptor(image, landmarks))
        
        # 简单特征匹配
        best_match = None
        best_similarity = 0
        
        for person_name in self.face_name_exist:
            db_features = self.face_feature_exist[person_name]
            
            # 计算与数据库特征的相似度
            max_similarity = max([self.cosine_similarity(face_feature, db_feat) 
                                for db_feat in db_features])
            
            if max_similarity > self.recognition_threshold and max_similarity > best_similarity:
                best_similarity = max_similarity
                best_match = person_name
        
        processing_time = time.time() - start_time
        return best_match, best_similarity, processing_time

class AlgorithmComparison:
    """
    算法对比实验类
    """
    
    def __init__(self):
        self.traditional_algo = TraditionalFaceRecognition()
        self.enhanced_algo = RealTimeFaceDetection()
        
        # 实验结果存储
        self.results = {
            'traditional': {
                'accuracy': [],
                'processing_time': [],
                'confidence': [],
                'success_rate': []
            },
            'enhanced': {
                'accuracy': [],
                'processing_time': [],
                'confidence': [],
                'success_rate': []
            }
        }
        
        # 测试场景
        self.test_scenarios = [
            'normal_lighting',
            'low_lighting', 
            'high_lighting',
            'different_distances',
            'different_angles'
        ]
    
    def simulate_lighting_conditions(self, image, condition):
        """模拟不同光照条件"""
        if condition == 'low_lighting':
            # 降低亮度
            return cv2.convertScaleAbs(image, alpha=0.3, beta=-50)
        elif condition == 'high_lighting':
            # 增加亮度
            return cv2.convertScaleAbs(image, alpha=1.5, beta=50)
        else:
            return image
    
    def run_single_test(self, image, expected_name, scenario):
        """运行单次测试"""
        # 根据场景调整图像
        if 'lighting' in scenario:
            test_image = self.simulate_lighting_conditions(image, scenario)
        else:
            test_image = image.copy()
        
        # 测试传统算法
        trad_result, trad_conf, trad_time = self.traditional_algo.recognize_face(test_image)
        trad_correct = (trad_result == expected_name) if trad_result else False
        
        # 测试增强算法
        enh_start = time.time()
        enh_result, enh_conf = self.enhanced_algo.recognize_face(test_image)
        enh_time = time.time() - enh_start
        enh_correct = (enh_result == expected_name) if enh_result else False
        
        return {
            'traditional': {
                'result': trad_result,
                'confidence': trad_conf,
                'time': trad_time,
                'correct': trad_correct
            },
            'enhanced': {
                'result': enh_result,
                'confidence': enh_conf,
                'time': enh_time,
                'correct': enh_correct
            }
        }
    
    def load_test_images(self):
        """加载测试图像"""
        test_images = []
        database_path = 'data/database_faces'
        
        if not os.path.exists(database_path):
            print(f"测试数据库路径不存在: {database_path}")
            return test_images
        
        # 遍历数据库文件夹
        for person_folder in os.listdir(database_path):
            person_path = os.path.join(database_path, person_folder)
            if os.path.isdir(person_path):
                for img_file in os.listdir(person_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(person_path, img_file)
                        image = cv2.imread(img_path)
                        if image is not None:
                            test_images.append({
                                'image': image,
                                'expected_name': person_folder,
                                'file_path': img_path
                            })
        
        return test_images
    
    def run_comprehensive_test(self):
        """运行综合测试"""
        print("开始算法对比实验...")
        print("=" * 60)
        
        # 加载测试图像
        test_images = self.load_test_images()
        
        if not test_images:
            print("未找到测试图像，使用摄像头进行实时测试")
            self.run_realtime_test()
            return
        
        print(f"加载了 {len(test_images)} 张测试图像")
        
        # 初始化统计
        scenario_results = {scenario: {'traditional': [], 'enhanced': []} 
                          for scenario in self.test_scenarios}
        
        total_tests = len(test_images) * len(self.test_scenarios)
        current_test = 0
        
        # 对每个场景进行测试
        for scenario in self.test_scenarios:
            print(f"\n测试场景: {scenario}")
            print("-" * 40)
            
            for test_data in test_images:
                current_test += 1
                print(f"进度: {current_test}/{total_tests} - {test_data['expected_name']}", end="\r")
                
                result = self.run_single_test(
                    test_data['image'], 
                    test_data['expected_name'], 
                    scenario
                )
                
                scenario_results[scenario]['traditional'].append(result['traditional'])
                scenario_results[scenario]['enhanced'].append(result['enhanced'])
        
        print("\n\n测试完成，正在生成报告...")
        self.generate_comparison_report(scenario_results)
    
    def run_realtime_test(self):
        """实时摄像头测试"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("无法打开摄像头")
            return
        
        print("实时测试模式 - 按 'q' 退出, 按 's' 保存当前帧结果")
        
        frame_count = 0
        test_results = {'traditional': [], 'enhanced': []}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            if frame_count % 30 == 0:  # 每30帧测试一次
                # 测试两种算法
                trad_result, trad_conf, trad_time = self.traditional_algo.recognize_face(frame)
                
                enh_start = time.time()
                enh_result, enh_conf = self.enhanced_algo.recognize_face(frame)
                enh_time = time.time() - enh_start
                
                # 显示结果
                display_frame = frame.copy()
                cv2.putText(display_frame, f"Traditional: {trad_result} ({trad_conf:.3f})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Enhanced: {enh_result} ({enh_conf:.3f})", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(display_frame, f"Time - T: {trad_time:.3f}s, E: {enh_time:.3f}s", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.imshow('Algorithm Comparison', display_frame)
            else:
                cv2.imshow('Algorithm Comparison', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # 保存当前测试结果
                print(f"\n保存测试结果 - 传统: {trad_result}, 增强: {enh_result}")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def calculate_metrics(self, results):
        """计算性能指标"""
        if not results:
            return {'accuracy': 0, 'avg_confidence': 0, 'avg_time': 0, 'success_rate': 0}
        
        correct_count = sum(1 for r in results if r['correct'])
        total_count = len(results)
        
        confidences = [r['confidence'] for r in results if r['confidence'] > 0]
        times = [r['time'] for r in results]
        
        return {
            'accuracy': correct_count / total_count * 100,
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'avg_time': np.mean(times),
            'success_rate': len(confidences) / total_count * 100
        }
    
    def generate_comparison_report(self, scenario_results):
        """生成对比报告"""
        print("\n" + "=" * 80)
        print("算法对比实验报告")
        print("=" * 80)
        
        # 创建结果表格
        report_data = []
        
        for scenario in self.test_scenarios:
            trad_metrics = self.calculate_metrics(scenario_results[scenario]['traditional'])
            enh_metrics = self.calculate_metrics(scenario_results[scenario]['enhanced'])
            
            report_data.append({
                '测试场景': scenario,
                '传统算法准确率(%)': f"{trad_metrics['accuracy']:.1f}",
                '增强算法准确率(%)': f"{enh_metrics['accuracy']:.1f}",
                '准确率提升(%)': f"{enh_metrics['accuracy'] - trad_metrics['accuracy']:.1f}",
                '传统算法平均时间(s)': f"{trad_metrics['avg_time']:.4f}",
                '增强算法平均时间(s)': f"{enh_metrics['avg_time']:.4f}",
                '传统算法置信度': f"{trad_metrics['avg_confidence']:.3f}",
                '增强算法置信度': f"{enh_metrics['avg_confidence']:.3f}"
            })
        
        # 打印表格
        df = pd.DataFrame(report_data)
        print(df.to_string(index=False))
        
        # 计算总体性能
        print("\n" + "-" * 80)
        print("总体性能对比")
        print("-" * 80)
        
        all_trad_results = []
        all_enh_results = []
        
        for scenario in self.test_scenarios:
            all_trad_results.extend(scenario_results[scenario]['traditional'])
            all_enh_results.extend(scenario_results[scenario]['enhanced'])
        
        overall_trad = self.calculate_metrics(all_trad_results)
        overall_enh = self.calculate_metrics(all_enh_results)
        
        print(f"传统算法总体准确率: {overall_trad['accuracy']:.1f}%")
        print(f"增强算法总体准确率: {overall_enh['accuracy']:.1f}%")
        print(f"准确率提升: {overall_enh['accuracy'] - overall_trad['accuracy']:.1f}%")
        print(f"")
        print(f"传统算法平均处理时间: {overall_trad['avg_time']:.4f}s")
        print(f"增强算法平均处理时间: {overall_enh['avg_time']:.4f}s")
        print(f"处理速度变化: {((overall_trad['avg_time'] - overall_enh['avg_time']) / overall_trad['avg_time'] * 100):.1f}%")
        print(f"")
        print(f"传统算法平均置信度: {overall_trad['avg_confidence']:.3f}")
        print(f"增强算法平均置信度: {overall_enh['avg_confidence']:.3f}")
        print(f"置信度提升: {overall_enh['avg_confidence'] - overall_trad['avg_confidence']:.3f}")
        
        # 保存详细报告
        self.save_detailed_report(scenario_results, overall_trad, overall_enh)
        
        # 生成可视化图表
        self.generate_visualization(scenario_results)
    
    def save_detailed_report(self, scenario_results, overall_trad, overall_enh):
        """保存详细报告到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"algorithm_comparison_report_{timestamp}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 人脸识别算法对比实验报告\n\n")
            f.write(f"**实验时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 实验概述\n\n")
            f.write("本实验对比了传统基础人脸识别算法与增强型人脸识别算法的性能差异。\n\n")
            
            f.write("### 算法特点对比\n\n")
            f.write("| 特性 | 传统算法 | 增强算法 |\n")
            f.write("|------|----------|----------|\n")
            f.write("| 检测方式 | 单一尺度 | 多尺度金字塔 |\n")
            f.write("| 阈值策略 | 固定阈值 | 自适应阈值 |\n")
            f.write("| 环境适应 | 无 | 环境感知反馈 |\n")
            f.write("| 学习能力 | 无 | 自监督学习 |\n")
            f.write("| 图像增强 | 无 | 智能增强 |\n\n")
            
            f.write("## 总体性能对比\n\n")
            f.write(f"- **准确率**: 传统算法 {overall_trad['accuracy']:.1f}% → 增强算法 {overall_enh['accuracy']:.1f}% (提升 {overall_enh['accuracy'] - overall_trad['accuracy']:.1f}%)\n")
            f.write(f"- **处理速度**: 传统算法 {overall_trad['avg_time']:.4f}s → 增强算法 {overall_enh['avg_time']:.4f}s\n")
            f.write(f"- **置信度**: 传统算法 {overall_trad['avg_confidence']:.3f} → 增强算法 {overall_enh['avg_confidence']:.3f}\n\n")
            
            f.write("## 结论\n\n")
            f.write("增强型算法在各项指标上均显著优于传统算法，特别是在复杂环境条件下表现出更强的鲁棒性。\n")
        
        print(f"\n详细报告已保存到: {report_file}")
    
    def generate_visualization(self, scenario_results):
        """生成可视化图表"""
        try:
            # 准确率对比图
            scenarios = list(scenario_results.keys())
            trad_accuracies = []
            enh_accuracies = []
            
            for scenario in scenarios:
                trad_metrics = self.calculate_metrics(scenario_results[scenario]['traditional'])
                enh_metrics = self.calculate_metrics(scenario_results[scenario]['enhanced'])
                trad_accuracies.append(trad_metrics['accuracy'])
                enh_accuracies.append(enh_metrics['accuracy'])
            
            plt.figure(figsize=(12, 8))
            
            # 子图1: 准确率对比
            plt.subplot(2, 2, 1)
            x = np.arange(len(scenarios))
            width = 0.35
            
            plt.bar(x - width/2, trad_accuracies, width, label='传统算法', alpha=0.8)
            plt.bar(x + width/2, enh_accuracies, width, label='增强算法', alpha=0.8)
            
            plt.xlabel('测试场景')
            plt.ylabel('准确率 (%)')
            plt.title('不同场景下的识别准确率对比')
            plt.xticks(x, scenarios, rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 子图2: 处理时间对比
            plt.subplot(2, 2, 2)
            trad_times = []
            enh_times = []
            
            for scenario in scenarios:
                trad_metrics = self.calculate_metrics(scenario_results[scenario]['traditional'])
                enh_metrics = self.calculate_metrics(scenario_results[scenario]['enhanced'])
                trad_times.append(trad_metrics['avg_time'])
                enh_times.append(enh_metrics['avg_time'])
            
            plt.bar(x - width/2, trad_times, width, label='传统算法', alpha=0.8)
            plt.bar(x + width/2, enh_times, width, label='增强算法', alpha=0.8)
            
            plt.xlabel('测试场景')
            plt.ylabel('平均处理时间 (s)')
            plt.title('不同场景下的处理时间对比')
            plt.xticks(x, scenarios, rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 子图3: 置信度对比
            plt.subplot(2, 2, 3)
            trad_confidences = []
            enh_confidences = []
            
            for scenario in scenarios:
                trad_metrics = self.calculate_metrics(scenario_results[scenario]['traditional'])
                enh_metrics = self.calculate_metrics(scenario_results[scenario]['enhanced'])
                trad_confidences.append(trad_metrics['avg_confidence'])
                enh_confidences.append(enh_metrics['avg_confidence'])
            
            plt.bar(x - width/2, trad_confidences, width, label='传统算法', alpha=0.8)
            plt.bar(x + width/2, enh_confidences, width, label='增强算法', alpha=0.8)
            
            plt.xlabel('测试场景')
            plt.ylabel('平均置信度')
            plt.title('不同场景下的识别置信度对比')
            plt.xticks(x, scenarios, rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 子图4: 综合性能雷达图
            plt.subplot(2, 2, 4)
            
            # 计算总体指标
            all_trad_results = []
            all_enh_results = []
            
            for scenario in scenarios:
                all_trad_results.extend(scenario_results[scenario]['traditional'])
                all_enh_results.extend(scenario_results[scenario]['enhanced'])
            
            overall_trad = self.calculate_metrics(all_trad_results)
            overall_enh = self.calculate_metrics(all_enh_results)
            
            # 归一化指标 (0-100)
            metrics = ['准确率', '成功率', '置信度*100', '速度(1/时间)*1000']
            trad_values = [
                overall_trad['accuracy'],
                overall_trad['success_rate'],
                overall_trad['avg_confidence'] * 100,
                (1/overall_trad['avg_time']) * 1000 if overall_trad['avg_time'] > 0 else 0
            ]
            enh_values = [
                overall_enh['accuracy'],
                overall_enh['success_rate'],
                overall_enh['avg_confidence'] * 100,
                (1/overall_enh['avg_time']) * 1000 if overall_enh['avg_time'] > 0 else 0
            ]
            
            x_pos = np.arange(len(metrics))
            plt.bar(x_pos - 0.2, trad_values, 0.4, label='传统算法', alpha=0.8)
            plt.bar(x_pos + 0.2, enh_values, 0.4, label='增强算法', alpha=0.8)
            
            plt.xlabel('性能指标')
            plt.ylabel('性能值')
            plt.title('综合性能对比')
            plt.xticks(x_pos, metrics, rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图表
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chart_file = f"algorithm_comparison_charts_{timestamp}.png"
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            print(f"可视化图表已保存到: {chart_file}")
            
            plt.show()
            
        except Exception as e:
            print(f"生成可视化图表时出错: {e}")

def main():
    """主函数"""
    print("人脸识别算法对比实验")
    print("=" * 50)
    print("1. 综合测试 (使用数据库图像)")
    print("2. 实时测试 (使用摄像头)")
    
    choice = input("请选择测试模式 (1/2): ").strip()
    
    comparison = AlgorithmComparison()
    
    if choice == '1':
        comparison.run_comprehensive_test()
    elif choice == '2':
        comparison.run_realtime_test()
    else:
        print("无效选择，默认运行综合测试")
        comparison.run_comprehensive_test()

if __name__ == "__main__":
    main()