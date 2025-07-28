#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI交互功能测试脚本
模拟用户操作GUI的各个功能模块
"""

import os
import sys
import time
import pandas as pd
import cv2
import numpy as np
import dlib
from pathlib import Path

# 添加GUI模块路径
sys.path.append('FaceRecUI')

def test_face_recognition_module():
    """测试人脸识别核心模块"""
    print("\n=== 人脸识别核心模块测试 ===")
    
    try:
        # 导入FaceRecognition模块
        from FaceRecognition import Face_MainWindow
        
        # 创建人脸识别实例（需要MainWindow参数，这里只测试导入）
        print("✓ 成功导入Face_MainWindow类")
        
        # 测试基本模型加载（不需要实例化GUI）
        try:
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor("data/data_dlib/shape_predictor_68_face_landmarks.dat")
            face_rec_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")
            print("✓ 人脸检测器加载成功")
            print("✓ 关键点预测器加载成功")
            print("✓ 人脸识别模型加载成功")
        except Exception as e:
            print(f"⚠ 模型加载失败: {e}")
            return False
        
        return True
        
    except ImportError as e:
        print(f"✗ 无法导入FaceRecognition模块: {e}")
        return False
    except Exception as e:
        print(f"✗ 人脸识别模块测试失败: {e}")
        return False

def test_database_operations():
    """测试数据库操作功能"""
    print("\n=== 数据库操作功能测试 ===")
    
    try:
        # 测试读取现有数据库
        csv_path = "data/features_all.csv"
        if os.path.exists(csv_path):
            # CSV文件没有标题行，第一列是姓名，后面128列是特征
            # 尝试多种编码方式
            encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb2312', 'latin1']
            csv_rd = None
            for encoding in encodings:
                try:
                    csv_rd = pd.read_csv(csv_path, header=None, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if csv_rd is None:
                raise Exception("无法使用任何编码读取CSV文件")
            print(f"✓ 成功读取数据库，当前有{csv_rd.shape[0]}条记录")
            
            # 显示数据库内容
            if csv_rd.shape[0] > 0:
                print("当前数据库记录:")
                for i in range(min(3, csv_rd.shape[0])):
                    name = csv_rd.iloc[i][0]  # 第一列是姓名
                    print(f"  {i+1}. 姓名: {name}")
            
            # 测试数据库结构
            expected_cols = 129  # 1个姓名字段 + 128个特征
            if csv_rd.shape[1] == expected_cols:
                print("✓ 数据库结构正确")
            else:
                print(f"⚠ 数据库结构异常，期望{expected_cols}列，实际{csv_rd.shape[1]}列")
            
            return True
        else:
            print("✗ 数据库文件不存在")
            return False
            
    except Exception as e:
        print(f"✗ 数据库操作测试失败: {e}")
        return False

def test_image_processing():
    """测试图像处理功能"""
    print("\n=== 图像处理功能测试 ===")
    
    try:
        # 加载模型
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("data/data_dlib/shape_predictor_68_face_landmarks.dat")
        face_rec_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")
        
        # 测试图片处理流程
        test_img_dir = "FaceRecUI/test_img"
        image_files = [f for f in os.listdir(test_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        processed_count = 0
        for img_file in image_files[:2]:  # 测试前2张图片
            img_path = os.path.join(test_img_dir, img_file)
            
            try:
                # 读取图片
                img_data = np.fromfile(img_path, dtype=np.uint8)
                img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                
                if img is None:
                    continue
                
                # 图像预处理
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # 人脸检测
                faces = detector(rgb_img)
                
                if len(faces) > 0:
                    print(f"✓ {img_file}: 检测到{len(faces)}张人脸")
                    
                    for face in faces:
                        # 关键点检测
                        landmarks = predictor(rgb_img, face)
                        
                        # 特征提取
                        face_descriptor = face_rec_model.compute_face_descriptor(rgb_img, landmarks)
                        
                        if len(face_descriptor) == 128:
                            print(f"  - 特征提取成功: 128维向量")
                            processed_count += 1
                        else:
                            print(f"  - 特征提取异常: {len(face_descriptor)}维")
                else:
                    print(f"⚠ {img_file}: 未检测到人脸")
                    
            except Exception as e:
                print(f"✗ 处理图片 {img_file} 失败: {e}")
        
        if processed_count > 0:
            print(f"✓ 图像处理功能正常，成功处理{processed_count}张人脸")
            return True
        else:
            print("✗ 图像处理功能异常")
            return False
            
    except Exception as e:
        print(f"✗ 图像处理测试失败: {e}")
        return False

def test_face_comparison():
    """测试人脸比对功能"""
    print("\n=== 人脸比对功能测试 ===")
    
    try:
        # 加载模型
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("data/data_dlib/shape_predictor_68_face_landmarks.dat")
        face_rec_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")
        
        # 读取数据库
        csv_path = "data/features_all.csv"
        if not os.path.exists(csv_path):
            print("✗ 人脸数据库不存在")
            return False
        
        # 尝试多种编码方式
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb2312', 'latin1']
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(csv_path, header=None, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            raise Exception("无法使用任何编码读取CSV文件")
        if len(df) == 0:
            print("✗ 人脸数据库为空")
            return False
        
        # 获取数据库中的特征向量
        db_features = []
        db_names = []
        for _, row in df.iterrows():
            features = [row[i] for i in range(1, 129)]  # 第1-128列是特征数据
            db_features.append(np.array(features))
            db_names.append(row[0])  # 第0列是姓名
        
        print(f"✓ 加载数据库特征，共{len(db_features)}个人员")
        
        # 测试图片比对
        test_img_dir = "FaceRecUI/test_img"
        image_files = [f for f in os.listdir(test_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        comparison_count = 0
        for img_file in image_files[:2]:  # 测试前2张图片
            img_path = os.path.join(test_img_dir, img_file)
            
            try:
                # 读取并处理图片
                img_data = np.fromfile(img_path, dtype=np.uint8)
                img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                
                if img is None:
                    continue
                
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                faces = detector(rgb_img)
                
                if len(faces) > 0:
                    for face in faces:
                        landmarks = predictor(rgb_img, face)
                        face_descriptor = face_rec_model.compute_face_descriptor(rgb_img, landmarks)
                        
                        if len(face_descriptor) == 128:
                            # 计算与数据库的相似度
                            similarities = []
                            for db_feature in db_features:
                                # 计算欧氏距离
                                distance = np.linalg.norm(np.array(face_descriptor) - db_feature)
                                similarity = 1 / (1 + distance)  # 转换为相似度
                                similarities.append(similarity)
                            
                            # 找到最相似的人员
                            max_similarity = max(similarities)
                            best_match_idx = similarities.index(max_similarity)
                            best_match_name = db_names[best_match_idx]
                            
                            print(f"✓ {img_file}: 最佳匹配 - {best_match_name} (相似度: {max_similarity:.3f})")
                            comparison_count += 1
                            
            except Exception as e:
                print(f"✗ 比对图片 {img_file} 失败: {e}")
        
        if comparison_count > 0:
            print(f"✓ 人脸比对功能正常，完成{comparison_count}次比对")
            return True
        else:
            print("✗ 人脸比对功能异常")
            return False
            
    except Exception as e:
        print(f"✗ 人脸比对测试失败: {e}")
        return False

def test_gui_ui_components():
    """测试GUI界面组件"""
    print("\n=== GUI界面组件测试 ===")
    
    try:
        # 检查UI文件
        ui_files = [
            "FaceRecUI/FaceRecognition_UI.py",
            "FaceRecUI/FaceRecognition_UI.ui",
            "FaceRecUI/runMain.py"
        ]
        
        for ui_file in ui_files:
            if os.path.exists(ui_file):
                print(f"✓ {os.path.basename(ui_file)} 存在")
            else:
                print(f"✗ {os.path.basename(ui_file)} 不存在")
                return False
        
        # 检查图片资源
        img_resource_dir = "FaceRecUI/images_test"
        if os.path.exists(img_resource_dir):
            resource_files = os.listdir(img_resource_dir)
            print(f"✓ 图片资源目录存在，包含{len(resource_files)}个文件")
        else:
            print("✗ 图片资源目录不存在")
            return False
        
        # 检查字体文件
        font_dir = "FaceRecUI/Font"
        if os.path.exists(font_dir):
            font_files = os.listdir(font_dir)
            print(f"✓ 字体目录存在，包含{len(font_files)}个字体文件")
        else:
            print("⚠ 字体目录不存在")
        
        return True
        
    except Exception as e:
        print(f"✗ GUI界面组件测试失败: {e}")
        return False

def test_file_system_permissions():
    """测试文件系统权限"""
    print("\n=== 文件系统权限测试 ===")
    
    try:
        # 测试数据目录读写权限
        data_dir = "data"
        if os.path.exists(data_dir) and os.access(data_dir, os.R_OK | os.W_OK):
            print("✓ 数据目录读写权限正常")
        else:
            print("✗ 数据目录权限异常")
            return False
        
        # 测试数据库文件权限
        csv_path = "data/features_all.csv"
        if os.path.exists(csv_path) and os.access(csv_path, os.R_OK | os.W_OK):
            print("✓ 数据库文件读写权限正常")
        else:
            print("✗ 数据库文件权限异常")
            return False
        
        # 测试模型文件权限
        model_dir = "data/data_dlib"
        if os.path.exists(model_dir) and os.access(model_dir, os.R_OK):
            print("✓ 模型目录读取权限正常")
        else:
            print("✗ 模型目录权限异常")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ 文件系统权限测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("GUI交互功能全面测试")
    print("=" * 60)
    
    test_results = []
    
    # 执行各项测试
    test_results.append(("人脸识别核心模块", test_face_recognition_module()))
    test_results.append(("数据库操作功能", test_database_operations()))
    test_results.append(("图像处理功能", test_image_processing()))
    test_results.append(("人脸比对功能", test_face_comparison()))
    test_results.append(("GUI界面组件", test_gui_ui_components()))
    test_results.append(("文件系统权限", test_file_system_permissions()))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("GUI功能测试结果汇总:")
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总体结果: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("🎉 所有GUI功能测试通过！系统可以正常使用")
        print("\n建议测试项目:")
        print("1. 启动GUI程序并测试人脸识别功能")
        print("2. 测试人脸录入功能")
        print("3. 测试数据库管理功能")
        print("4. 测试实时摄像头识别")
    else:
        print("⚠ 部分GUI功能测试失败，请检查相关组件")
    
    return passed == total

if __name__ == "__main__":
    main()