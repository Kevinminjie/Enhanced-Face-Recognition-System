#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI功能测试脚本
测试人脸识别系统的各项功能
"""

import os
import sys
import pandas as pd
import cv2
import numpy as np
import dlib
from pathlib import Path

def test_environment():
    """测试环境配置"""
    print("=== 环境配置测试 ===")
    
    # 检查必要的库
    try:
        import dlib
        print("✓ dlib库导入成功")
    except ImportError:
        print("✗ dlib库导入失败")
        return False
    
    try:
        import cv2
        print(f"✓ OpenCV版本: {cv2.__version__}")
    except ImportError:
        print("✗ OpenCV导入失败")
        return False
    
    try:
        import pandas as pd
        print(f"✓ Pandas版本: {pd.__version__}")
    except ImportError:
        print("✗ Pandas导入失败")
        return False
    
    return True

def test_dlib_models():
    """测试Dlib模型文件"""
    print("\n=== Dlib模型测试 ===")
    
    # 检查模型文件
    model_path = "../data/data_dlib/shape_predictor_68_face_landmarks.dat"
    if os.path.exists(model_path):
        print("✓ 人脸关键点检测模型存在")
    else:
        print("✗ 人脸关键点检测模型不存在")
        return False
    
    recognition_model_path = "../data/data_dlib/dlib_face_recognition_resnet_model_v1.dat"
    if os.path.exists(recognition_model_path):
        print("✓ 人脸识别模型存在")
    else:
        print("✗ 人脸识别模型不存在")
        return False
    
    # 测试模型加载
    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(model_path)
        face_rec_model = dlib.face_recognition_model_v1(recognition_model_path)
        print("✓ 所有Dlib模型加载成功")
        return True
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return False

def test_face_database():
    """测试人脸数据库"""
    print("\n=== 人脸数据库测试 ===")
    
    csv_path = "../data/features_all.csv"
    if not os.path.exists(csv_path):
        print("✗ 人脸特征数据库不存在")
        return False
    
    try:
        # 测试读取CSV文件
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        print(f"✓ 成功读取人脸数据库，共{len(df)}条记录")
        
        # 检查数据结构
        expected_columns = ['name', 'id', 'type'] + [f'feature_{i}' for i in range(128)]
        if len(df.columns) == 131:  # 3个基本字段 + 128个特征
            print("✓ 数据库结构正确")
        else:
            print(f"✗ 数据库结构异常，列数: {len(df.columns)}")
            return False
        
        # 显示数据库内容
        if len(df) > 0:
            print("数据库中的人员:")
            for _, row in df.iterrows():
                print(f"  - 姓名: {row['name']}, 工号: {row['id']}, 工种: {row['type']}")
        else:
            print("⚠ 数据库为空")
        
        return True
    except Exception as e:
        print(f"✗ 读取数据库失败: {e}")
        return False

def test_test_images():
    """测试测试图片"""
    print("\n=== 测试图片检查 ===")
    
    test_img_dir = "test_img"
    if not os.path.exists(test_img_dir):
        print("✗ 测试图片目录不存在")
        return False
    
    image_files = [f for f in os.listdir(test_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(image_files) == 0:
        print("✗ 测试图片目录为空")
        return False
    
    print(f"✓ 找到{len(image_files)}张测试图片:")
    for img_file in image_files[:5]:  # 只显示前5张
        print(f"  - {img_file}")
    
    # 测试图片读取 - 尝试多张图片
    success_count = 0
    for img_file in image_files[:3]:
        test_img_path = os.path.join(test_img_dir, img_file)
        try:
            # 使用cv2.imdecode处理中文路径
            img_data = np.fromfile(test_img_path, dtype=np.uint8)
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            
            if img is not None:
                print(f"✓ 成功读取测试图片: {img_file} (尺寸: {img.shape})")
                success_count += 1
            else:
                print(f"✗ 无法解码测试图片: {img_file}")
        except Exception as e:
            print(f"✗ 读取图片时出错 {img_file}: {e}")
    
    if success_count > 0:
        print(f"✓ 成功读取{success_count}张测试图片")
        return True
    else:
        print("✗ 无法读取任何测试图片")
        return False

def test_face_detection():
    """测试人脸检测功能"""
    print("\n=== 人脸检测功能测试 ===")
    
    try:
        # 加载模型
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("../data/data_dlib/shape_predictor_68_face_landmarks.dat")
        face_rec_model = dlib.face_recognition_model_v1("../data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")
        
        # 测试图片
        test_img_dir = "test_img"
        image_files = [f for f in os.listdir(test_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        success_count = 0
        for img_file in image_files[:3]:  # 测试前3张图片
            img_path = os.path.join(test_img_dir, img_file)
            
            try:
                # 使用cv2.imdecode处理中文路径
                img_data = np.fromfile(img_path, dtype=np.uint8)
                img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                
                if img is None:
                    print(f"⚠ {img_file}: 无法读取图片")
                    continue
                
                # 转换为RGB
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # 检测人脸
                faces = detector(rgb_img)
                
                if len(faces) > 0:
                    print(f"✓ {img_file}: 检测到{len(faces)}张人脸")
                    
                    # 提取特征
                    for face in faces:
                        landmarks = predictor(rgb_img, face)
                        face_descriptor = face_rec_model.compute_face_descriptor(rgb_img, landmarks)
                        if len(face_descriptor) == 128:
                            success_count += 1
                            print(f"  - 成功提取128维特征向量")
                        else:
                            print(f"  - 特征向量维度异常: {len(face_descriptor)}")
                else:
                    print(f"⚠ {img_file}: 未检测到人脸")
            except Exception as e:
                print(f"✗ 处理图片 {img_file} 时出错: {e}")
        
        if success_count > 0:
            print(f"✓ 人脸检测功能正常，成功处理{success_count}张人脸")
            return True
        else:
            print("✗ 人脸检测功能异常")
            return False
            
    except Exception as e:
        print(f"✗ 人脸检测测试失败: {e}")
        return False

def test_gui_components():
    """测试GUI组件"""
    print("\n=== GUI组件测试 ===")
    
    # 检查GUI相关文件
    gui_files = [
        "FaceRecognition.py",
        "FaceRecognition_UI.py",
        "runMain.py"
    ]
    
    for file in gui_files:
        if os.path.exists(file):
            print(f"✓ {file} 存在")
        else:
            print(f"✗ {file} 不存在")
            return False
    
    # 检查图片资源
    if os.path.exists("images_test"):
        print("✓ GUI图片资源目录存在")
    else:
        print("✗ GUI图片资源目录不存在")
    
    return True

def main():
    """主测试函数"""
    print("人脸识别GUI系统功能测试")
    print("=" * 50)
    
    # 切换到GUI目录
    os.chdir("FaceRecUI")
    
    test_results = []
    
    # 执行各项测试
    test_results.append(("环境配置", test_environment()))
    test_results.append(("Dlib模型", test_dlib_models()))
    test_results.append(("人脸数据库", test_face_database()))
    test_results.append(("测试图片", test_test_images()))
    test_results.append(("人脸检测", test_face_detection()))
    test_results.append(("GUI组件", test_gui_components()))
    
    # 汇总结果
    print("\n" + "=" * 50)
    print("测试结果汇总:")
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总体结果: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！GUI系统功能正常")
    else:
        print("⚠ 部分测试失败，请检查相关组件")
    
    return passed == total

if __name__ == "__main__":
    main()