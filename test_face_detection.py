#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试人脸检测功能
"""

import cv2
import dlib
import numpy as np
import pandas as pd
import os

def test_face_detection():
    print("开始测试人脸检测功能...")
    
    # 初始化dlib检测器
    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('./data/data_dlib/shape_predictor_68_face_landmarks.dat')
        face_reco_model = dlib.face_recognition_model_v1('./data/data_dlib/dlib_face_recognition_resnet_model_v1.dat')
        print("✓ Dlib模型加载成功")
    except Exception as e:
        print(f"✗ Dlib模型加载失败: {e}")
        return False
    
    # 检查人脸数据库
    try:
        df = pd.read_csv('./data/features_all.csv')
        print(f"✓ 人脸数据库加载成功，包含 {len(df)} 个人脸特征")
    except Exception as e:
        print(f"✗ 人脸数据库加载失败: {e}")
        return False
    
    # 测试图像人脸检测
    test_images = [
        './FaceRecUI/test_img/宋仲基.jpeg',
        './FaceRecUI/test_img/朴信惠-1.jpeg',
        './FaceRecUI/test_img/玄彬-1.jpeg'
    ]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            try:
                # 读取图像
                img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
                if img is None:
                    print(f"✗ 无法读取图像: {img_path}")
                    continue
                
                # 检测人脸
                faces = detector(img, 1)
                print(f"✓ {os.path.basename(img_path)}: 检测到 {len(faces)} 个人脸")
                
                # 如果检测到人脸，提取特征
                if len(faces) > 0:
                    shape = predictor(img, faces[0])
                    face_descriptor = face_reco_model.compute_face_descriptor(img, shape)
                    print(f"  - 成功提取128维人脸特征")
                    
            except Exception as e:
                print(f"✗ 处理图像 {img_path} 时出错: {e}")
        else:
            print(f"✗ 测试图像不存在: {img_path}")
    
    print("\n人脸检测功能测试完成！")
    return True

if __name__ == "__main__":
    test_face_detection()