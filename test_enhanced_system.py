#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
增强人脸识别系统测试脚本
测试自监督学习和环境反馈功能
'''

import sys
import os
import time
from datetime import datetime

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from test import RealTimeFaceDetection
    import cv2
    import numpy as np
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保所有依赖库已正确安装")
    sys.exit(1)

def test_enhanced_system():
    """测试增强人脸识别系统"""
    print("\n" + "="*60)
    print("🚀 增强人脸识别系统测试")
    print("功能: 多尺度检测 + 自适应阈值 + 环境反馈 + 自监督学习")
    print("="*60)
    
    try:
        # 导入增强模块
        print("\n1. 初始化增强人脸识别系统...")
        face_system = RealTimeFaceDetection()
        
        print("✅ 增强模块导入成功")
        
        # 显示系统配置
        print(f"✓ 系统初始化成功")
        print(f"  - 检测尺度数量: {len(face_system.detection_scales)}")
        print(f"  - 初始自适应阈值: {face_system.environmental_feedback['adaptive_threshold']:.3f}")
        print(f"  - 环境反馈缓冲区大小: {face_system.environmental_feedback['lighting_history'].maxlen}")
        print(f"  - 自监督学习缓冲区大小: {face_system.self_supervised['feature_buffer_size']}")
        
        # 检查摄像头
        print("\n2. 检查摄像头连接...")
        if not face_system.cap.isOpened():
            print("✗ 摄像头未连接或无法打开")
            return False
        print("✓ 摄像头连接正常")
        
        # 测试图像增强功能
        print("\n3. 测试图像增强功能...")
        ret, test_frame = face_system.cap.read()
        if ret:
            # 测试低光照增强
            enhanced_frame = face_system.enhance_image_for_low_light(test_frame)
            lighting_level = face_system.assess_lighting_condition(test_frame)
            print(f"✓ 图像增强功能正常")
            print(f"  - 当前光照水平: {lighting_level:.2f}")
            print(f"  - 增强算法: CLAHE + 伽马校正")
        else:
            print("✗ 无法获取测试帧")
            return False
        
        # 测试多尺度检测
        print("\n4. 测试多尺度人脸检测...")
        detected_faces = face_system.multi_scale_face_detection(test_frame)
        print(f"✓ 多尺度检测完成")
        print(f"  - 检测到人脸数量: {len(detected_faces)}")
        
        # 显示环境模型状态
        print("\n5. 环境模型状态...")
        model_file = "environmental_model.pkl"
        if os.path.exists(model_file):
            print(f"✓ 发现已有环境模型: {model_file}")
            print(f"  - 模型将在识别过程中持续学习和优化")
        else:
            print(f"ℹ 未发现环境模型，将创建新模型")
        
        # 提供测试选项
        print("\n6. 测试选项:")
        print("  a) 运行完整人脸识别测试")
        print("  b) 仅测试系统组件")
        print("  c) 退出测试")
        
        choice = input("\n请选择测试选项 (a/b/c): ").strip().lower()
        
        if choice == 'a':
            print("\n=== 开始完整人脸识别测试 ===")
            print("请面向摄像头，系统将进行识别...")
            
            start_time = time.time()
            recognized_name = face_system.run()
            end_time = time.time()
            
            print(f"\n=== 测试结果 ===")
            print(f"识别结果: {recognized_name if recognized_name else '未识别到有效人脸'}")
            print(f"识别耗时: {end_time - start_time:.2f} 秒")
            
            # 显示最终环境状态
            print(f"\n=== 最终环境状态 ===")
            print(f"自适应阈值: {face_system.environmental_feedback['adaptive_threshold']:.3f}")
            print(f"光照历史记录: {len(face_system.environmental_feedback['lighting_history'])} 条")
            print(f"识别置信度历史: {len(face_system.environmental_feedback['recognition_confidence'])} 条")
            
            # 显示自监督学习状态
            print(f"\n=== 自监督学习状态 ===")
            for name, features in face_system.self_supervised['feature_buffers'].items():
                confidence = face_system.self_supervised['confidence_weights'].get(name, 0.0)
                print(f"用户 {name}: {len(features)} 个特征样本, 置信度权重: {confidence:.3f}")
            
        elif choice == 'b':
            print("\n=== 组件测试完成 ===")
            print("所有系统组件工作正常")
            
        else:
            print("\n测试已取消")
        
        return True
        
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        print("请确保所有依赖已正确安装")
        return False
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 清理资源
        try:
            if 'face_system' in locals():
                face_system.cap.release()
            cv2.destroyAllWindows()
            print("\n✓ 资源已清理")
        except:
            pass

def main():
    """主函数"""
    print("增强人脸识别系统测试工具")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    success = test_enhanced_system()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ 测试完成")
    else:
        print("✗ 测试失败")
    
    input("\n按回车键退出...")

if __name__ == "__main__":
    main()