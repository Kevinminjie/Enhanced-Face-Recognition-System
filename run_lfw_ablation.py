#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LFW消融实验启动脚本

快速启动test.py算法在LFW数据集上的消融实验
"""

import os
import sys
import subprocess
from pathlib import Path

def check_environment():
    """检查实验环境"""
    print("🔍 检查实验环境...")
    
    # 检查必要的Python包
    required_packages = [
        'opencv-python',
        'dlib', 
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'scikit-learn':
                import sklearn
            else:
                __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} (缺失)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ 缺少必要的Python包: {', '.join(missing_packages)}")
        print("💡 请运行以下命令安装:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    # 检查dlib模型文件
    model_files = [
        "data/data_dlib/shape_predictor_68_face_landmarks.dat",
        "data/data_dlib/dlib_face_recognition_resnet_model_v1.dat"
    ]
    
    for model_file in model_files:
        if Path(model_file).exists():
            print(f"  ✅ {model_file}")
        else:
            print(f"  ❌ {model_file} (缺失)")
            print("💡 请确保dlib模型文件已下载到正确位置")
            return False
    
    print("✅ 环境检查通过")
    return True

def run_lfw_experiment(subset_size=500):
    """运行LFW消融实验"""
    print(f"\n🚀 启动LFW消融实验 (子集大小: {subset_size})")
    print("=" * 60)
    
    try:
        # 导入并运行实验
        from lfw_ablation_study import LFWAblationStudy
        import random
        import numpy as np
        
        # 设置随机种子
        random.seed(42)
        np.random.seed(42)
        
        # 初始化实验
        study = LFWAblationStudy(subset_size=subset_size)
        
        # 下载并准备数据
        if not study.download_lfw_dataset():
            print("❌ LFW数据集准备失败")
            return False
        
        train_data, test_data = study.prepare_lfw_subset()
        if not train_data or not test_data:
            print("❌ 数据子集准备失败")
            return False
        
        # 运行实验
        study.run_all_experiments(train_data, test_data)
        
        # 生成报告
        results_df = study.generate_lfw_report()
        
        print("\n🎉 LFW消融实验完成！")
        return True
        
    except Exception as e:
        print(f"❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_usage():
    """显示使用说明"""
    print("""
📖 LFW消融实验使用说明

🎯 实验目标:
  使用LFW标准数据集验证test.py增强型人脸识别算法的有效性

🔧 实验配置:
  - 数据集: LFW (Labeled Faces in the Wild)
  - 测试模块: 图像增强、多尺度检测、自适应阈值、自监督学习
  - 评估指标: 准确率、精确率、召回率、F1分数、检测成功率

📊 实验输出:
  - lfw_ablation_results.csv: 数值结果
  - lfw_ablation_study_results.png: 可视化图表
  - lfw_ablation_study_report.md: 详细报告

⚠️ 注意事项:
  1. 首次运行会自动下载LFW数据集 (~173MB)
  2. 实验时间取决于子集大小 (推荐500-1000张图片)
  3. 需要稳定的网络连接下载数据集
  4. 确保有足够的磁盘空间 (至少1GB)

💡 使用建议:
  - 快速测试: 子集大小 200-500
  - 标准测试: 子集大小 500-1000  
  - 完整测试: 子集大小 1000+
""")

def main():
    """主函数"""
    print("🔬 test.py增强型人脸识别算法 - LFW消融实验")
    print("=" * 70)
    
    # 显示使用说明
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        show_usage()
        return
    
    # 检查环境
    if not check_environment():
        print("\n❌ 环境检查失败，请解决上述问题后重试")
        return
    
    # 获取子集大小参数
    subset_size = 500  # 默认值
    if len(sys.argv) > 1:
        try:
            subset_size = int(sys.argv[1])
            if subset_size < 100:
                print("⚠️ 子集大小过小，建议至少100张图片")
                subset_size = 100
            elif subset_size > 2000:
                print("⚠️ 子集大小过大，可能导致实验时间过长")
                response = input("是否继续? (y/n): ")
                if response.lower() != 'y':
                    return
        except ValueError:
            print(f"❌ 无效的子集大小参数: {sys.argv[1]}")
            print("💡 使用默认值: 500")
    
    print(f"\n📋 实验配置:")
    print(f"  📊 数据集: LFW (Labeled Faces in the Wild)")
    print(f"  🔢 子集大小: {subset_size} 张图片")
    print(f"  🎯 实验类型: 消融研究")
    print(f"  📈 评估指标: 准确率、精确率、召回率、F1分数等")
    
    # 确认开始实验
    print("\n⚠️ 注意: 首次运行会下载LFW数据集 (~173MB)")
    response = input("是否开始实验? (y/n): ")
    if response.lower() != 'y':
        print("❌ 实验已取消")
        return
    
    # 运行实验
    success = run_lfw_experiment(subset_size)
    
    if success:
        print("\n🎉 实验成功完成！")
        print("\n📁 生成的文件:")
        print("  📊 lfw_ablation_results.csv - 实验数据")
        print("  📈 lfw_ablation_study_results.png - 可视化图表")
        print("  📄 lfw_ablation_study_report.md - 详细报告")
        
        print("\n💡 下一步:")
        print("  1. 查看可视化图表了解各模块性能")
        print("  2. 阅读详细报告了解技术分析")
        print("  3. 使用CSV数据进行进一步分析")
    else:
        print("\n❌ 实验失败")
        print("💡 故障排除:")
        print("  1. 检查网络连接")
        print("  2. 确保有足够磁盘空间")
        print("  3. 检查dlib模型文件")
        print("  4. 查看错误日志")

if __name__ == "__main__":
    main()