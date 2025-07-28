#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速运行消融实验脚本

使用方法:
1. 确保本地人脸数据库存在 (data/database_faces/)
2. 确保dlib模型文件存在
3. 运行: python run_ablation.py
"""

import os
import sys
from pathlib import Path

def check_requirements():
    """检查运行环境"""
    print("🔍 检查运行环境...")
    
    # 检查数据库目录
    db_path = Path("data/database_faces")
    if not db_path.exists():
        print(f"❌ 人脸数据库目录不存在: {db_path}")
        print("   请确保数据库目录存在并包含人脸图片")
        return False
    
    # 检查是否有人脸数据
    person_dirs = [d for d in db_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    if len(person_dirs) < 2:
        print(f"❌ 人脸数据库中身份数量不足: {len(person_dirs)}")
        print("   至少需要2个不同身份的人脸数据")
        return False
    
    total_images = 0
    for person_dir in person_dirs:
        images = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.jpeg")) + list(person_dir.glob("*.png"))
        total_images += len(images)
    
    if total_images < 10:
        print(f"❌ 图片数量不足: {total_images}")
        print("   至少需要10张人脸图片")
        return False
    
    print(f"✅ 发现 {len(person_dirs)} 个身份，共 {total_images} 张图片")
    
    # 检查dlib模型文件
    model_files = [
        "data/data_dlib/shape_predictor_68_face_landmarks.dat",
        "data/data_dlib/dlib_face_recognition_resnet_model_v1.dat"
    ]
    
    for model_file in model_files:
        if not Path(model_file).exists():
            print(f"❌ dlib模型文件不存在: {model_file}")
            print("   请确保dlib模型文件已下载并放置在正确位置")
            return False
    
    print("✅ dlib模型文件检查通过")
    
    # 检查Python包
    required_packages = ['cv2', 'dlib', 'numpy', 'pandas', 'matplotlib', 'seaborn', 'sklearn']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ 缺少必要的Python包: {', '.join(missing_packages)}")
        print("   请安装: pip install opencv-python dlib numpy pandas matplotlib seaborn scikit-learn")
        return False
    
    print("✅ Python包检查通过")
    
    return True

def run_ablation_study():
    """运行消融实验"""
    print("\n🚀 启动消融实验")
    print("=" * 60)
    
    try:
        from local_ablation_study import main
        main()
    except ImportError as e:
        print(f"❌ 导入消融实验模块失败: {e}")
        print("   请确保 local_ablation_study.py 文件存在")
        return False
    except Exception as e:
        print(f"❌ 运行消融实验时出错: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def show_usage():
    """显示使用说明"""
    print("""
📖 消融实验使用说明

🎯 实验目的:
   验证test.py增强型人脸识别算法各个创新模块的有效性

📋 实验内容:
   1. 基线算法测试 (传统方法)
   2. 图像增强模块测试
   3. 多尺度检测模块测试
   4. 完整增强算法测试

📊 评估指标:
   - 识别准确率
   - 检测成功率
   - 平均处理时间
   - 平均置信度

📁 数据要求:
   - 数据库路径: data/database_faces/
   - 每个身份至少2张图片
   - 总计至少10张图片
   - 支持格式: jpg, jpeg, png

🔧 环境要求:
   - Python 3.6+
   - OpenCV, dlib, numpy, pandas, matplotlib, seaborn, scikit-learn
   - dlib预训练模型文件

📈 输出结果:
   - local_ablation_results.csv: 数值结果
   - local_ablation_study_results.png: 可视化图表
   - local_ablation_study_report.md: 详细报告

💡 使用建议:
   - 确保数据库包含多样化的人脸图片
   - 包含不同光照、角度、距离的图片
   - 每个身份至少3-5张图片效果更好
""")

def main():
    """主函数"""
    print("🔬 test.py增强型人脸识别算法消融实验")
    print("=" * 60)
    
    # 显示使用说明
    show_usage()
    
    # 检查运行环境
    if not check_requirements():
        print("\n❌ 环境检查失败，请解决上述问题后重试")
        return
    
    # 询问是否继续
    print("\n🤔 环境检查通过，是否开始消融实验？")
    response = input("   输入 'y' 或 'yes' 继续，其他键退出: ").lower().strip()
    
    if response not in ['y', 'yes', '是', '好']:
        print("👋 实验已取消")
        return
    
    # 运行消融实验
    success = run_ablation_study()
    
    if success:
        print("\n🎉 消融实验完成！")
        print("\n📁 生成的文件:")
        print("   📊 local_ablation_results.csv - 实验数据")
        print("   📈 local_ablation_study_results.png - 结果图表")
        print("   📄 local_ablation_study_report.md - 详细报告")
        
        print("\n💡 建议:")
        print("   1. 查看生成的图表了解各模块性能")
        print("   2. 阅读详细报告了解技术分析")
        print("   3. 根据结果优化算法参数")
    else:
        print("\n❌ 消融实验失败，请检查错误信息")

if __name__ == "__main__":
    main()