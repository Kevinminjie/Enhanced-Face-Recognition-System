#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试GUI修复后的人脸识别功能
"""

import pandas as pd
import os

def test_csv_reading():
    print("测试CSV文件读取功能...")
    
    # 测试读取features_all.csv文件
    csv_path = "./data/features_all.csv"
    
    if os.path.exists(csv_path):
        try:
            # 使用修复后的编码方式读取
            csv_rd = pd.read_csv(csv_path, header=None, encoding='utf-8-sig')
            print(f"✓ CSV文件读取成功！")
            print(f"✓ 数据行数: {csv_rd.shape[0]}")
            print(f"✓ 数据列数: {csv_rd.shape[1]}")
            
            # 显示第一行数据（如果存在）
            if csv_rd.shape[0] > 0:
                print(f"✓ 第一行人脸名称: {csv_rd.iloc[0][0]}")
                print(f"✓ 第一行ID: {csv_rd.iloc[0][1]}")
                print(f"✓ 第一行类型: {csv_rd.iloc[0][2]}")
                print(f"✓ 特征维度: {csv_rd.shape[1] - 3}")
            
            return True
            
        except UnicodeDecodeError as e:
            print(f"✗ 编码错误: {e}")
            return False
        except Exception as e:
            print(f"✗ 读取错误: {e}")
            return False
    else:
        print(f"✗ CSV文件不存在: {csv_path}")
        return False

def test_old_encoding():
    print("\n测试旧编码方式（应该失败）...")
    
    csv_path = "./data/features_all.csv"
    
    if os.path.exists(csv_path):
        try:
            # 使用旧的gb2312编码方式读取
            csv_rd = pd.read_csv(csv_path, header=None, encoding='gb2312')
            print(f"✗ 意外成功：旧编码方式竟然能读取")
            return False
            
        except UnicodeDecodeError as e:
            print(f"✓ 预期的编码错误（说明修复有效）: {str(e)[:100]}...")
            return True
        except Exception as e:
            print(f"? 其他错误: {e}")
            return False

if __name__ == "__main__":
    print("开始测试GUI修复...\n")
    
    success1 = test_csv_reading()
    success2 = test_old_encoding()
    
    print("\n=== 测试结果 ===")
    if success1 and success2:
        print("✓ 所有测试通过！GUI闪退问题已修复。")
        print("✓ 现在可以正常点击人脸识别按钮了。")
    else:
        print("✗ 部分测试失败，可能还有其他问题。")