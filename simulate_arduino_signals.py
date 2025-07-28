#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模拟Arduino信号发送的测试脚本
用于测试control_ad.py的录音功能
"""

import serial
import time

def simulate_signals():
    """模拟发送Arduino限位信号"""
    try:
        # 连接到同一个串口
        ser = serial.Serial('COM3', 9600, timeout=1)
        time.sleep(2)
        
        print("开始模拟Arduino信号发送...")
        print("将依次发送: X_AXIS -> 等待5秒 -> Z_AXIS")
        
        # 发送X轴限位信号（开始录音）
        print("\n发送 X_AXIS_LIMIT_TRIGGERED 信号...")
        ser.write(b"X_AXIS_LIMIT_TRIGGERED\n")
        ser.flush()
        
        # 等待5秒让录音进行
        for i in range(5, 0, -1):
            print(f"等待 {i} 秒后发送停止信号...")
            time.sleep(1)
        
        # 发送Z轴限位信号（停止录音）
        print("\n发送 Z_AXIS_LIMIT_TRIGGERED 信号...")
        ser.write(b"Z_AXIS_LIMIT_TRIGGERED\n")
        ser.flush()
        
        print("信号发送完成！")
        
        ser.close()
        
    except serial.SerialException as e:
        print(f"串口连接失败: {e}")
        print("请确保control_ad.py程序正在运行并且串口可用")
    except Exception as e:
        print(f"发送信号时出错: {e}")

if __name__ == "__main__":
    simulate_signals()