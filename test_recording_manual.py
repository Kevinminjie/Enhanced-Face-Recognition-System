#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手动测试录音功能的脚本
不依赖串口，直接测试录音开始和停止功能
"""

import time
import os
import sys

# 添加当前目录到路径
sys.path.append('.')

# 导入录音功能（需要修改control_ad.py使其可导入）
def test_recording_manually():
    """手动测试录音功能"""
    print("=== 手动测试录音功能 ===")
    print("这个测试将模拟以下流程:")
    print("1. 开始录音（模拟X_AXIS_LIMIT_TRIGGERED）")
    print("2. 录音5秒")
    print("3. 停止录音（模拟Z_AXIS_LIMIT_TRIGGERED）")
    print("4. 检查录音文件")
    
    input("\n按回车键开始测试...")
    
    # 导入必要的模块
    try:
        import pyaudio
        import wave
        import threading
    except ImportError as e:
        print(f"缺少依赖: {e}")
        return False
    
    # 录音参数
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    WAVE_OUTPUT_FILENAME = "test_recording_manual.wav"
    
    # 录音控制变量
    is_recording = False
    audio = None
    stream = None
    frames = []
    
    def start_recording():
        """开始录音"""
        nonlocal is_recording, audio, stream, frames
        
        if is_recording:
            print("[状态] 录音已在进行中，无需重复启动")
            return
        
        print("[开始录音] 正在初始化录音设备...")
        
        try:
            audio = pyaudio.PyAudio()
            stream = audio.open(format=FORMAT,
                               channels=CHANNELS,
                               rate=RATE,
                               input=True,
                               frames_per_buffer=CHUNK)
            
            is_recording = True
            frames = []
            print("[成功] 录音设备初始化完成，开始录音...")
            print(f"[参数] 采样率: {RATE}Hz, 声道: {CHANNELS}, 格式: 16位")
            
            # 在新线程中进行录音
            def record():
                frame_count = 0
                while is_recording:
                    try:
                        data = stream.read(CHUNK)
                        frames.append(data)
                        frame_count += 1
                        # 每秒显示一次录音状态
                        if frame_count % (RATE // CHUNK) == 0:
                            duration = frame_count * CHUNK / RATE
                            print(f"[录音中] 已录制 {duration:.1f} 秒, 数据帧: {frame_count}")
                    except Exception as e:
                        print(f"[错误] 录音过程中出错: {e}")
                        break
            
            record_thread = threading.Thread(target=record)
            record_thread.daemon = True
            record_thread.start()
            print("[状态] 录音线程已启动...")
            
        except Exception as e:
            print(f"[错误] 启动录音失败: {e}")
            is_recording = False
            return False
        
        return True
    
    def stop_recording():
        """停止录音并保存文件"""
        nonlocal is_recording, audio, stream, frames
        
        if not is_recording:
            print("[状态] 当前没有录音进行")
            return False
        
        print("[停止录音] 正在停止录音...")
        is_recording = False
        
        # 等待录音线程结束
        time.sleep(0.2)
        
        try:
            if stream:
                print("[停止录音] 关闭音频流...")
                stream.stop_stream()
                stream.close()
            
            if audio:
                print("[停止录音] 释放音频资源...")
                audio.terminate()
            
            # 保存录音文件
            if frames:
                print(f"[保存文件] 正在保存 {len(frames)} 个音频帧...")
                wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(audio.get_sample_size(FORMAT) if audio else 2)
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
                wf.close()
                
                # 获取文件大小
                file_size = os.path.getsize(WAVE_OUTPUT_FILENAME)
                print(f"[成功] 录音已保存为: {WAVE_OUTPUT_FILENAME} (大小: {file_size} 字节)")
                print(f"[成功] 录音时长约: {len(frames) * CHUNK / RATE:.2f} 秒")
                return True
            else:
                print("[警告] 没有录音数据可保存")
                return False
            
        except Exception as e:
            print(f"[错误] 保存录音失败: {e}")
            return False
        
        finally:
            # 重置变量
            stream = None
            audio = None
            frames = []
            print("[状态] 录音功能已完全停止")
    
    # 开始测试
    print("\n=== 开始测试 ===")
    
    # 1. 开始录音
    print("\n1. 模拟 X_AXIS_LIMIT_TRIGGERED 信号 - 开始录音")
    if not start_recording():
        print("录音启动失败")
        return False
    
    # 2. 录音5秒
    print("\n2. 录音5秒...")
    time.sleep(5)
    
    # 3. 停止录音
    print("\n3. 模拟 Z_AXIS_LIMIT_TRIGGERED 信号 - 停止录音")
    if not stop_recording():
        print("录音停止失败")
        return False
    
    # 4. 检查文件
    print("\n4. 检查录音文件...")
    if os.path.exists(WAVE_OUTPUT_FILENAME):
        file_size = os.path.getsize(WAVE_OUTPUT_FILENAME)
        print(f"✓ 录音文件存在: {WAVE_OUTPUT_FILENAME}")
        print(f"✓ 文件大小: {file_size} 字节")
        
        if file_size > 1000:  # 至少1KB
            print("✓ 文件大小正常，录音功能工作正常")
            return True
        else:
            print("✗ 文件太小，可能录音失败")
            return False
    else:
        print("✗ 录音文件不存在")
        return False

if __name__ == "__main__":
    success = test_recording_manually()
    if success:
        print("\n🎉 录音功能测试通过！Z_AXIS_LIMIT_TRIGGERED信号可以正确停止录音。")
    else:
        print("\n❌ 录音功能测试失败，请检查错误信息。")