import serial
import cv2
import time
import pyaudio
import wave
import threading

# 初始化串口连接（修改为你实际的串口号）
ser = serial.Serial('COM3', 9600, timeout=1)
time.sleep(2)  # 等待串口连接稳定

# 初始化摄像头（可选，暂未使用）
cap = cv2.VideoCapture(0)

# 录音参数
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 60  # 最大录音时长
WAVE_OUTPUT_FILENAME = "recording.wav"

# 录音控制变量
is_recording = False
audio = None
stream = None
frames = []

def start_recording():
    """开始录音"""
    global is_recording, audio, stream, frames
    
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
                    # 每5秒显示一次录音状态
                    if frame_count % (RATE // CHUNK * 5) == 0:
                        duration = frame_count * CHUNK / RATE
                        print(f"[录音中] 已录制 {duration:.1f} 秒, 数据帧: {frame_count}")
                except Exception as e:
                    print(f"[错误] 录音过程中出错: {e}")
                    break
        
        record_thread = threading.Thread(target=record)
        record_thread.daemon = True
        record_thread.start()
        print("[状态] 录音线程已启动，等待Z_AXIS_LIMIT_TRIGGERED信号停止...")
        
    except Exception as e:
        print(f"[错误] 启动录音失败: {e}")
        is_recording = False

def stop_recording():
    """停止录音并保存文件"""
    global is_recording, audio, stream, frames
    
    if not is_recording:
        print("[状态] 当前没有录音进行")
        return
    
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
            import os
            file_size = os.path.getsize(WAVE_OUTPUT_FILENAME)
            print(f"[成功] 录音已保存为: {WAVE_OUTPUT_FILENAME} (大小: {file_size} 字节)")
            print(f"[成功] 录音时长约: {len(frames) * CHUNK / RATE:.2f} 秒")
        else:
            print("[警告] 没有录音数据可保存")
        
    except Exception as e:
        print(f"[错误] 保存录音失败: {e}")
    
    # 重置变量
    stream = None
    audio = None
    frames = []
    print("[状态] 录音功能已完全停止，等待下一次信号...")

def main():
    try:
        print("等待Arduino限位信号...")
        print("发送'start'开始录音，或等待限位信号")
        
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                print(f"收到消息: {line}")

                if line == "X_AXIS_LIMIT_TRIGGERED":
                    print("X轴限位触发")
                    start_recording()
                elif line == "Y_AXIS_LIMIT_TRIGGERED":
                    print("Y轴限位触发")
                elif line == "Z_AXIS_LIMIT_TRIGGERED":
                    print("Z轴限位触发 - 停止录音")
                    stop_recording()
            
            # Windows系统下的键盘输入检查（用于手动控制）
            try:
                import msvcrt
                if msvcrt.kbhit():
                    key = msvcrt.getch().decode('utf-8').lower()
                    if key == 's':
                        start_recording()
                        print("手动开始录音")
                    elif key == 'q':
                        stop_recording()
                        print("手动停止录音")
                    elif key == 'x':
                        print("退出程序")
                        break
            except ImportError:
                # 非Windows系统的处理（保留原有逻辑但简化）
                pass
            
            time.sleep(0.1)  # 短暂延时避免CPU占用过高

    except KeyboardInterrupt:
        print("程序终止")
    finally:
        if is_recording:
            stop_recording()
        cap.release()
        cv2.destroyAllWindows()
        ser.close()

if __name__ == "__main__":
    main()
