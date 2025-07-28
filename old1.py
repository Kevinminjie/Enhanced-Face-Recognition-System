from flask import Flask, render_template, request
import time
import os
import json
from threading import Lock
import mysql.connector
from mysql.connector import Error

app = Flask(__name__)

# MySQL数据库配置
DB_CONFIG = {
    'host': 'aws.tianle666.xyz',
    'port': 3306,
    'database': 'factory_management',
    'user': 'root',
    'password': 'Ztl20040720.',
    'charset': 'utf8mb4'
}

# 全局变量存储结果
current_result = None
current_person = 1  # 默认人员类型
current_person = 1
result_lock = Lock()


def fetch_latest_data(job_type):
    """从数据库中获取最新数据"""
    connection = None
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            cursor = connection.cursor(dictionary=True)
            sql = f"""
            SELECT * FROM employee_workpiece
            WHERE job_type = %s
            ORDER BY updated_at DESC
            LIMIT 1
            
            """
            cursor.execute(sql, (job_type,))
            result = cursor.fetchone()
            return result
    except Error as e:
        print(f"数据库操作错误: {e}")
        return None
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()


@app.route('/')
def index():
    global current_result, current_person
    with result_lock:
        try:
            # 获取阶段6和阶段7的数据
            stage6_data = fetch_latest_data('工人')
            stage7_data = fetch_latest_data('质检员')

            return render_template('version2140.html',
                                   stage6=stage6_data,
                                   stage7=stage7_data)
        except Exception as e:
            print(f"解析JSON出错: {str(e)}")
            return render_template('version2140.html', error=str(e))


@app.route('/api/result', methods=['POST'])
def receive_result():
    global current_result, current_person, PERSON_id
    with result_lock:
        try:
            current_result = request.json.get('result')
            current_person = request.json.get('person')  # 默认工人
            PERSON_id = request.json.get('person_id')  # 默认工人
            print(PERSON_id)
            print(current_person)

            print(f"接收到原始数据: {current_result}")
            if isinstance(current_result, str):
                # 尝试解析字符串为字典
                try:
                    current_result = json.loads(current_result)
                except json.JSONDecodeError:
                    # 处理键值对字符串
                    data = {}
                    for line in current_result.split('\n'):
                        if ':' in line:
                            key, value = line.split(':', 1)
                            data[key.strip()] = value.strip()
                    current_result = data

            print(f"处理后数据(PERSON={current_person},{PERSON_id}): {current_result}")
            return {'status': 'success'}
        except Exception as e:
            print(f"处理结果出错: {str(e)}")
            return {'status': 'error', 'message': str(e)}, 500


@app.route('/api/latest-data/<job_type>', methods=['GET'])
def get_latest_data(job_type):
    """提供最新数据的API"""
    try:
        data = fetch_latest_data(job_type)
        if data:
            return {'status': 'success', 'data': data}
        else:
            return {'status': 'error', 'message': 'No data found'}, 404
    except Exception as e:
        print(f"获取数据出错: {str(e)}")
        return {'status': 'error', 'message': str(e)}, 500


if __name__ == '__main__':
    app.run(port=5001, debug=True)
