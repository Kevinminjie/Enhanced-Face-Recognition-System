#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自定义正则匹配函数实现
"""

import re
from typing import List, Dict, Any, Union

def reg_search(text: str, regex_list: List[Dict[str, str]]) -> List[Dict[str, Union[str, List[str]]]]:
    """
    自定义正则匹配函数
    
    Args:
        text: 需要正则匹配的文本内容
        regex_list: 正则表达式列表，格式为 [{'key1': 'pattern1', 'key2': 'pattern2'}]
    
    Returns:
        匹配到的结果列表，格式为 [{'key1': 'match1', 'key2': ['match2', 'match3']}]
    """
    results = []
    
    for regex_dict in regex_list:
        result_dict = {}
        
        for key, pattern in regex_dict.items():
            # 根据不同的匹配需求定义正则表达式
            if key == '标的证券':
                # 匹配股票代码格式 (6位数字.SH 或 .SZ)
                stock_pattern = r'(\d{6}\.[A-Z]{2})'
                matches = re.findall(stock_pattern, text)
                if matches:
                    result_dict[key] = matches[0]  # 取第一个匹配
                else:
                    result_dict[key] = ''
            
            elif key == '换股期限':
                # 匹配日期格式，并转换为标准格式
                date_patterns = [
                    r'(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日',  # 2023年6月2日
                    r'(\d{4})-(\d{1,2})-(\d{1,2})',  # 2023-6-2
                    r'(\d{4})/(\d{1,2})/(\d{1,2})',  # 2023/6/2
                ]
                
                all_dates = []
                for date_pattern in date_patterns:
                    matches = re.findall(date_pattern, text)
                    for match in matches:
                        if len(match) == 3:  # 年月日三个部分
                            year, month, day = match
                            # 格式化为 YYYY-MM-DD
                            formatted_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                            all_dates.append(formatted_date)
                
                if all_dates:
                    result_dict[key] = all_dates
                else:
                    result_dict[key] = []
            
            else:
                # 对于其他自定义模式，使用通用匹配
                if pattern == '*自定义*':
                    # 根据key的含义进行智能匹配
                    if '代码' in key or '编号' in key:
                        # 匹配代码格式
                        code_pattern = r'([A-Z0-9]{6,12})'
                        matches = re.findall(code_pattern, text)
                        result_dict[key] = matches[0] if matches else ''
                    
                    elif '日期' in key or '时间' in key:
                        # 匹配日期
                        date_pattern = r'(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日]?)'
                        matches = re.findall(date_pattern, text)
                        result_dict[key] = matches if matches else []
                    
                    elif '金额' in key or '价格' in key:
                        # 匹配金额
                        amount_pattern = r'([\d,]+\.?\d*)[元万亿]?'
                        matches = re.findall(amount_pattern, text)
                        result_dict[key] = matches[0] if matches else ''
                    
                    else:
                        # 通用文本匹配，提取关键词后的内容
                        general_pattern = f'{re.escape(key)}[：:]*([^，。；;\n]+)'
                        matches = re.findall(general_pattern, text)
                        result_dict[key] = matches[0].strip() if matches else ''
                
                else:
                    # 使用提供的正则表达式
                    try:
                        matches = re.findall(pattern, text)
                        if matches:
                            if len(matches) == 1:
                                result_dict[key] = matches[0]
                            else:
                                result_dict[key] = matches
                        else:
                            result_dict[key] = ''
                    except re.error as e:
                        print(f"正则表达式错误 '{pattern}': {e}")
                        result_dict[key] = ''
        
        results.append(result_dict)
    
    return results

def test_reg_search():
    """
    测试函数
    """
    # 测试用例
    text = '''
标的证券：本期发行的证券为可交换为发行人所持中国长江电力股份
有限公司股票（股票代码：600900.SH，股票简称：长江电力）的可交换公司债
券。
换股期限：本期可交换公司债券换股期限自可交换公司债券发行结束
之日满 12 个月后的第一个交易日起至可交换债券到期日止，即 2023 年 6 月 2
日至 2027 年 6 月 1 日止。
'''
    
    regex_list = [{
        '标的证券': '*自定义*',
        '换股期限': '*自定义*'
    }]
    
    result = reg_search(text, regex_list)
    
    print("测试结果:")
    print(result)
    
    # 期望结果
    expected = [{
        '标的证券': '600900.SH',
        '换股期限': ['2023-06-02', '2027-06-01']
    }]
    
    print("\n期望结果:")
    print(expected)
    
    # 验证结果
    if result == expected:
        print("\n✅ 测试通过！")
    else:
        print("\n❌ 测试失败！")
        print(f"实际结果: {result}")
        print(f"期望结果: {expected}")

def advanced_reg_search_examples():
    """
    更多测试用例
    """
    print("\n=== 高级测试用例 ===")
    
    # 测试用例1：多种格式的日期
    text1 = """
    发行日期：2023年3月15日
    到期日期：2028-12-31
    付息日：每年的6/30和12/30
    """
    
    regex_list1 = [{
        '发行日期': '*自定义*',
        '到期日期': '*自定义*',
        '付息日': '*自定义*'
    }]
    
    result1 = reg_search(text1, regex_list1)
    print("测试用例1 - 日期匹配:")
    print(result1)
    
    # 测试用例2：自定义正则表达式
    text2 = """
    债券代码：123456
    发行规模：100.5亿元
    票面利率：3.25%
    """
    
    regex_list2 = [{
        '债券代码': r'债券代码[：:](\d+)',
        '发行规模': r'发行规模[：:]([\d.]+)亿',
        '票面利率': r'票面利率[：:]([\d.]+)%'
    }]
    
    result2 = reg_search(text2, regex_list2)
    print("\n测试用例2 - 自定义正则:")
    print(result2)

if __name__ == "__main__":
    test_reg_search()
    advanced_reg_search_examples()