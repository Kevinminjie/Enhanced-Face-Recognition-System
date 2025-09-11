#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
开发工程师测试题 - 主程序
包含债券数据爬取和正则匹配两个功能
"""

from bond_scraper import BondScraper
from reg_search import reg_search, test_reg_search
import sys

def main():
    """
    主函数 - 执行两个编程题
    """
    print("=" * 60)
    print("开发工程师测试题")
    print("=" * 60)
    
    while True:
        print("\n请选择要执行的功能:")
        print("1. 债券数据爬取 (从中国货币网获取Treasury Bond 2023数据)")
        print("2. 正则匹配函数测试")
        print("3. 执行所有功能")
        print("4. 退出")
        
        choice = input("\n请输入选择 (1-4): ").strip()
        
        if choice == '1':
            execute_bond_scraping()
        elif choice == '2':
            execute_regex_test()
        elif choice == '3':
            execute_bond_scraping()
            print("\n" + "="*50)
            execute_regex_test()
        elif choice == '4':
            print("程序退出")
            break
        else:
            print("无效选择，请重新输入")

def execute_bond_scraping():
    """
    执行债券数据爬取功能
    """
    print("\n" + "="*50)
    print("一、债券数据爬取")
    print("="*50)
    
    try:
        scraper = BondScraper()
        
        print("正在从中国货币网获取数据...")
        print("筛选条件: Bond Type=Treasury Bond, Issue Year=2023")
        
        bond_data = scraper.get_bond_data(bond_type="Treasury Bond", issue_year="2023")
        
        if bond_data:
            print(f"\n✅ 成功获取 {len(bond_data)} 条债券数据")
            
            # 保存为CSV
            filename = "treasury_bonds_2023.csv"
            scraper.save_to_csv(bond_data, filename)
            
            # 显示数据预览
            print("\n📊 数据预览 (前3条):")
            print("-" * 80)
            for i, bond in enumerate(bond_data[:3]):
                print(f"\n第 {i+1} 条记录:")
                for key, value in bond.items():
                    print(f"  {key:15}: {value}")
            
            print(f"\n💾 完整数据已保存到: {filename}")
            
        else:
            print("\n❌ 未能获取到数据")
            print("注意: 由于网站可能有反爬虫机制，当前显示的是模拟数据")
            print("实际部署时需要根据网站具体结构调整爬虫策略")
            
    except Exception as e:
        print(f"\n❌ 执行债券数据爬取时出错: {e}")

def execute_regex_test():
    """
    执行正则匹配函数测试
    """
    print("\n" + "="*50)
    print("二、正则匹配函数测试")
    print("="*50)
    
    try:
        # 执行基本测试
        print("\n🧪 执行基本功能测试...")
        test_reg_search()
        
        # 交互式测试
        print("\n" + "-"*40)
        print("💡 交互式测试")
        print("-"*40)
        
        while True:
            print("\n请选择测试选项:")
            print("1. 使用示例文本测试")
            print("2. 输入自定义文本测试")
            print("3. 返回主菜单")
            
            sub_choice = input("请选择 (1-3): ").strip()
            
            if sub_choice == '1':
                demo_regex_test()
            elif sub_choice == '2':
                custom_regex_test()
            elif sub_choice == '3':
                break
            else:
                print("无效选择")
                
    except Exception as e:
        print(f"\n❌ 执行正则匹配测试时出错: {e}")

def demo_regex_test():
    """
    演示正则匹配功能
    """
    print("\n📝 使用题目示例进行测试:")
    
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
    
    print("\n输入文本:")
    print(text)
    
    print("\n正则表达式列表:")
    print(regex_list)
    
    result = reg_search(text, regex_list)
    
    print("\n🎯 匹配结果:")
    for i, res in enumerate(result):
        print(f"结果 {i+1}: {res}")

def custom_regex_test():
    """
    自定义文本测试
    """
    print("\n✏️  自定义文本测试")
    
    print("请输入要匹配的文本 (输入 'END' 结束):")
    lines = []
    while True:
        line = input()
        if line.strip().upper() == 'END':
            break
        lines.append(line)
    
    text = '\n'.join(lines)
    
    if not text.strip():
        print("文本为空，返回")
        return
    
    print("\n请输入要匹配的关键词 (用逗号分隔):")
    keywords = input().strip().split(',')
    keywords = [k.strip() for k in keywords if k.strip()]
    
    if not keywords:
        print("没有输入关键词，返回")
        return
    
    # 构建正则表达式列表
    regex_dict = {}
    for keyword in keywords:
        regex_dict[keyword] = '*自定义*'
    
    regex_list = [regex_dict]
    
    print(f"\n🔍 开始匹配关键词: {keywords}")
    
    result = reg_search(text, regex_list)
    
    print("\n🎯 匹配结果:")
    for i, res in enumerate(result):
        print(f"结果 {i+1}:")
        for key, value in res.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n程序执行出错: {e}")
        sys.exit(1)