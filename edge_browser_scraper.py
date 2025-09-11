#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用Edge浏览器爬取中国货币网债券数据
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.edge.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import pandas as pd
import csv
from datetime import datetime

class EdgeBrowserScraper:
    def __init__(self):
        self.driver = None
        self.setup_driver()
    
    def setup_driver(self):
        """设置Edge浏览器驱动"""
        try:
            edge_options = Options()
            # 设置为非无头模式，可以看到浏览器操作
            # edge_options.add_argument('--headless')
            edge_options.add_argument('--no-sandbox')
            edge_options.add_argument('--disable-dev-shm-usage')
            edge_options.add_argument('--disable-gpu')
            edge_options.add_argument('--window-size=1920,1080')
            
            # 设置用户代理
            edge_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59')
            
            # 禁用图片加载以提高速度
            prefs = {
                "profile.managed_default_content_settings.images": 2
            }
            edge_options.add_experimental_option("prefs", prefs)
            
            self.driver = webdriver.Edge(options=edge_options)
            self.driver.implicitly_wait(10)
            print("Edge浏览器驱动设置成功")
            return True
            
        except Exception as e:
            print(f"Edge浏览器驱动设置失败: {str(e)}")
            print("请确保已安装Microsoft Edge WebDriver")
            print("下载地址: https://developer.microsoft.com/en-us/microsoft-edge/tools/webdriver/")
            return False
    
    def access_page(self):
        """访问债券信息页面"""
        try:
            url = 'https://www.chinamoney.com.cn/english/bdInfo/'
            print(f"\n正在使用Edge浏览器访问: {url}")
            
            self.driver.get(url)
            time.sleep(5)  # 等待页面完全加载
            
            # 获取页面标题
            title = self.driver.title
            print(f"页面标题: {title}")
            
            # 检查页面是否正常加载
            if "Bond Information" in title or "CFETS" in title:
                print("✓ 页面访问成功")
                return True
            else:
                print("✗ 页面标题异常")
                return False
                
        except Exception as e:
            print(f"页面访问失败: {str(e)}")
            return False
    
    def wait_and_analyze_page(self):
        """等待并分析页面内容"""
        try:
            print("\n等待页面动态内容加载...")
            time.sleep(8)  # 给更多时间让JavaScript执行
            
            # 分析页面结构
            print("\n=== 页面结构分析 ===")
            
            # 查找表单
            forms = self.driver.find_elements(By.TAG_NAME, "form")
            print(f"找到 {len(forms)} 个表单")
            
            # 查找选择框
            selects = self.driver.find_elements(By.TAG_NAME, "select")
            print(f"找到 {len(selects)} 个选择框")
            
            # 查找输入框
            inputs = self.driver.find_elements(By.TAG_NAME, "input")
            print(f"找到 {len(inputs)} 个输入框")
            
            # 查找按钮
            buttons = self.driver.find_elements(By.TAG_NAME, "button")
            input_buttons = self.driver.find_elements(By.CSS_SELECTOR, "input[type='button'], input[type='submit']")
            total_buttons = len(buttons) + len(input_buttons)
            print(f"找到 {total_buttons} 个按钮")
            
            # 查找表格
            tables = self.driver.find_elements(By.TAG_NAME, "table")
            print(f"找到 {len(tables)} 个表格")
            
            # 详细分析选择框
            if selects:
                print("\n=== 选择框详细分析 ===")
                for i, select in enumerate(selects):
                    try:
                        select_obj = Select(select)
                        options = select_obj.options
                        print(f"\n选择框 {i+1}:")
                        print(f"  选项数量: {len(options)}")
                        
                        # 显示所有选项
                        for j, option in enumerate(options):
                            option_text = option.text.strip()
                            option_value = option.get_attribute('value')
                            print(f"  选项 {j+1}: '{option_text}' (value='{option_value}')")
                            
                            # 检查是否有国债相关选项
                            if any(keyword in option_text.lower() for keyword in ['treasury', 'government', 'bond', '国债']):
                                print(f"    *** 发现国债相关选项: {option_text} ***")
                                
                    except Exception as e:
                        print(f"分析选择框 {i+1} 时出错: {str(e)}")
            
            # 详细分析表格
            if tables:
                print("\n=== 表格详细分析 ===")
                for i, table in enumerate(tables):
                    try:
                        rows = table.find_elements(By.TAG_NAME, "tr")
                        print(f"\n表格 {i+1}:")
                        print(f"  行数: {len(rows)}")
                        
                        if rows:
                            # 分析表头
                            header_row = rows[0]
                            header_cells = header_row.find_elements(By.TAG_NAME, "th") + header_row.find_elements(By.TAG_NAME, "td")
                            if header_cells:
                                header_texts = [cell.text.strip() for cell in header_cells]
                                print(f"  表头: {header_texts}")
                            
                            # 分析数据行
                            if len(rows) > 1:
                                print(f"  数据行数: {len(rows) - 1}")
                                for j, row in enumerate(rows[1:3]):  # 只显示前2行数据
                                    cells = row.find_elements(By.TAG_NAME, "td")
                                    if cells:
                                        cell_texts = [cell.text.strip() for cell in cells]
                                        print(f"  数据行 {j+1}: {cell_texts}")
                                        
                    except Exception as e:
                        print(f"分析表格 {i+1} 时出错: {str(e)}")
            
            return len(selects) > 0 or len(tables) > 0
            
        except Exception as e:
            print(f"分析页面时出错: {str(e)}")
            return False
    
    def try_search_operations(self):
        """尝试搜索操作"""
        try:
            print("\n=== 尝试搜索操作 ===")
            
            # 查找可能的搜索相关元素
            search_elements = []
            
            # 查找包含"国债"或"Treasury"的选项
            selects = self.driver.find_elements(By.TAG_NAME, "select")
            for select in selects:
                try:
                    select_obj = Select(select)
                    options = select_obj.options
                    
                    for option in options:
                        option_text = option.text.strip().lower()
                        if any(keyword in option_text for keyword in ['treasury', 'government', 'bond', '国债']):
                            print(f"找到相关选项: {option.text}")
                            # 尝试选择这个选项
                            try:
                                select_obj.select_by_visible_text(option.text)
                                print(f"已选择: {option.text}")
                                time.sleep(2)
                                search_elements.append(('select', option.text))
                            except Exception as e:
                                print(f"选择选项失败: {str(e)}")
                                
                except Exception as e:
                    print(f"处理选择框时出错: {str(e)}")
            
            # 查找并点击搜索按钮
            search_buttons = [
                "button[onclick*='search']",
                "input[value*='Search']",
                "input[value*='查询']",
                "button:contains('Search')",
                "#searchBtn",
                ".search-btn",
                "button[type='submit']",
                "input[type='submit']"
            ]
            
            for selector in search_buttons:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        print(f"找到搜索按钮: {selector}")
                        elements[0].click()
                        print("已点击搜索按钮")
                        time.sleep(5)  # 等待搜索结果
                        search_elements.append(('button', selector))
                        break
                except Exception as e:
                    continue
            
            return len(search_elements) > 0
            
        except Exception as e:
            print(f"搜索操作时出错: {str(e)}")
            return False
    
    def extract_final_data(self):
        """提取最终数据"""
        try:
            print("\n=== 提取最终数据 ===")
            
            # 再次检查表格
            tables = self.driver.find_elements(By.TAG_NAME, "table")
            print(f"当前页面表格数量: {len(tables)}")
            
            extracted_data = []
            
            for i, table in enumerate(tables):
                try:
                    rows = table.find_elements(By.TAG_NAME, "tr")
                    print(f"\n表格 {i+1} 包含 {len(rows)} 行")
                    
                    if len(rows) > 1:  # 有数据行
                        # 获取表头
                        header_row = rows[0]
                        headers = []
                        header_cells = header_row.find_elements(By.TAG_NAME, "th") + header_row.find_elements(By.TAG_NAME, "td")
                        for cell in header_cells:
                            headers.append(cell.text.strip())
                        
                        print(f"表头: {headers}")
                        
                        # 获取数据行
                        for j, row in enumerate(rows[1:]):
                            cells = row.find_elements(By.TAG_NAME, "td")
                            if cells:
                                row_data = []
                                for cell in cells:
                                    row_data.append(cell.text.strip())
                                
                                # 检查是否有实际数据（非空）
                                if any(data.strip() for data in row_data):
                                    extracted_data.append(row_data)
                                    print(f"数据行 {j+1}: {row_data}")
                                    
                                    # 限制显示数量
                                    if j >= 4:  # 只显示前5行
                                        print(f"... 还有 {len(rows) - 1 - j - 1} 行数据")
                                        break
                        
                except Exception as e:
                    print(f"提取表格 {i+1} 数据时出错: {str(e)}")
            
            # 保存数据
            if extracted_data:
                print(f"\n成功提取到 {len(extracted_data)} 行真实数据！")
                
                # 保存到CSV文件
                filename = f"edge_scraped_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                
                # 创建标准的债券数据格式
                standard_headers = ['Bond Code', 'Bond Name', 'Issue Date', 'Maturity Date', 'Coupon Rate', 'Issue Amount', 'Status']
                
                with open(filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(standard_headers)
                    
                    for i, row in enumerate(extracted_data):
                        # 将提取的数据映射到标准格式
                        standard_row = []
                        for j in range(len(standard_headers)):
                            if j < len(row):
                                standard_row.append(row[j])
                            else:
                                standard_row.append('')  # 填充空值
                        writer.writerow(standard_row)
                
                print(f"数据已保存到: {filename}")
                return True
            else:
                print("未提取到真实数据")
                return False
                
        except Exception as e:
            print(f"提取数据时出错: {str(e)}")
            return False
    
    def run_scraping(self):
        """运行完整的爬取流程"""
        try:
            print("=== 开始使用Edge浏览器爬取债券数据 ===")
            
            if not self.driver:
                print("✗ Edge浏览器驱动初始化失败")
                return False
            
            # 1. 访问页面
            if not self.access_page():
                return False
            
            # 2. 等待并分析页面
            has_elements = self.wait_and_analyze_page()
            
            # 3. 尝试搜索操作
            if has_elements:
                search_success = self.try_search_operations()
                if search_success:
                    print("搜索操作执行成功")
                    time.sleep(5)  # 等待搜索结果加载
                else:
                    print("未找到可操作的搜索元素")
            
            # 4. 提取最终数据
            data_extracted = self.extract_final_data()
            
            # 5. 总结结果
            print("\n=== 爬取结果总结 ===")
            print(f"页面访问: ✓")
            print(f"找到页面元素: {'✓' if has_elements else '✗'}")
            print(f"提取到真实数据: {'✓' if data_extracted else '✗'}")
            
            if data_extracted:
                print("\n🎉 成功！使用Edge浏览器获取到了真实数据！")
            else:
                print("\n❌ 仍然无法获取真实数据")
                print("\n可能的原因:")
                print("1. 页面需要特定的用户交互才能显示数据")
                print("2. 数据通过复杂的AJAX请求异步加载")
                print("3. 网站检测到自动化访问并限制数据显示")
                print("4. 需要登录或特殊权限才能访问数据")
            
            return data_extracted
            
        except Exception as e:
            print(f"爬取过程中出现异常: {str(e)}")
            return False
        
        finally:
            if self.driver:
                print("\n保持浏览器打开10秒供查看...")
                time.sleep(10)  # 让用户看到最终状态
                print("关闭浏览器...")
                self.driver.quit()

def main():
    """主函数"""
    print("使用Edge浏览器爬取中国货币网债券数据")
    print("请确保已安装Microsoft Edge WebDriver")
    
    scraper = EdgeBrowserScraper()
    success = scraper.run_scraping()
    
    if success:
        print("\n✅ 爬取成功：获取到真实数据")
    else:
        print("\n❌ 爬取完成：未获取到真实数据")
        print("\n建议:")
        print("1. 手动访问网站查看是否需要特殊操作")
        print("2. 检查网站是否有反爬虫机制")
        print("3. 考虑使用官方API或其他数据源")

if __name__ == '__main__':
    main()