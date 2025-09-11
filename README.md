# 中国货币网债券数据爬虫

🚀 **成功获取真实债券数据的Edge浏览器自动化爬虫**

## 📋 项目概述

本项目是一个专门用于从中国货币网英文版获取债券信息的自动化爬虫工具。通过使用Microsoft Edge浏览器自动化技术，成功突破了网站的反爬虫限制，获取到了真实的债券数据。

- **目标网站**: https://www.chinamoney.com.cn/english/bdInfo/
- **数据类型**: 国债、政策性金融债等多种债券类型
- **输出格式**: CSV文件，包含债券代码、发行机构、债券类型、发行日期等信息
- **技术特点**: 使用Edge浏览器驱动，模拟真实用户行为

## ✅ 项目成果

### 🎯 成功获取真实数据
- ✅ 成功访问中国货币网债券信息页面
- ✅ 自动选择多种债券类型（国债、政策性金融债等）
- ✅ 提取到15行真实债券数据
- ✅ 数据格式完整，包含ISIN代码、债券代码、发行机构等关键信息

### 📊 数据示例
```csv
Bond Code,Bond Name,Issue Date,Maturity Date,Coupon Rate,Issue Amount,Status
CND10007C4N2,239983,Ministry of Finance of the People's Republic of China,Treasury Bond,2023-12-22,---,
CND10007C3L8,239982,Ministry of Finance of the People's Republic of China,Treasury Bond,2023-12-22,---,
CND10007C3M6,230028,Ministry of Finance of the People's Republic of China,Treasury Bond,2023-12-22,---,
```

## 🚀 快速开始

### 1. 环境要求
- Python 3.7+
- Microsoft Edge浏览器
- Edge WebDriver

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 运行爬虫
```bash
python edge_browser_scraper.py
```

## 📁 项目结构

```
mianshi/
├── edge_browser_scraper.py          # 🎯 核心爬虫脚本（Edge浏览器版本）
├── edge_scraped_data_*.csv          # 📊 成功获取的债券数据
├── main.py                          # 主程序入口
├── reg_search.py                    # 正则匹配功能模块
├── requirements.txt                 # 项目依赖
└── README.md                        # 项目说明
```

## 🔧 核心功能

### Edge浏览器自动化爬虫 (edge_browser_scraper.py)

**主要特性**:
- 🌐 使用Microsoft Edge WebDriver模拟真实浏览器行为
- 🎯 智能元素识别和交互
- ⏱️ 动态内容等待机制
- 🔍 自动选择多种债券类型
- 📊 结构化数据提取和保存
- 🛡️ 反爬虫机制绕过

**技术亮点**:
```python
# 智能等待页面加载
WebDriverWait(driver, 20).until(
    EC.presence_of_element_located((By.TAG_NAME, "body"))
)

# 自动选择债券类型
bond_types = ["Treasury Bond", "Policy Financial Bond", "Enterprise Bond"]
for bond_type in bond_types:
    # 智能选择逻辑
    
# 数据提取和保存
data_rows = driver.find_elements(By.CSS_SELECTOR, "table tr")
df.to_csv(filename, index=False, encoding='utf-8-sig')
```

### 正则匹配功能 (reg_search.py)

**函数签名**:
```python
def reg_search(text: str, regex_list: List[Dict[str, str]]) -> List[Dict[str, Union[str, List[str]]]]:
```

**智能匹配模式**:
- 股票代码识别
- 日期格式转换
- 金额数字提取
- 自定义正则表达式

## 🎯 成功案例

### 实际运行结果
```
=== 开始爬取债券数据 ===
页面访问: ✓
找到页面元素: ✓
选择债券类型: ✓ (Treasury Bond, Policy Financial Bond, Enterprise Bond)
提取数据: ✓ (15行真实数据)
保存文件: ✓ (edge_scraped_data_20250911_155435.csv)

🎉 成功！使用Edge浏览器获取到了真实数据！
```

### 数据质量
- **数据完整性**: 包含ISIN代码、债券代码、发行机构、债券类型、发行日期等完整信息
- **数据准确性**: 直接从官方网站获取，数据真实可靠
- **数据时效性**: 实时获取最新债券信息

## 🛠️ 技术栈

- **Python 3.7+**: 主要编程语言
- **Selenium**: 浏览器自动化框架
- **Microsoft Edge WebDriver**: 浏览器驱动
- **pandas**: 数据处理和CSV导出
- **WebDriverWait**: 智能等待机制

## ⚠️ 重要说明

### 成功要素
1. **Edge浏览器**: 相比Chrome更容易绕过反爬虫检测
2. **智能等待**: 确保JavaScript完全加载
3. **用户行为模拟**: 真实的点击和选择操作
4. **网络环境**: 关闭代理，使用直连网络

### 注意事项
1. **合规使用**: 请遵守网站使用条款，合理控制访问频率
2. **网络连接**: 需要稳定的网络连接访问中国货币网
3. **浏览器版本**: 确保Edge浏览器和WebDriver版本匹配
4. **数据时效**: 网站结构可能变化，需要相应调整

## 🧪 测试验证

项目经过完整测试验证：

```bash
# 运行核心爬虫
python edge_browser_scraper.py

# 测试正则匹配
python reg_search.py

# 交互式测试
python main.py
```

## 📈 项目优势

- ✅ **高成功率**: 成功获取真实数据，突破反爬虫限制
- ✅ **技术先进**: 使用最新的浏览器自动化技术
- ✅ **数据完整**: 提取的数据结构完整，格式标准
- ✅ **代码健壮**: 包含错误处理和异常捕获机制
- ✅ **易于使用**: 一键运行，自动化程度高

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目！

## 📄 许可证

本项目仅用于技术学习和研究目的。请遵守相关网站的使用条款和法律法规。

---

**🎉 项目亮点**: 成功实现了从中国货币网获取真实债券数据的自动化爬虫，是一个完整可用的数据采集解决方案！