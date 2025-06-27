import requests
from bs4 import BeautifulSoup

# 目标网页的URL
url = "https://www.zysj.com.cn/lilunshuji/neikexue/quanben.html"  # 替换为你要解析的网页地址

# 目标网页的URL
# url = "https://example.com"  # 替换为你要解析的网页地址

# 发送HTTP请求获取网页内容
response = requests.get(url)

# 检查请求是否成功
if response.status_code == 200:
    # 设置响应的编码（如果需要）
    response.encoding = 'utf-8'  # 假设网页内容是UTF-8编码

    # 使用BeautifulSoup解析HTML
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # 查找id为"content"的div元素
    content_div = soup.find('div', id='content')
    # print(content_div)
    # 如果找到了这个div元素
    if content_div:
        # 查找div下的所有section元素
        sections = content_div.find_all('div', class_='section')
        print(f"找到{len(sections)}个section元素。")
        
        # 遍历每个section
        for section in sections:
            # 提取section中的title（假设title是一个h1、h2或其他标题标签）
            title = section.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            if title:
                print(f"标题: {title.get_text(strip=True)}")
            
            # 提取section中的所有段落
            paragraphs = section.find_all('p')
            for p in paragraphs:
                print(f"段落: {p.get_text(strip=True)}")
    else:
        print("未找到id为'content'的div元素。")
else:
    print(f"请求失败，状态码：{response.status_code}")