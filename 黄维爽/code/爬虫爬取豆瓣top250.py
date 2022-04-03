from email import header
import encodings
from sqlite3 import paramstyle
from urllib import response
import requests
from lxml import etree
from bs4 import BeautifulSoup
import re
import time
import pandas as pd
import openpyxl

requests.adapters.DEFAULT_RETRIES=100000

url="https://movie.douban.com/top250?start={}&filter="


headers={
    'Cookie':'bid=AlLrm-C4bms; douban-fav-remind=1; gr_user_id=80cc223f-24fb-49c0-938f-eef55f7583e4; viewed="26899701_1121883"; ll="118295"; ap_v=0,6.0; dbcl2="201753095:8YMFO+y3Q9s"; ck=vXWf; push_noty_num=0; push_doumail_num=0',
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36 Edg/99.0.1150.39'
}
pro={
    'https':'http://201.96.33.140'
}  #代理ip

movies=[]  #创建空列表存放电影信息

for i in range(10):
    session=requests.session()   #解决Caused by ProxyError的问题
    session.trust_env=False
    session.keep_alive = False
    response=session.get(url=url.format(i*25),headers=headers,verify=False)
   

# print(response.text)


    html=etree.HTML(response.text)

    lis1=html.xpath('//*[@id="content"]/div/div[1]/ol/li')  
    lis2=html.xpath('//div[@class="info"]')  #爬取 语言与上映日期

    for li in lis1:
        time.sleep(2)
        movie={}

        rank=li.xpath('./div/div[1]/em/text()')[0]  #text()匹配文本
        movie['排名']=rank

        title=li.xpath('./div/div[2]/div[1]/a/span[1]/text()')[0]
        movie['名称']=title

        str_Directors_Actors=li.xpath('./div/div[2]/div[2]/p[1]/text()')[0]
        director=str(str_Directors_Actors.split("主演:")[0]).strip().strip('导演:')  #split拆分字符串  strip()去除前后空格
        movie['导演']=director

        score=li.xpath('./div/div[2]/div[2]/div/span[2]/text()')[0]
        movie['豆瓣评分']=score
        
#       
        movies.append(movie)
    for l in lis2:

        links=l.xpath('div[@class="hd"]/a/@href')[0]
        
        # movie['电影网址']=links
            
        headers={
                'Cookie':'bid=AlLrm-C4bms; douban-fav-remind=1; gr_user_id=80cc223f-24fb-49c0-938f-eef55f7583e4; viewed="26899701_1121883"; ll="118295"; ap_v=0,6.0; dbcl2="201753095:8YMFO+y3Q9s"; ck=vXWf; push_noty_num=0; push_doumail_num=0',
                'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36 Edg/99.0.1150.39',
                'Connection':'close'
                }

        response_link_url=requests.get(url=links,headers=headers)

        #re爬取“语言”
        obj=re.compile(r'<div id="info">.*?<span class="pl">语言:</span>(?P<语言>.*?)<br/>.*?',re.S)  
        result=obj.finditer(response_link_url.text)
        for it in result:

                # print(it.group("语言"))
            movie['语言']=it.group("语言")

        #上映日期用xpath
        movie_html=etree.HTML(response_link_url.text)

        date=movie_html.xpath('//div[@id="info"]//span[@property="v:initialReleaseDate"]/text()')

        movie['上映日期']=date

        response_link_url.close()
print(movies)
response.close()










