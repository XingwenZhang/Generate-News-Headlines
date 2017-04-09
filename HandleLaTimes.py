
# coding: utf-8

# In[68]:

import urllib.request as ur
import re
import os
import json
from lxml import html as ht


# In[70]:

dir = '/home/angus/work/mastercourse/544/project/data/LATimesData/LATimesDownloadData/'
pattern_content = re.compile(r'<meta name="Description" content="(.*?)">',re.S)
pattern_title = re.compile(r'<meta property="og:title" content="(.*?)">', re.S)

#pattern_content = re.compile(r'<meta.*?name="Description".*?content="(.*?)".*?/>',re.S)
#pattern_title = re.compile(r'<meta property="og:title" content="(.*?)".*?/>', re.S)
files = os.listdir(dir)
json_file=[]
for file in files:
    #print(file)
    file_name = os.path.join(dir,file)
    #html = ur.urlopen(file_name)
    html = ht.parse(file_name)
    #html = html.decode('utf-8')
    html = ht.tostring(html).decode('utf-8')
    #print(html)
    content = re.findall(pattern_content, html)
    title = re.findall(pattern_title, html)
    #print(content)
    #print(title)
    title_content = dict()
    title_content['title']=title
    title_content['content']=content
    json_file.append(title_content)
    print(len(json_file))
with open('/home/angus/work/mastercourse/544/project/LaTimes.txt','w') as f:
    f.write(json.dumps(json_file, separators=(',',':'), indent=4, ensure_ascii=False))


# html = ur.urlopen('file:///home/angus/work/mastercourse/544/project/data/LATimesData/LATimesDownloadData/0a0ab482-3c21-45e5-b8a3-db5615b9db1a.html').read()
# #result = html.xpath('//meta')
# html = html.decode('utf-8')
# print(html)


# In[53]:

# pattern_content = re.compile(r'<meta.*?name="Description".*?content="(.*?)".*?/>',re.S)
# pattern_title = re.compile(r'<meta property="og:title" content="(.*?)".*?/>', re.S)


# In[55]:

# content = re.findall(pattern_content, html)
# title = re.findall(pattern_title, html)
# print(content)
# print(title)


# In[ ]:



