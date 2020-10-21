def selectOne(line, exp, index=0):
    res = line.select(exp)
    if not res:
        return None
    if len(res) <= index:
        return None
    return res[index].text.strip()


import time
import requests
import xlsxwriter
from bs4 import BeautifulSoup as bs

workbook = xlsxwriter.Workbook('books%s.xlsx'%str(time.time()))
worksheet = workbook.add_worksheet()
worksheet.write(0, 0, 'NO')
worksheet.write(0, 1, 'Book Title')
worksheet.write(0, 2, 'Author')
worksheet.write(0, 3, 'Publisher')
worksheet.write(0, 4, 'URL')


row = 1
for page in range(1,11):
    url = 'http://search.dangdang.com/?key=%BD%BB%CD%A8&act=input&page_index='
    html = requests.get(url+str(page)).text
    soup = bs(html)
    for line in soup.select('.bigimg')[0].select('li'):
        try:
            worksheet.write(row, 0, row)
            worksheet.write(row, 1, selectOne(line, 'a[dd_name=单品标题]'))
            worksheet.write(row, 2, selectOne(line, 'a[dd_name=单品作者]'))
            worksheet.write(row, 3, selectOne(line, 'a[dd_name=单品出版社]'))
            worksheet.write(row, 4, line.select('a[dd_name=单品标题]')[0]['href'])
        except Exception as e:
            print('error:'+str(line))
            print(str(e))
        row += 1
    print('page '+ str(page)+', total '+str(row-1))

workbook.close()
print('finished crawling!')