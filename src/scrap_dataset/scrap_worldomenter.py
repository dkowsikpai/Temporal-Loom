# import necessary libraries 
from bs4 import BeautifulSoup 
import requests 
import re 
import pandas as pd
from tqdm import tqdm
  
  
# function to extract html document from given url 
def getHTMLdocument(url): 
      
    # request for HTML document of given url 
    response = requests.get(url) 
      
    # response will be provided in JSON format 
    return response.text 
  
    
# assign required credentials 
# assign URL 
url_to_scrape = "https://www.worldometers.info/world-population/population-by-country/"
  
# create document 
html_document = getHTMLdocument(url_to_scrape) 
  
# create soap object 
soup = BeautifulSoup(html_document, 'html.parser') 
  
# find the table
table = soup.find('table', attrs={'id':'example2'})
# print(table)
rows = table.find_all('tr')
links = []
for row in rows[1:]:
    link = row.find_all('td')[1].find('a').get('href')
    links.append(link)

print(links)

for link in tqdm(links):
    url = "https://www.worldometers.info" + link
    html_document = getHTMLdocument(url) 
    soup = BeautifulSoup(html_document, 'html.parser') 
    table = soup.find('table', attrs={'class':'table table-striped table-bordered table-hover table-condensed table-list'})
    df = pd.read_html(str(table))[0]
    df.to_csv('./data/worldometer/countries/' + link.split('/')[2] + '.csv', index=False)
    # rows = table.find_all('tr')
    # for row in rows[1:]:
    #     print(row.find_all('td')[0].text)
    # print('------------------')

# # find all the anchor tags with "href"  
# # attribute starting with "https://" 
# for link in soup.find_all('a',  
#                           attrs={'href': re.compile("^https://")}): 
#     # display the actual urls 
#     print(link.get('href'))   