# -*- coding: utf-8 -*-
"""
Created on Sun May  7 19:48:31 2023

@author: HP
"""

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException
import time
import pandas as pd

PATH = 'D:/Study/MLDL/chromedriver'


l=list()
o={}


target_url_list = []
target_url_list.append("https://www.glassdoor.com/Job/india-data-scientist-jobs-SRCH_IL.0,5_IN115_KO6,20.htm?clickSource=searchBox")
target_url_list.append("https://www.glassdoor.com/Job/india-data-scientist-jobs-SRCH_IL.0,5_IN115_KO6,20_IP2.htm?includeNoSalaryJobs=true&pgc=AB4AAYEAHgAAAAAAAAAAAAAAAgJ%2F49wAQgEAA8FD%2B6SU5TPNVdQU3C4%2FnLR41rkYvTJyixa7C%2FrGu%2BIuy4CoQNjKdBGzO07mEs00%2FleyBbUpAhUFQheYumTNigAA")
target_url_list.append("https://www.glassdoor.com/Job/india-data-scientist-jobs-SRCH_IL.0,5_IN115_KO6,20_IP3.htm?includeNoSalaryJobs=true&pgc=AB4AAoEAPAAAAAAAAAAAAAAAAgJ%2F49wAdAEDAQoaBhgHAZn0yhJbEutTOf9eqw4zKzYfHiFS6OnJhptVVmOOIZUFcCjEbnFjr4uasHrLgDYlt49NQhPlsYjc0ta%2BKTA0U4%2Fi%2BNDGRZC8kYbX1OlzthzsfptMErO5O2vqAGzqQwNrkFkhFZWhpEmDVAA5AAA%3D")
target_url_list.append("https://www.glassdoor.com/Job/india-data-scientist-jobs-SRCH_IL.0,5_IN115_KO6,20_IP4.htm?includeNoSalaryJobs=true&pgc=AB4AA4EAWgAAAAAAAAAAAAAAAgJ%2F49wAlAECARo2BwGIdWRU9AuDO8bfRuGmI2UFiMGOq7BPmmKh4hVY9INXFXKutPS1z3Nwlmw7HrUij9ucyLghcpmyiHV3c97m65%2F72Tp175z7OUTqDmAASuHCHMY3eD0tK1i2o9ZvK5Ki%2BnEFWAoqffVSn56HQxrFpNAQ0Xmp6ihZXHiJUWQw%2FYyvlQw1%2BnQJjbruMWwIs%2F8AAA%3D%3D")
target_url_list.append("https://www.glassdoor.com/Job/india-data-scientist-jobs-SRCH_IL.0,5_IN115_KO6,20_IP5.htm?includeNoSalaryJobs=true&pgc=AB4ABIEAeAAAAAAAAAAAAAAAAgJ%2F49wArQECARo2By71TfFdokL6r%2FKqqzvHkzXcbFXyrv%2FEEEfbn8X3NM%2BgrLmNCNan1bEE7R%2Fo9t2MaynGw2%2FzPMoy6Fw4Tm9i3oIZ8wYkH7VzaD5D6vcNU%2BjYWFrQAwgnot0V5Ej28p2Sd4joLy74UMVolxPA%2FUICRclgehEzgBn9scqoIszgJkU1zd7w97KQWADN89PFbeOeQCUVGDCDgiufS1SeYUnX1pk7qUYftILlAAA%3D")
target_url_list.append("https://www.glassdoor.com/Job/india-data-scientist-jobs-SRCH_IL.0,5_IN115_KO6,20_IP6.htm?includeNoSalaryJobs=true&pgc=AB4ABYEAlgAAAAAAAAAAAAAAAgJ%2F49wAyQEDASJsCDgGMOYjdCR5f0%2BG6gLhRxFCaa%2BmtMtq1wdmCmokGxhoIIC8G%2BwCWY0LXEHNcTKKaG5OA3%2Bx7eSk6hqsqNXndSQ63kiLzDdL0EWtXsC4N2le29sYHfbbXDXT1BDU6A92mzUZbnnZYVhwS8g4wE5I%2FBe9GDJv1wOiMKlQ1XXCDHdNZqzYcD2wduoOA4lZYn0TVTV1ineJNzp9Nq3i2xkZ2UMyl1pcCStELNJ7XIGzysMvB%2BiAsS6VEzXOCGpracph93BtCAAA")
target_url_list.append("https://www.glassdoor.com/Job/india-data-scientist-jobs-SRCH_IL.0,5_IN115_KO6,20_IP7.htm?includeNoSalaryJobs=true&pgc=AB4ABoEAtAAAAAAAAAAAAAAAAgKAXEYA3QEEASJaEUwGLgkbmuZ4sZ91DvIfXpZQYBnFUwVJO7dloRqRALD5Kv7KxrGhFxgVGhV0QSFmERkQ1S3lmPuRxwzjDSLNDAmHjtitM%2FMKReuUTifPLOzO%2BtrWYQCjIxHvCh1XcPrAiveR1TJ%2BsqkJP9pQ3LPtxmSDMQvEA%2BrV40wvwljm18N7u%2BeFKtTT%2FH2R2IuKjA8f0G9IJunvPvr2dlJslNwz5LbEIQcOWzXXATZuEnp%2BqRA9vOIzKX%2F57E4a0I9BaMzI19c01186h%2FNECcrI5jpgyUA1YiGX25%2BIAAA%3D")
target_url_list.append("https://www.glassdoor.com/Job/india-data-scientist-jobs-SRCH_IL.0,5_IN115_KO6,20_IP8.htm?includeNoSalaryJobs=true")
target_url_list.append("https://www.glassdoor.com/Job/india-data-scientist-jobs-SRCH_IL.0,5_IN115_KO6,20_IP9.htm?includeNoSalaryJobs=true")
target_url_list.append("https://www.glassdoor.com/Job/india-data-scientist-jobs-SRCH_IL.0,5_IN115_KO6,20_IP10.htm?includeNoSalaryJobs=true")
target_url_list.append("https://www.glassdoor.com/Job/india-data-scientist-jobs-SRCH_IL.0,5_IN115_KO6,20_IP11.htm?includeNoSalaryJobs=true")
target_url_list.append("https://www.glassdoor.com/Job/india-data-scientist-jobs-SRCH_IL.0,5_IN115_KO6,20_IP12.htm?includeNoSalaryJobs=true")
target_url_list.append("https://www.glassdoor.com/Job/india-data-scientist-jobs-SRCH_IL.0,5_IN115_KO6,20_IP13.htm?includeNoSalaryJobs=true")
target_url_list.append("https://www.glassdoor.com/Job/india-data-scientist-jobs-SRCH_IL.0,5_IN115_KO6,20_IP14.htm?includeNoSalaryJobs=true")
target_url_list.append("https://www.glassdoor.com/Job/india-data-scientist-jobs-SRCH_IL.0,5_IN115_KO6,20_IP15.htm?includeNoSalaryJobs=true")
target_url_list.append("https://www.glassdoor.com/Job/india-data-scientist-jobs-SRCH_IL.0,5_IN115_KO6,20_IP16.htm?includeNoSalaryJobs=true")
target_url_list.append("https://www.glassdoor.com/Job/india-data-scientist-jobs-SRCH_IL.0,5_IN115_KO6,20_IP17.htm?includeNoSalaryJobs=true")
target_url_list.append("https://www.glassdoor.com/Job/india-data-scientist-jobs-SRCH_IL.0,5_IN115_KO6,20_IP18.htm?includeNoSalaryJobs=true")
target_url_list.append("https://www.glassdoor.com/Job/india-data-scientist-jobs-SRCH_IL.0,5_IN115_KO6,20_IP19.htm?includeNoSalaryJobs=true")
target_url_list.append("https://www.glassdoor.com/Job/india-data-scientist-jobs-SRCH_IL.0,5_IN115_KO6,20_IP20.htm?includeNoSalaryJobs=true")
target_url_list.append("https://www.glassdoor.com/Job/india-data-scientist-jobs-SRCH_IL.0,5_IN115_KO6,20_IP21.htm?includeNoSalaryJobs=true")
target_url_list.append("https://www.glassdoor.com/Job/india-data-scientist-jobs-SRCH_IL.0,5_IN115_KO6,20_IP22.htm?includeNoSalaryJobs=true")
target_url_list.append("https://www.glassdoor.com/Job/india-data-scientist-jobs-SRCH_IL.0,5_IN115_KO6,20_IP23.htm?includeNoSalaryJobs=true")
target_url_list.append("https://www.glassdoor.com/Job/india-data-scientist-jobs-SRCH_IL.0,5_IN115_KO6,20_IP24.htm?includeNoSalaryJobs=true")
target_url_list.append("https://www.glassdoor.com/Job/india-data-scientist-jobs-SRCH_IL.0,5_IN115_KO6,20_IP25.htm?includeNoSalaryJobs=true")
target_url_list.append("https://www.glassdoor.com/Job/india-data-scientist-jobs-SRCH_IL.0,5_IN115_KO6,20_IP26.htm?includeNoSalaryJobs=true")
target_url_list.append("https://www.glassdoor.com/Job/india-data-scientist-jobs-SRCH_IL.0,5_IN115_KO6,20_IP27.htm?includeNoSalaryJobs=true")
target_url_list.append("https://www.glassdoor.com/Job/india-data-scientist-jobs-SRCH_IL.0,5_IN115_KO6,20_IP28.htm?includeNoSalaryJobs=true")
target_url_list.append("https://www.glassdoor.com/Job/india-data-scientist-jobs-SRCH_IL.0,5_IN115_KO6,20_IP29.htm?includeNoSalaryJobs=true")
target_url_list.append("https://www.glassdoor.com/Job/india-data-scientist-jobs-SRCH_IL.0,5_IN115_KO6,20_IP30.htm?includeNoSalaryJobs=true")

for target_url in target_url_list:

    driver=webdriver.Chrome(PATH)
    
    driver.get(target_url)
    
    #Let the page load. Change this number based on your internet speed.
    #Or, wait until the webpage is loaded, instead of hardcoding it.
    time.sleep(15)

     #Test for the "Sign Up" prompt and get rid of it.
    try:
        driver.find_element_by_class_name("selected").click()
    except ElementClickInterceptedException:
        pass
 
    time.sleep(.1)
    try:
        driver.find_element_by_css_selector('[alt="Close"]').click() #clicking to the X.
        print(' x out worked')
    except NoSuchElementException:
        print(' x out failed')
        pass

    
    resp = driver.page_source
    
    
    soup=BeautifulSoup(resp,'html.parser')
    
    allJobsContainer = soup.find("ul",{"class":"css-7ry9k1"})
    
    allJobs = allJobsContainer.find_all("li")
    print('jobs Found')
    for job in allJobs:
        try:
            o["name-of-company"]=job.find("div",{"class":"d-flex justify-content-between align-items-start"}).text
        except:
            o["name-of-company"]=None
    
        try:
            o["name-of-job"]=job.find("a",{"class":"jobLink css-1rd3saf eigr9kq2"}).text
        except:
            o["name-of-job"]=None
    
    
        try:
            o["location"]=job.find("div",{"class":"d-flex flex-wrap css-11d3uq0 e1rrn5ka2"}).text
        except:
            o["location"]=None
    
    
        try:
            o["salary"]=job.find("div",{"class":"css-3g3psg pr-xxsm"}).text
        except:
            o["salary"]=None
            
            
        l.append(o)
        o={}
    print(len(l))
    print(l[len(l)-1])
        

    driver.close()


df = pd.DataFrame(l)
#df.to_csv('jobs.csv', index=False, encoding='utf-8')

