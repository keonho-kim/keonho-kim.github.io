import os
import re
import numpy as np
import pandas as pd

import FinanceDataReader as fdr
from pykrx import stock

import plotly.express as px

# 파일 경로 설정
path = os.getcwd()
clas_path = path +'\Classification'
img_path = path +'\images'

# 날짜 설정
print("입력예시: 2020-01-01")
previous = input("이전 거래일을 입력하세요: ").replace('-', '')
today = input("오늘 날짜를 입력하세요: ").replace('-', '')

# 표준산업분류 불러오기
classification = pd.read_csv(clas_path +'\classification.csv', usecols=['L1','L2','L3'])

# 코스피 종목 정보 불러오기
kospi_list = fdr.StockListing('KOSPI')
    # 우선주, 투자신탁 제거
kospi_list = kospi_list.dropna(axis=0).reset_index(drop=True)
    # 코스피 종목 정보 추리기
kospi_info= pd.DataFrame(kospi_list, columns = ['Symbol', 'Name', 'Sector'])

# 상위 산업 추가하기

# 띄워쓰기가 다른 경우가 있음 -> 띄워쓰기 전부 제거
# 특수 문자 모두 제거

for idx, row in classification.iterrows():
    no_space = row['L3'].replace(' ', '')
    row['L3'] = no_space

    no_specials = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', row['L3'])
    row['L3'] = no_specials

# 코스피 상위 산업 추가하기

kospi_info['L2'] = None
kospi_info['L1'] = None

for idx, row in kospi_info.iterrows():
    
    sector = row['Sector'].replace(' ', '') # 코스피의 섹터도 동일하게 띄워쓰기 제거
    sector = re.sub('[-=+,#/·\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', sector) # 모든 특수문자 제거

    L1 = list(classification[classification['L3'] == sector]['L1'])
    L2 = list(classification[classification['L3']== sector]['L2'])
    
    if len(L1) > 0:
        kospi_info['L1'][idx] = L1[0]
        kospi_info['L2'][idx] = L2[0]
    else:
        print(sector)

# 컬럼명 바꾸기

kospi_info = kospi_info.rename(columns={'Symbol':'Ticker', 'Sector': 'L3'})

kospi_ohlcv_pre = stock.get_market_ohlcv_by_ticker(previous, "KOSPI").reset_index(drop=False)
kospi_ohlcv_pre = kospi_ohlcv_pre.rename(
    columns = {
        '종목코드':'Ticker',
        '종목명':'Name',
        '시가': 'Open',
        '고가': 'High',
        '저가': 'Low',
        '종가': 'Close',
        '거래량': 'Volume',
        '거래대금': 'Transaction Volume',
        '시가총액': 'Market Cap',
        '시총비중': 'Weight',
        '상장주식수': 'Share Outstanding'
         }
    )

kospi_ohlcv_today = stock.get_market_ohlcv_by_ticker(today, "KOSPI").reset_index(drop=False)
kospi_ohlcv_today = kospi_ohlcv_today.rename(
    columns = {
        '종목코드':'Ticker',
        '종목명':'Name',
        '시가': 'Open',
        '고가': 'High',
        '저가': 'Low',
        '종가': 'Close',
        '거래량': 'Volume',
        '거래대금': 'Transaction Volume',
        '시가총액': 'Market Cap',
        '시총비중': 'Weight',
        '상장주식수': 'Share Outstanding'
         }
    )

# 코스피 데이터 병합
kospi_info['Open'] = None
kospi_info['Close'] = None
kospi_info['Pr_Change'] = None
kospi_info['Change'] = None
kospi_info['MarCap'] = None
kospi_info['sqrtMarCap'] = None
kospi_info['Status'] = None
for idx, row in kospi_info.iterrows():
    ticker = row['Ticker']
    
    stock_ohlcv_today = kospi_ohlcv_today[kospi_ohlcv_today['Ticker'] == ticker]

    if stock_ohlcv_today['Open'].iloc[0] != 0: # 거래정지가 아닌 경우
        row['Open'] = stock_ohlcv_today['Open'].iloc[0]
        row['Close'] = stock_ohlcv_today['Close'].iloc[0]
        
        # 전일 대비 가격 변동
        pre_close = kospi_ohlcv_pre[kospi_ohlcv_pre['Ticker'] == ticker]['Close'].iloc[0]
        
        ch = row['Close'] - pre_close
        row['Pr_Change'] = ch
        pch = round((row['Close'] - pre_close) / pre_close * 100, 2)
        row['Change'] = pch

        mcap = str(stock_ohlcv_today['Market Cap'].iloc[0])[:-8]
        row['MarCap'] = int(mcap)
        row['sqrtMarCap'] = np.sqrt(int(mcap))
        row['Status'] = 'Active'
    else: # 거래정지
        row['Open'] = stock_ohlcv_today['Close'].iloc[0]
        row['Close'] = stock_ohlcv_today['Close'].iloc[0]
        row['Change'] = 0
        mcap = str(stock_ohlcv_today['Market Cap'].iloc[0])[:-8]
        row['MarCap'] = int(mcap)
        row['sqrtMarCap'] = np.sqrt(int(mcap))
        row['Status'] = 'Suspend'

kospi_info['Market'] = 'KOSPI'
kospi_info['Open'] = kospi_info['Open'].astype(float)
kospi_info['Close'] = kospi_info['Close'].astype(float)
kospi_info['Change'] = kospi_info['Change'].astype(float)
kospi_info['MarCap'] = kospi_info['MarCap'].astype(float)
kospi_info['sqrtMarCap'] = kospi_info['sqrtMarCap'].astype(float)

fig = px.treemap(
    kospi_info,
    path = ['Market', 'L1', 'L2', 'Name'],
    values = 'sqrtMarCap',
    color = 'Change',
    color_continuous_scale= [[0, '#14029e'], [0.5, '#424242'], [1, '#8c0618']],
    color_continuous_midpoint = 0,
    range_color = [-3,3],
    branchvalues = 'total',
    custom_data = ['Change'],
    maxdepth=5
)

fig.update_traces(
    textposition = 'middle center',
    marker_line_width= 0.2,
    hovertemplate = '<b>%{label}</b><br><br>전일대비 증감율: %{color:.2f}%',
    texttemplate = '<b>%{label}</b><br><br>%{customdata[0]:.2f}%'
    )

fig.update_layout(
    autosize = False,
    width = 1280,
    height = 720,
    margin = dict(l=0, r=0, t=0, b=0),
    coloraxis_showscale = False
)


fig.write_html(img_path+'\\'+today+'.html')

#전일 파일 삭제
os.remove(img_path + '\\' + previous + '.html')

print("Process Complete")