# coding: utf-8
import re
import os
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import time

try:
    import FinanceDataReader as fdr
    from pykrx import stock
except:
    os.system('pip install finance-datareader')
    os.system('pip install pykrx')
    import FinanceDataReader as fdr
    from pykrx import stock

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
except:
    os.system('pip install plotly')
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    
# turn of pandas error
pd.set_option('mode.chained_assignment',  None)
pd.options.display.float_format = '{:.2f}'.format

# set file path
path = os.getcwd()
clas_path = path +'/Classification/'
img_path = path +'/images/'

# set date
print("input example: 2020-01-01")
previous = str(input("Enter Previous Trasnaction Date: ").replace('-', ''))
today = str(input("Enter Recent Transaction Date: ").replace('-', ''))

# 표준산업분류 불러오기
classification = pd.read_csv(clas_path +'/classification.csv', usecols=['L1','L2','L3'])

print("------------------")
print("Collecting Data")
print("------------------")

kospi_list = fdr.StockListing('KOSPI')
kospi_list = kospi_list.dropna(axis=0).reset_index(drop=True)
kospi_info= pd.DataFrame(kospi_list, columns = ['Symbol', 'Name', 'Sector'])



for idx, row in classification.iterrows():
    no_space = row['L3'].replace(' ', '')
    row['L3'] = no_space

    no_specials = re.sub('[-=+,#/\?:^$.@*\"~&%ㆍ!\\‘|\(\)\[\]\<\>`\'…]', '', row['L3'])
    row['L3'] = no_specials


kospi_info['L2'] = None
kospi_info['L1'] = None

for idx, row in kospi_info.iterrows():
    # eliminate specials 
    sector = row['Sector'].replace(' ', '') 
    sector = re.sub('[-=+,#/·\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', sector) 

    L1 = list(classification[classification['L3'] == sector]['L1'])
    L2 = list(classification[classification['L3']== sector]['L2'])
    
    if len(L1) > 0:
        kospi_info['L1'][idx] = L1[0]
        kospi_info['L2'][idx] = L2[0]
    else:
        print(sector)

# change column names

kospi_info = kospi_info.rename(columns={'Symbol':'Ticker', 'Sector': 'L3'})

kospi_ohlcv_pre = stock.get_market_ohlcv_by_ticker(previous, "KOSPI").reset_index(drop=False)
kospi_ohlcv_pre = kospi_ohlcv_pre.rename(
    columns = {
        '티커':'Ticker',
        '종목명':'Name',
        '시가': 'Open',
        '고가': 'High',
        '저가': 'Low',
        '종가': 'Close',
        '거래량': 'Volume',
        '거래대금': 'Transaction Volume'
         }
    )

kospi_ohlcv_today = stock.get_market_ohlcv_by_ticker(today, "KOSPI").reset_index(drop=False)
kospi_ohlcv_today = kospi_ohlcv_today.rename(
    columns = {
        '티커':'Ticker',
        '종목명':'Name',
        '시가': 'Open',
        '고가': 'High',
        '저가': 'Low',
        '종가': 'Close',
        '거래량': 'Volume',
        '거래대금': 'Transaction Volume'
         }
    )


market_cap = stock.get_market_cap_by_ticker(today, "KOSPI").reset_index(drop=False)

market_cap = market_cap.rename(
    columns = {
        '티커':'Ticker',
        '종가': 'Close',
        '거래량': 'Volume',
        '거래대금': 'Transaction Volume',
        '시가총액': 'Market Cap',
        '상장주식수': 'Share Outstanding'
         }
    )

print("------------------")
print("Processing KOSPI Data")
print("------------------")

# incoporate kospi data
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
    mcap = market_cap[market_cap["Ticker"] == ticker]['Market Cap'].iloc[0]
    
    try: # if a compnay is not a new listed

        if stock_ohlcv_today['Open'].iloc[0] != 0: # if not transaction stopped case
            row['Open'] = stock_ohlcv_today['Open'].iloc[0]
            row['Close'] = stock_ohlcv_today['Close'].iloc[0]
            
			# get daily return
            pre_close = kospi_ohlcv_pre[kospi_ohlcv_pre['Ticker'] == ticker]['Close'].iloc[0]
            
            ch = row['Close'] - pre_close
            row['Pr_Change'] = ch
            pch = round((row['Close'] - pre_close) / pre_close * 100, 2)
            row['Change'] = pch

            
            row['MarCap'] = int(mcap)
            row['sqrtMarCap'] = np.sqrt(int(mcap))
            row['Status'] = 'Active'
        else: # transaction stopped
            row['Open'] = stock_ohlcv_today['Close'].iloc[0]
            row['Close'] = stock_ohlcv_today['Close'].iloc[0]
            row['Change'] = 0
            
            row['MarCap'] = int(mcap)
            row['sqrtMarCap'] = np.sqrt(int(mcap))
            row['Status'] = 'Suspend'

    except: # newly listed company is not included 
        pass 


kospi_info['Market'] = 'KOSPI'
kospi_info['Open'] = kospi_info['Open'].astype(float)
kospi_info['Close'] = kospi_info['Close'].astype(float)
kospi_info['Change'] = kospi_info['Change'].astype(float)
kospi_info['MarCap'] = kospi_info['MarCap'].astype(float)
kospi_info['sqrtMarCap'] = kospi_info['sqrtMarCap'].astype(float)
kospi_info = kospi_info.dropna()

print("------------------")
print("Start Creating KOSPI Map")
print("------------------")


fig = px.treemap(
    kospi_info,
    path = ['Market', 'L1', 'L2', 'Name'],
    values = 'sqrtMarCap',
    color = 'Change',
    color_continuous_scale= [[0, '#14029e'], [0.5, '#424242'], [1, '#8c0618']],
    color_continuous_midpoint = 0,
    range_color = [-3,3],
    branchvalues = 'total',
    custom_data = ['Change', 'Close', 'Pr_Change'],
    maxdepth=5
)

fig.update_traces(
    textposition = 'middle center',
    marker_line_width= 0.2,
    hovertemplate = '<b>%{label}</b>',
    texttemplate = '%{label}<br><br>%{customdata[0]:.2f}%'
    )

fig.update_layout(
    autosize = False,
    width = 1080,
    height = 640,
    margin = dict(l=0, r=0, t=0, b=0),
    coloraxis_showscale = False
)

# save file
try:
    os.remove(img_path + "Map_KOSPI_" + previous  + '.html')
except:
    pass
fig.write_html(img_path + "Map_KOSPI_" + today  + ".html")


# get kosdaq list
kosdaq_list = fdr.StockListing('KOSDAQ')
# remove preffered and irrelavant equities
kosdaq_list = kosdaq_list.dropna(axis=0).reset_index(drop=True)
kosdaq_info= pd.DataFrame(kosdaq_list, columns = ['Symbol', 'Name', 'Sector'])



print("------------------")
print("Processing KOSDAQ Data")
print("------------------")


kosdaq_info['L2'] = None
kosdaq_info['L1'] = None

for idx, row in kosdaq_info.iterrows():
    
    sector = row['Sector'].replace(' ', '') # 코스피의 섹터도 동일하게 띄워쓰기 제거
    sector = re.sub('[-=+,#/·\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', sector) # 모든 특수문자 제거

    L1 = list(classification[classification['L3'] == sector]['L1'])
    L2 = list(classification[classification['L3']== sector]['L2'])
    
    if len(L1) > 0:
        kosdaq_info['L1'][idx] = L1[0]
        kosdaq_info['L2'][idx] = L2[0]
    else:
        print(sector)

kosdaq_info = kosdaq_info.rename(columns={'Symbol':'Ticker', 'Sector': 'L3'})

kosdaq_ohlcv_pre = stock.get_market_ohlcv_by_ticker(previous, "KOSDAQ").reset_index(drop=False)
kosdaq_ohlcv_pre = kosdaq_ohlcv_pre.rename(
    columns = {
        '티커':'Ticker',
        '종목명':'Name',
        '시가': 'Open',
        '고가': 'High',
        '저가': 'Low',
        '종가': 'Close',
        '거래량': 'Volume',
        '거래대금': 'Transaction Volume'
         }
    )

kosdaq_ohlcv_today = stock.get_market_ohlcv_by_ticker(today, "KOSDAQ").reset_index(drop=False)
kosdaq_ohlcv_today = kosdaq_ohlcv_today.rename(
    columns = {
        '티커':'Ticker',
        '종목명':'Name',
        '시가': 'Open',
        '고가': 'High',
        '저가': 'Low',
        '종가': 'Close',
        '거래량': 'Volume',
        '거래대금': 'Transaction Volume'
         }
    )


market_cap_pre = stock.get_market_cap_by_ticker(previous, "KOSDAQ").reset_index(drop=False)

market_cap_pre = market_cap_pre.rename(
    columns = {
        '티커':'Ticker',
        '종가': 'Close',
        '거래량': 'Volume',
        '거래대금': 'Transaction Volume',
        '시가총액': 'Market Cap',
        '상장주식수': 'Share Outstanding'
         }
    )


market_cap_today = stock.get_market_cap_by_ticker(today, "KOSDAQ").reset_index(drop=False)

market_cap_today = market_cap_today.rename(
    columns = {
        '티커':'Ticker',
        '종가': 'Close',
        '거래량': 'Volume',
        '거래대금': 'Transaction Volume',
        '시가총액': 'Market Cap',
        '상장주식수': 'Share Outstanding'
         }
    )


kosdaq_info['Open'] = None
kosdaq_info['Close'] = None
kosdaq_info['Pr_Change'] = None
kosdaq_info['Change'] = None
kosdaq_info['MarCap_pre'] = None
kosdaq_info['MarCap_today'] = None
kosdaq_info['sqrtMarCap'] = None
kosdaq_info['Status'] = None

for idx, row in kosdaq_info.iterrows():
    ticker = row['Ticker']
    
    stock_ohlcv_today = kosdaq_ohlcv_today[kosdaq_ohlcv_today['Ticker'] == ticker]

    try: 
        mcap_td = market_cap_today[market_cap_today["Ticker"] == ticker]['Market Cap'].iloc[0]  
        row['MarCap_today'] = int(mcap_td)

        mcap_pr = market_cap_pre[market_cap_pre["Ticker"] == ticker]['Market Cap'].iloc[0] 
        row['MarCap_pre'] = int(mcap_pr)

        if stock_ohlcv_today['Open'].iloc[0] != 0: 
            row['Open'] = stock_ohlcv_today['Open'].iloc[0]
            row['Close'] = stock_ohlcv_today['Close'].iloc[0]
            
            pre_close = kosdaq_ohlcv_pre[kosdaq_ohlcv_pre['Ticker'] == ticker]['Close'].iloc[0]
            ch = row['Close'] - pre_close
            row['Pr_Change'] = ch
            pch = round((row['Close'] - pre_close) / pre_close * 100, 2)
            row['Change'] = pch
            
            row['sqrtMarCap'] = np.sqrt(int(mcap_td))
            row['Status'] = 'Active'

        else: 
            row['Open'] = stock_ohlcv_today['Close'].iloc[0]
            row['Close'] = stock_ohlcv_today['Close'].iloc[0]
            row['Change'] = 0
            
            row['sqrtMarCap'] = np.sqrt(int(mcap_td))
            row['Status'] = 'Suspend'
    except: 
        pass 

kosdaq_info['Market'] = 'KOSDAQ'
kosdaq_info['Open'] = kosdaq_info['Open'].astype(float)
kosdaq_info['Close'] = kosdaq_info['Close'].astype(float)
kosdaq_info['Change'] = kosdaq_info['Change'].astype(float)
kosdaq_info['MarCap_pre'] = kosdaq_info['MarCap_pre'].astype(float)
kosdaq_info['MarCap_today'] = kosdaq_info['MarCap_today'].astype(float)
kosdaq_info['sqrtMarCap'] = kosdaq_info['sqrtMarCap'].astype(float)
kosdaq_info = kosdaq_info.dropna()

print("------------------")
print("Start Creating KOSDAQ Map")
print("------------------")

fig = px.treemap(
    kosdaq_info,
    path = ['Market','L1', 'L2', 'Name'],
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
    hovertemplate = '<b>%{label}</b>',
    texttemplate = '%{label}<br><br>%{customdata[0]:.2f}%'
    )

fig.update_layout(
    autosize = False,
    width = 1080,
    height = 640,
    margin = dict(l=0, r=0, t=0, b=0),
    coloraxis_showscale = False
)

try:
    os.remove(img_path + "Map_KOSDAQ_" + previous + '.html')
except:
    pass
fig.write_html(img_path + "Map_KOSDAQ_" + today + ".html")


print("------------------")
print("Complete Creating Map")
print("------------------")

"""
# Create Dict.
tiger_dict = {'277630': 'Tiger 코스피',
 '102110': 'Tiger 200',
 '139260': 'Tiger 200 IT',
 '139220': 'Tiger 200 건설',
 '139290': 'Tiger 200 경기소비재',
 '227550': 'Tiger 200 산업재',
 '157490': 'Tiger 소프트웨어',
 '315270': 'Tiger 200 커뮤니케이션서비스',
 '091230': 'Tiger 반도체',
 '139250': 'Tiger 200 에너지화학',
 '091220': 'Tiger 은행',
 '139270': 'Tiger 200 금융',
 '157500': 'Tiger 증권',
 '139240': 'Tiger 200 철강소재',
 '139230': 'Tiger 200 중공업',
 '227560': 'Tiger 200 생활소비재'}

tiger = [val for key, val in tiger_dict.items()]

# 3개월 전 날짜 구하기
previous = pd.to_datetime(today)
previous = previous + timedelta(weeks=-12)
previous = str(previous)[0:10].replace('-', '')

etf = pd.DataFrame(columns = tiger)

# 각 인덱스 가격 구하기
for key, val in tiger_dict.items():
    ticker = key
    p = stock.get_etf_ohlcv_by_date(fromdate=previous, todate=today, ticker=key)
    etf[val] = p['종가']
    

# Gap 구하기

idx = list(tiger[:2])
sectors = list(tiger[2:])

idx_mean = etf[idx].mean(axis=1)
idx_return = (idx_mean / idx_mean.iloc[0] - 1.0) * 100

sector = etf[sectors]
sector_returns = (sector / sector.iloc[0] - 1.0) * 100

gap = sector_returns
gap['시장수익률'] = idx_return

# 인덱스 날짜별로 설정
gap = gap.reset_index()
gap['날짜'] = pd.to_datetime(gap['날짜'])

print("------------------")
print("Start Creating Line Chart")
print("------------------")

# 선 그래프

fig = make_subplots(
    rows = 14,
    cols = 1,
    vertical_spacing = 0.015,
    y_title = '수익률',
    subplot_titles =(sectors)
)

r = 1
for i in range(len(sectors)):
    sec = sectors[i]
    cols = [sec].extend(['날짜', 'MA5', 'MA20', '시장수익률'])
    df = pd.DataFrame(columns = cols)
    
    df[sec] = None
    df[sec] = gap[sec]
    
    df['날짜'] = None
    df['날짜'] = gap['날짜']
    df['날짜'] = pd.to_datetime(df['날짜'])
    df = df.set_index(df['날짜'])
    
    df['MA5'] = None
    df['MA5'] = df[sec].rolling(window=5).mean()
    df['MA20'] = None
    df['MA20'] = df[sec].rolling(window=20).mean()
    
    df['시장수익률'] = None
    df['시장수익률'] = idx_return

    df = df.dropna()

    fig.add_trace(
        go.Scatter(
            x = df['날짜'],
            y = df['시장수익률'],
            name = f'시장수익률',
            marker_color = 'Red'
        ),
        row = r,
        col = 1)
    
    fig.add_trace(
        go.Scatter(
            x = df['날짜'],
            y = df[sec],
            name = f'섹터 수익률',
            marker_color = 'white'
        ),
        row = r,
        col = 1)
    
    fig.add_trace(
        go.Scatter(
            x = df['날짜'],
            y = df['MA5'],
            name = f'MA5',
            marker_color = 'Blue'
        ),
        row = r,
        col = 1)
    
    fig.add_trace(
        go.Scatter(
            x = df['날짜'],
            y = df['MA20'],
            name = f'MA20',
            marker_color = 'Green'
        ),
        row = r,
        col = 1)
   
    r += 1
fig.update_yaxes(dtick=5)
fig.update_xaxes(showgrid=False)

fig.update_layout(
    title = {
        'text':"3개월 시장수익률, 섹터수익률, 섹터 이동평균(5, 20)",
        'xanchor':'center',
        'yanchor':'top',
        'x':0.5,
        'y':0.995
    },
    width = 1366, 
    height = 6000, 
    hovermode = 'x unified',
    showlegend=False, 
    template = 'plotly_dark')

# 파일 저장
fig.write_html(img_path + "Line_" + today +".html")


print("------------------")
print("Complete Creating Line Chart")
print("------------------")


print("------------------")
print("Complete Creating Bar Chart")
print("------------------")

# 막대 그래프
# 데이터 프레임
result = pd.DataFrame(gap.iloc[-1, 1:])
result = result - result.iloc[-1]
result = result.iloc[0:-1]
result.columns = ['상대수익률']
result = result.sort_values(by=['상대수익률'])

# 색깔 

result['type'] = None

for idx, row in result.iterrows():
    if row['상대수익률'] < 0:
        row['type'] = 'Underperform'
    else:
        row['type'] = 'Outperform'

# 플롯

fig = px.bar(
    result, x = result.index, 
    y = '상대수익률',
    color='type',
    template='plotly_dark'
    )

fig.update_traces(
    marker_line_width= 0.2,
    hovertemplate = '<b>%{label}</b><br><br><b>상대수익률: %{y:.2f}%</b>',
    hoverlabel = dict(font=(dict(color='white')))
    )

fig.update_traces(
    texttemplate='%{y:.2f}', 
    textposition='outside', 
    textfont_size=11)

fig.add_shape(
    type='line',
    x0=-0.5,
    x1=13.5,
    y0=0,
    y1=0,
    line=dict(color='White',dash='dot',),
    )

fig.update_xaxes(
    tickangle = 70,
    tickfont = dict(size=12)
)

fig.update_layout(
    title = {
        'text':"코스피 & 코스피200 평균 대비 섹터 퍼포먼스 (3개월)",
        'xanchor':'center',
        'yanchor':'top',
        'x':0.5,
    },
    xaxis = {
        'title' : ''
    },
    legend = {
        'title' : 'Type'
    },
    autosize = True,
    width = 1366,
    height = 728
)


# 파일 저장
fig.write_html(img_path + "Bar_" + today + ".html")



print("------------------")
print("Complete Creating Bar Chart")
print("------------------")
"""
