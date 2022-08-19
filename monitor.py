from datetime import datetime, timedelta, timezone
from binance.spot import Spot
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import requests
import os

API_KEY = ''
SECRET_KEY = ''
ADDR = ''
DB_NAME = ''
pull = lambda day: os.system(f"gsutil cp gs://{DB_NAME}/{day}.json data.json")
def graph (symbol, date, time='00:00:00'):
  raw = datetime.fromisoformat(f'{date} {time}.000+00:00').timestamp()
  day = round(floor(raw, '1d').timestamp())
  week = round(floor(raw, '1w').timestamp())
  pull(day)
  data = load('data')
  traces = list(data.values())[-1][symbol]['traces']
  traces = {week: list(traces.values())}
  client = Spot(API_KEY, SECRET_KEY, show_limit_usage=True)
  trades = [(datetime.fromtimestamp(trd['time'] / 1000), float(trd['price']), 'BUY' if trd['isBuyer'] else 'SELL') for trd in client.my_trades(symbol)['data']]
  print(trades)
  df = get_klines(symbol, '15m', day)
  plot(df, traces, trades, '1d')

def stats (date, time='00:00:00'):
  raw = datetime.fromisoformat(f'{date} {time}.000+00:00').timestamp()
  day = round(floor(raw, '1d').timestamp())
  quarter = round(floor(raw, '15m').timestamp())
  pull(day)
  data = load('data')
  stat = -1
  try:
    stat = data[str(quarter)]
  except KeyError:
    print(f"err: specified date can't be found.")
    print(quarter)
    return 
  return stat

def floor(date, interval, delta=None):
  if delta != None:
    delta = delta.total_seconds()
  else:
    delta = 0
  if not isinstance(date, (int, float)):
    date = date.timestamp()

  units = {'w': (604800, 345600), 'd': (86400, 0), 'h': (3600, 0), 'm': (60, 0), 's': (1, 0)}
  freq = int(''.join([i for i in interval if i.isdigit()]))
  unit = ''.join([i for i in interval if i.isalpha()])
  coef = units[unit][0] * freq
  delt = units[unit][1] + delta

  result = (date - delt) - ((date - delt) % coef) + delt
  return datetime.fromtimestamp(int(result))

def load(name):
  with open(name + '.json', 'r') as file:
      var = json.load(file)
      return var

def zulu(delta=timedelta()):
  return datetime.now(tz=timezone.utc) - delta

def get_klines(symbol, interval, start):
  client = Spot(show_limit_usage=True)
  klines = client.klines(symbol, interval, startTime=start * 1000, endTime=(start + 86400) * 1000)
  output = []
  for r in reversed(klines['data']):
    entry = {'time': float(r[0]), 'open': float(r[1]), 'high': float(r[2]), 'low': float(r[3]), 'close': float(r[4]),  'volume': float(r[5])}
    output.append(entry)

  df = pd.DataFrame(output)
  df["time"] = pd.to_datetime(df["time"], unit='ms')

  return df

def trace_match (x, traces):
  if x.timestamp() in traces.keys():
    return traces[x.timestamp()]
  else:
    return [0,0,0,0]

def merge_traces (df, traces):
  floored = (df['time'] - df['time'].dt.weekday.astype('timedelta64[D]')).dt.floor('1D')
  tdf = floored.apply(trace_match, args=[traces])
  tdf = tdf.apply(pd.Series).rename(columns = lambda x : ['T1','T2','T3','T4'][x])
  return pd.concat([df,tdf],axis=1)

def plot (df, traces, trades, interval, even=False, symbol='symbol: unknown'):
  k = df
  k = merge_traces(k, traces)
  start = k.iloc[-1]['time']
  trades = [trd for trd in trades if trd[0] > start]

  traced = k.loc[k['T1']!=0]

  fig = go.Figure(data=[go.Candlestick(x=k['time'], open=k['open'], high=k['high'], low=k['low'], close=k['close'])])
  fig.add_trace(go.Scatter(x=traced['time'], y=traced['T1'] ,line=dict(color='red', dash='dot',width=1)))
  fig.add_trace(go.Scatter(x=traced['time'], y=traced['T2'] ,line=dict(color='red', dash='dot',width=1)))
  fig.add_trace(go.Scatter(x=traced['time'], y=traced['T3'] ,line=dict(color='green', dash='dot',width=1)))
  fig.add_trace(go.Scatter(x=traced['time'], y=traced['T4'] ,line=dict(color='green', dash='dot',width=1)))
  
  buys = [trd for trd in trades if trd[2]=='BUY']
  sells = [trd for trd in trades if trd[2]=='SELL']
  if even:
    bg = "rgba(30,30,30,1)"
  else:
    bg = "rgba(33,33,32,1)"

  fig.add_trace(go.Scatter(x=[trd[0] for trd in buys], y=[trd[1] for trd in buys], mode='markers', marker_size=11, marker_line_width=3, marker_color='red', name='Bought'))
  fig.add_trace(go.Scatter(x=[trd[0] for trd in sells], y=[trd[1] for trd in sells], mode='markers', marker_size=11, marker_line_width=3, marker_color='green', name='Sold'))
  fig.update_layout(title_text=symbol, title_x=0.5, paper_bgcolor=bg, plot_bgcolor='rgba(255,0,0,0)',font_color="rgba(184,184,184,1)")
  fig.update_xaxes(gridcolor='rgba(39,40,34,1)')
  fig.update_yaxes(gridcolor='rgba(39,40,34,1)')

  fig.show()

def plot_weight():
  pass

def get (endpoint):
  return requests.get(ADDR + 'stats').json()

def graphs ():
  raw = datetime.now().timestamp()
  day = round(floor(raw, '1d').timestamp())
  week = round(floor(raw, '1w').timestamp())
  {}
  pull(day)
  data = load('data')
  symbols = list(list(data.values())[-1].keys())
  count = 0
  for symbol in symbols:
    try:
      trace_data = list(data.values())[-1][symbol]['traces']
      traces = {week: list(trace_data.values())}

      client = Spot(API_KEY, SECRET_KEY, show_limit_usage=True)
      trades = [(datetime.fromtimestamp(trd['time'] / 1000), float(trd['price']), 'BUY' if trd['isBuyer'] else 'SELL') for trd in client.my_trades(symbol)['data']]

      length = round(raw - 86400)
      df = get_klines(symbol, '15m', length)

      plot(df, traces, trades, '7d', even=(count%2 == 0), symbol=symbol)
      count+=1
    except Exception as e:
      continue
      print(e)

def worth ():
  prices = {}
  quote_asset = 'USDT'
  client = Spot(API_KEY, SECRET_KEY, show_limit_usage=True)

  for ticker in client.ticker_price():
    prices[ticker['symbol']] = float(ticker['price'])
  prices

  balance = 0
  bnb = 0
  for bal in client.account()['balances']:
    if bal['asset'] not in  [quote_asset]:
      try:
        if bal['asset'] == 'BNB':
          bnb = prices[bal['asset'] + quote_asset] * (float(bal['free']) + float(bal['locked']))
        else:
          balance+= prices[bal['asset'] + quote_asset] * (float(bal['free']) + float(bal['locked']))
      except:
        pass
    else:
      balance += (float(bal['free']) + float(bal['locked']))
  print(f"{round(balance + bnb,2)} ({round(balance,2)} + {round(bnb,2)})")

def portfolio ():
  for k, v in get('stats').items():
    pos = f"{v['position']['full']}\t({v['position']['free']} + {v['position']['locked']})"
    msg = f"{k}:\t{pos}"
    print(msg)

def last_check ():
  check = datetime.fromtimestamp(get('stats')['LINKUSDT']['saved_at'])
  print(f"{round((datetime.utcnow() - check).total_seconds())} seconds ago")
