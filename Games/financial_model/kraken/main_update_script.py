import datetime

import requests

from Games.financial_model.archive.Archive import Archive
from Games.financial_model.kraken.FinModPrep.stocks.datatypes.entities import Price, MarketCapitalization, Dividend, \
    FinancialStatement

"""Config"""
max_symbols = 10
allowed_exchanges = ["NASDAQ"]
"""-"""

archive = Archive('../Games/archive.json')

symbols = requests.get(
    'https://financialmodelingprep.com/api/v3/stock/list?apikey=ae94d52fd1ea5b77c7614381ced3d130'
).json()

for symbol in symbols:
    if symbol["exchangeShortName"] not in allowed_exchanges:
        continue
    if symbol["type"] != "stock":
        continue

    symbol = symbol['symbol']

    try:
        price_timelines = Price.request_by(symbol)

        market_caps = MarketCapitalization.request_by(symbol)

        dividend_timelines = Dividend.request_by(symbol)

        fin_statement_timelines = FinancialStatement.request_by(symbol)

    except AssertionError:
        print(symbol + " failed")
        continue

    path = [symbol, 'price']
    for key in price_timelines:
        archive[path + [key]] = price_timelines[key]

    path = [symbol, 'market_cap']
    for key in market_caps:
        archive[path + [key]] = market_caps[key]

    path = [symbol, 'dividend']
    for key in dividend_timelines:
        archive[path + [key]] = dividend_timelines[key]

    path = [symbol, 'financial_statement']
    for key in fin_statement_timelines:
        archive[path + [key]] = fin_statement_timelines[key]

    print(symbol + " succeeded")

    if len(archive.dict) >= max_symbols:
        break

archive.save()
