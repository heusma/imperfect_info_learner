from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import List, Tuple, Callable

import requests

from Games.financial_model.kraken.FinModPrep.stocks.utilities.common_assertions import assert_is_float
from Games.financial_model.kraken.FinModPrep.stocks.utilities.common_entity_functions import day_to_timestamp, \
    load_history

income_statement_mapping: List[Tuple[str, Callable]] = [
    ('revenue', assert_is_float),
    ('costOfRevenue', assert_is_float),
    ('grossProfit', assert_is_float),
    ('grossProfitRatio', assert_is_float),
    ('researchAndDevelopmentExpenses', assert_is_float),
    ('generalAndAdministrativeExpenses', assert_is_float),
    ('sellingAndMarketingExpenses', assert_is_float),
    ('sellingGeneralAndAdministrativeExpenses', assert_is_float),
    ('otherExpenses', assert_is_float),
    ('operatingExpenses', assert_is_float),
    ('costAndExpenses', assert_is_float),
    ('interestIncome', assert_is_float),
    ('interestExpense', assert_is_float),
    ('depreciationAndAmortization', assert_is_float),
    ('ebitda', assert_is_float),
    ('ebitdaratio', assert_is_float),
    ('operatingIncome', assert_is_float),
    ('operatingIncomeRatio', assert_is_float),
    ('totalOtherIncomeExpensesNet', assert_is_float),
    ('incomeBeforeTax', assert_is_float),
    ('incomeBeforeTaxRatio', assert_is_float),
    ('incomeTaxExpense', assert_is_float),
    ('netIncome', assert_is_float),
    ('netIncomeRatio', assert_is_float),
    ('eps', assert_is_float),
    ('weightedAverageShsOut', assert_is_float),
    ('weightedAverageShsOutDil', assert_is_float),
]

balance_sheet_mapping: List[Tuple[str, Callable]] = [
    ('cashAndCashEquivalents', assert_is_float),
    ('shortTermInvestments', assert_is_float),
    ('cashAndShortTermInvestments', assert_is_float),
    ('netReceivables', assert_is_float),
    ('inventory', assert_is_float),
    ('otherCurrentAssets', assert_is_float),
    ('totalCurrentAssets', assert_is_float),
    ('propertyPlantEquipmentNet', assert_is_float),
    ('goodwill', assert_is_float),
    ('intangibleAssets', assert_is_float),
    ('goodwillAndIntangibleAssets', assert_is_float),
    ('longTermInvestments', assert_is_float),
    ('taxAssets', assert_is_float),
    ('otherNonCurrentAssets', assert_is_float),
    ('totalNonCurrentAssets', assert_is_float),
    ('otherAssets', assert_is_float),
    ('totalAssets', assert_is_float),
    ('accountPayables', assert_is_float),
    ('shortTermDebt', assert_is_float),
    ('taxPayables', assert_is_float),
    ('deferredRevenue', assert_is_float),
    ('otherCurrentLiabilities', assert_is_float),
    ('totalCurrentLiabilities', assert_is_float),
    ('longTermDebt', assert_is_float),
    ('deferredRevenueNonCurrent', assert_is_float),
    ('deferredTaxLiabilitiesNonCurrent', assert_is_float),
    ('otherNonCurrentLiabilities', assert_is_float),
    ('totalNonCurrentLiabilities', assert_is_float),
    ('otherLiabilities', assert_is_float),
    ('capitalLeaseObligations', assert_is_float),
    ('totalLiabilities', assert_is_float),
    ('preferredStock', assert_is_float),
    ('commonStock', assert_is_float),
    ('retainedEarnings', assert_is_float),
    ('accumulatedOtherComprehensiveIncomeLoss', assert_is_float),
    ('othertotalStockholdersEquity', assert_is_float),
    ('totalStockholdersEquity', assert_is_float),
    ('totalLiabilitiesAndStockholdersEquity', assert_is_float),
    ('minorityInterest', assert_is_float),
    ('totalEquity', assert_is_float),
    ('totalLiabilitiesAndTotalEquity', assert_is_float),
    ('totalInvestments', assert_is_float),
    ('totalDebt', assert_is_float),
    ('netDebt', assert_is_float),
]

cash_flow_statement_mapping: List[Tuple[str, Callable]] = [
    ('netIncome', assert_is_float),
    ('depreciationAndAmortization', assert_is_float),
    ('deferredIncomeTax', assert_is_float),
    ('stockBasedCompensation', assert_is_float),
    ('changeInWorkingCapital', assert_is_float),
    ('accountsReceivables', assert_is_float),
    ('inventory', assert_is_float),
    ('accountsPayables', assert_is_float),
    ('otherWorkingCapital', assert_is_float),
    ('otherNonCashItems', assert_is_float),
    ('netCashProvidedByOperatingActivities', assert_is_float),
    ('investmentsInPropertyPlantAndEquipment', assert_is_float),
    ('acquisitionsNet', assert_is_float),
    ('purchasesOfInvestments', assert_is_float),
    ('salesMaturitiesOfInvestments', assert_is_float),
    ('otherInvestingActivites', assert_is_float),
    ('netCashUsedForInvestingActivites', assert_is_float),
    ('debtRepayment', assert_is_float),
    ('commonStockIssued', assert_is_float),
    ('commonStockRepurchased', assert_is_float),
    ('dividendsPaid', assert_is_float),
    ('otherFinancingActivites', assert_is_float),
    ('netCashUsedProvidedByFinancingActivities', assert_is_float),
    ('effectOfForexChangesOnCash', assert_is_float),
    ('netChangeInCash', assert_is_float),
    ('cashAtEndOfPeriod', assert_is_float),
    ('cashAtBeginningOfPeriod', assert_is_float),
    ('operatingCashFlow', assert_is_float),
    ('capitalExpenditure', assert_is_float),
    ('freeCashFlow', assert_is_float),
]

key_metric_mapping: List[Tuple[str, Callable]] = [
    ('revenuePerShare', assert_is_float),
    ('netIncomePerShare', assert_is_float),
    ('operatingCashFlowPerShare', assert_is_float),
    ('freeCashFlowPerShare', assert_is_float),
    ('cashPerShare', assert_is_float),
    ('bookValuePerShare', assert_is_float),
    ('tangibleBookValuePerShare', assert_is_float),
    ('shareholdersEquityPerShare', assert_is_float),
    ('interestDebtPerShare', assert_is_float),
    ('marketCap', assert_is_float),
    ('enterpriseValue', assert_is_float),
    ('peRatio', assert_is_float),
    ('priceToSalesRatio', assert_is_float),
    ('pocfratio', assert_is_float),
    ('pfcfRatio', assert_is_float),
    ('pbRatio', assert_is_float),
    ('ptbRatio', assert_is_float),
    ('evToSales', assert_is_float),
    ('enterpriseValueOverEBITDA', assert_is_float),
    ('evToOperatingCashFlow', assert_is_float),
    ('evToFreeCashFlow', assert_is_float),
    ('earningsYield', assert_is_float),
    ('freeCashFlowYield', assert_is_float),
    ('debtToEquity', assert_is_float),
    ('debtToAssets', assert_is_float),
    ('netDebtToEBITDA', assert_is_float),
    ('currentRatio', assert_is_float),
    ('interestCoverage', assert_is_float),
    ('incomeQuality', assert_is_float),
    ('dividendYield', assert_is_float),
    ('payoutRatio', assert_is_float),
    ('salesGeneralAndAdministrativeToRevenue', assert_is_float),
    ('researchAndDdevelopementToRevenue', assert_is_float),
    ('intangiblesToTotalAssets', assert_is_float),
    ('capexToOperatingCashFlow', assert_is_float),
    ('capexToRevenue', assert_is_float),
    ('capexToDepreciation', assert_is_float),
    ('stockBasedCompensationToRevenue', assert_is_float),
    ('grahamNumber', assert_is_float),
    ('roic', assert_is_float),
    ('returnOnTangibleAssets', assert_is_float),
    ('grahamNetNet', assert_is_float),
    ('workingCapital', assert_is_float),
    ('tangibleAssetValue', assert_is_float),
    ('netCurrentAssetValue', assert_is_float),
    ('investedCapital', assert_is_float),
    ('averageReceivables', assert_is_float),
    ('averagePayables', assert_is_float),
    ('averageInventory', assert_is_float),
    ('daysSalesOutstanding', assert_is_float),
    ('daysPayablesOutstanding', assert_is_float),
    ('daysOfInventoryOnHand', assert_is_float),
    ('receivablesTurnover', assert_is_float),
    ('payablesTurnover', assert_is_float),
    ('inventoryTurnover', assert_is_float),
    ('roe', assert_is_float),
    ('capexPerShare', assert_is_float),
]


def pre_processor_income_statement(obj: dict):
    assert isinstance(obj, dict)
    assert 'fillingDate' in obj

    # Das fillingDate ist des späteste Datum. Wir sagen hier,
    # dass unser Model erst dann über das statement informiert wird.
    obj['date'] = day_to_timestamp(obj['fillingDate'])

    return [obj['date']], obj


def pre_processor_balance_sheet(obj: dict):
    assert isinstance(obj, dict)
    assert 'fillingDate' in obj

    # Das fillingDate ist des späteste Datum. Wir sagen hier,
    # dass unser Model erst dann über das statement informiert wird.
    obj['date'] = day_to_timestamp(obj['fillingDate'])

    return [obj['date']], obj


def pre_processor_cash_flow_statement(obj: dict):
    assert isinstance(obj, dict)
    assert 'fillingDate' in obj

    # Das fillingDate ist des späteste Datum. Wir sagen hier,
    # dass unser Model erst dann über das statement informiert wird.
    obj['date'] = day_to_timestamp(obj['fillingDate'])

    return [obj['date']], obj


def pre_processor_key_metric(obj: dict):
    assert isinstance(obj, dict)
    assert 'date' in obj

    obj['date'] = day_to_timestamp(obj['date'])

    return [obj['date']], obj


def request_income_statements(symbol: str):
    assert isinstance(symbol, str)

    response = requests.get(
        'https://financialmodelingprep.com/api/v3/income-statement/{symbol}?period=quarter&limit=9999999&apikey=ae94d52fd1ea5b77c7614381ced3d130'.format(
            symbol=symbol)
    ).json()
    if not response:
        return []
    assert isinstance(response, List)
    assert all(statement['symbol'] == symbol and statement['reportedCurrency'] == 'USD' for statement in response)
    return load_history(response, pre_processor_income_statement, income_statement_mapping)


def request_balance_sheets(symbol: str):
    assert isinstance(symbol, str)

    response = requests.get(
        'https://financialmodelingprep.com/api/v3/balance-sheet-statement/{symbol}?period=quarter&limit=9999999&apikey=ae94d52fd1ea5b77c7614381ced3d130'.format(
            symbol=symbol)
    ).json()
    if not response:
        return []
    assert isinstance(response, List)
    assert all(statement['symbol'] == symbol and statement['reportedCurrency'] == 'USD' for statement in response)
    return load_history(response, pre_processor_balance_sheet, balance_sheet_mapping)


def request_cash_flow_statements(symbol: str):
    assert isinstance(symbol, str)

    response = requests.get(
        'https://financialmodelingprep.com/api/v3/cash-flow-statement/{symbol}?period=quarter&limit=9999999&apikey=ae94d52fd1ea5b77c7614381ced3d130'.format(
            symbol=symbol)
    ).json()
    if not response:
        return []
    assert isinstance(response, List)
    assert all(statement['symbol'] == symbol and statement['reportedCurrency'] == 'USD' for statement in response)
    return load_history(response, pre_processor_cash_flow_statement, cash_flow_statement_mapping)


def request_key_metrics(symbol: str):
    assert isinstance(symbol, str)

    response = requests.get(
        'https://financialmodelingprep.com/api/v3/key-metrics/{symbol}?period=quarter&limit=9999999&apikey=ae94d52fd1ea5b77c7614381ced3d130'.format(
            symbol=symbol)
    ).json()
    if not response:
        return []
    assert isinstance(response, List)
    assert all(statement['symbol'] == symbol for statement in response)
    return load_history(response, pre_processor_key_metric, key_metric_mapping)


def request_by(symbol: str):
    income_statements = request_income_statements(symbol)
    balance_sheets = request_balance_sheets(symbol)
    cash_flow_statements = request_cash_flow_statements(symbol)
    key_metrics = request_key_metrics(symbol)

    result = dict()
    for key in income_statements:
        result[key] = income_statements[key]
    for key in balance_sheets:
        result[key] = balance_sheets[key]
    for key in cash_flow_statements:
        result[key] = cash_flow_statements[key]
    for key in key_metrics:
        result[key] = key_metrics[key]
    return result
