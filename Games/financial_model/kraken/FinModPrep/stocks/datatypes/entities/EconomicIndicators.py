from datetime import datetime
from typing import List, Callable, Tuple

import requests

from Games.financial_model.kraken.FinModPrep.stocks.utilities.common_assertions import assert_float_greater_zero, \
    assert_is_int, assert_is_float
from Games.financial_model.kraken.FinModPrep.stocks.utilities.common_entity_functions import day_to_timestamp, \
    load_history

mapping: List[Tuple[str, Callable]] = [
    ('value', assert_is_float),
]


# Erlärung zum timestamp:
# timestamp: Gespeichert werden Preisaggregate, haben also eigentlich einen Start- und Endzeitpunkt.
#            Wir speichern nur den Endzeitpunkt als timestamp.
def pre_processor(obj: dict):
    obj['date'] = day_to_timestamp(obj['date'])
    return [obj['date']], obj


list_of_indicators = [
    'GDP',
    'realGDP',
    'nominalPotentialGDP',
    'realGDPPerCapita',
    'federalFunds',
    'CPI',
    'inflationRate',
    'inflation',
    'retailSales',
    'consumerSentiment',
    'durableGoods',
    'unemploymentRate',
    'totalNonfarmPayroll',
    'initialClaims',
    'industrialProductionTotalIndex',
    'newPrivatelyOwnedHousingUnitsStartedTotalUnits',
    'totalVehicleSales',
    'retailMoneyFunds',
    'smoothedUSRecessionProbabilities',
    '3MonthOr90DayRatesAndYieldsCertificatesOfDeposit',
    'commercialBankInterestRateOnCreditCardPlansAllAccounts',
    '30YearFixedRateMortgageAverage',
    '15YearFixedRateMortgageAverage',
]


def request():
    today = datetime.now().strftime('%Y-%m-%d')
    indicator_timeline_dict = dict()
    for indicator_name in list_of_indicators:
        response = requests.get(
            'https://financialmodelingprep.com/api/v4/economic?name={indicator}&from=1900-01-01&to={current_date}&apikey=ae94d52fd1ea5b77c7614381ced3d130'.format(
                indicator=indicator_name, current_date=today)
        ).json()
        if not response:
            return []
        assert isinstance(response, List)
        indicator_timeline_dict[indicator_name] = load_history(response, pre_processor, mapping)

    return indicator_timeline_dict
