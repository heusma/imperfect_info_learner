from Games.financial_model.archive.Archive import Archive
from Games.financial_model.archive.Normalizer import Normalizer

archive = Archive('../../../StockWorldData/archive.json')
normalizer = Normalizer('../../../StockWorldData/norm.json')

normalizer.build(archive)

# This is just a test and never saved
normalizer.apply(archive)

normalizer.save()