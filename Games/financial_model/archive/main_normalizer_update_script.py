from Games.financial_model.archive.Archive import Archive
from Games.financial_model.archive.Normalizer import Normalizer

archive = Archive('../Games/archive.json')
normalizer = Normalizer('../Games/norm.json')

normalizer.build(archive)

# This is just a test and never saved
normalizer.apply(archive)

normalizer.save()