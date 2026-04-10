from enum import Enum

from algorithms.alg import *


class DatasetsEnum(str, Enum):
    amazon23office = "amazon23office"
    zenodo = "zenodo"
    deezermarch = "deezermarch"

class AlgorithmsEnum(Enum):
    avgmatching = AverageQueryMatching
    crossmatching = CrossAttentionQueryMatching
    sparsematching = SparseMoEQueryMatching
    talkrec = TalkingToYourRecSys
    twotower = TwoTowerModel
    pop = PopItems
    random = RandomItems
