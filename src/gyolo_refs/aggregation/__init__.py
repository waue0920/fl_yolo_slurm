
from .fedavg import aggregate as fedavg_aggregate
from .fedprox import aggregate as fedprox_aggregate
from .scaffold import aggregate as scaffold_aggregate
from .fedavgm import aggregate as fedavgm_aggregate
from .fednova import aggregate as fednova_aggregate
from .fedawa import aggregate as fedawa_aggregate
from .fedopt import aggregate as fedopt_aggregate

AGGREGATORS = {
    "fedavg": fedavg_aggregate,
    "fedprox": fedprox_aggregate,
    "scaffold": scaffold_aggregate,
    "fedavgm": fedavgm_aggregate,
    "fednova": fednova_aggregate,
    "fedawa": fedawa_aggregate,
    "fedopt": fedopt_aggregate,
}
