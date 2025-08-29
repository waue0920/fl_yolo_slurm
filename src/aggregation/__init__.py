from .fedavg import aggregate as fedavg_aggregate
from .fedprox import aggregate as fedprox_aggregate
from .scaffold import aggregate as scaffold_aggregate

AGGREGATORS = {
    "fedavg": fedavg_aggregate,
    "fedprox": fedprox_aggregate,
    "scaffold": scaffold_aggregate,
}
