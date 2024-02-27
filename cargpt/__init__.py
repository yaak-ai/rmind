import sys

from beartype import BeartypeConf, BeartypeStrategy
from beartype.claw import beartype_this_package

# NOTE: contrary to the docs beartype does not seem to repsect PYTHONOPTIMIZE - make it do so
# https://beartype.readthedocs.io/en/latest/api_decor/#beartype.BeartypeStrategy.O0
beartype_this_package(
    conf=BeartypeConf(
        strategy=BeartypeStrategy.O0 if sys.flags.optimize > 0 else BeartypeStrategy.O1,
        is_color=False,
    )
)
