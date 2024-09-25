#################################################################################
# PRIMO - The P&A Project Optimizer was produced under the Methane Emissions
# Reduction Program (MERP) and National Energy Technology Laboratory's (NETL)
# National Emissions Reduction Initiative (NEMRI).
#
# NOTICE. This Software was developed under funding from the U.S. Government
# and the U.S. Government consequently retains certain rights. As such, the
# U.S. Government has been granted for itself and others acting on its behalf
# a paid-up, nonexclusive, irrevocable, worldwide license in the Software to
# reproduce, distribute copies to the public, prepare derivative works, and
# perform publicly and display publicly, and to permit others to do so.
#################################################################################

"""
This module contains functions that can be used for domain validation
with the `ConfigDict()` data structure from Pyomo.
"""
# Standard libs
from typing import Dict

# Installed libs
from pyomo.common.config import NonNegativeFloat

# User-defined libs
from primo.utils.opt_utils import in_bounds


# pylint: disable-next = invalid-name
def InRange(lb, ub):
    """
    Domain validator for 1D compact sets.

    Parameters
    ----------
    lb : float
        Lower bound

    ub : float
        Upper bound

    Returns
    -------
    _in_range :
        Pointer to a domain validator function
    """

    def _in_range(val):
        if in_bounds(val, lb, ub, 0.0):
            return val
        raise ValueError(f"Value {val} lies outside the admissible range [{lb}, {ub}]")

    return _in_range


def validate_mobilization_cost(data: Dict[int, float]):
    """
    Validates the mobilization cost data

    Parameters
    ----------
    data : dict
        key => number of wells, value => cost of plugging

    Returns
    -------
    data
        Validated mobilization cost data.
    """
    for k in range(1, len(data) + 1):
        if k not in data:
            raise KeyError(f"Mobilization cost for {k} wells is not provided.")
    for key, val in data.items():
        data[key] = NonNegativeFloat(val)
    return data
