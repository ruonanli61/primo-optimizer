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

# Installed libs
import pytest

# User-defined libs
from primo.utils.domain_validators import InRange, validate_mobilization_cost


def test_inrange():
    """Tests InRange domain validator"""
    val = 5.0
    assert val == InRange(0.0, 10.0)(val)

    val = -5.0
    with pytest.raises(
        ValueError,
        match=("Value -5.0 lies outside the admissible range \\[0.0, 10.0\\]"),
    ):
        InRange(0.0, 10.0)(val)


def test_validate_mobilization_cost():
    """Tests the validate_mobilization_cost function"""
    data = {1: 100, 2: 200, 3: 300}
    result = validate_mobilization_cost(data)
    assert result == {1: 100.0, 2: 200.0, 3: 300.0}

    data[5] = 500
    with pytest.raises(
        KeyError, match="Mobilization cost for 4 wells is not provided."
    ):
        validate_mobilization_cost(data)
