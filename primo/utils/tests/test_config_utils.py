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

# Standard libs
import os
import pathlib

# Installed libs
import ipywidgets as widgets
import numpy as np
import playwright
import pytest

# User-defined libs
from primo.data_parser import (
    EfficiencyMetrics,
    ImpactMetrics,
    WellData,
    WellDataColumnNames,
)
from primo.opt_model.model_options import OptModelInputs
from primo.opt_model.tests.test_model_options import get_column_names_fixture
from primo.utils.config_utils import (
    AddWidgetReturn,
    OverrideSelections,
    SelectWidgetAdd,
    SubSelectWidget,
    UserSelection,
    WidgetReturn,
    _get_checkbox_params,
    copy_dict,
    copy_values,
    is_valid,
    read_config,
    read_defaults,
    update_defaults,
)


@pytest.mark.parametrize(
    "input_dict,output_dict,result,status_ok",
    [
        # Existing test cases
        (
            {
                "input_str": "orange",
                "input_float": 42.0,
                "input_dict": {"nested_str": "banana", "nested_int": 42},
            },
            {
                "input_str": "apple",
                "input_float": 99.0,
                "input_dict": {"nested_str": "banana", "nested_int": 83},
                "input_default": "default",
            },
            {
                "input_str": "orange",
                "input_float": 42.0,
                "input_dict": {"nested_str": "banana", "nested_int": 42},
                "input_default": "default",
            },
            True,
        ),
        ({}, {}, {}, True),
        (
            {"unknown_input": 33, "known_input": "OK"},
            {"known_input": "AOK"},
            {},
            False,
        ),
        # New test cases
        (
            {"input_list": [1, 2, 3]},
            {"input_list": [4, 5, 6]},
            {"input_list": [1, 2, 3]},
            True,
        ),
        (
            {"input_str": "orange", "input_float": 42.0, "extra_key": "unexpected"},
            {"input_str": "apple", "input_float": 99.0},
            {},
            False,
        ),
    ],
)
def test_copy_dict(input_dict, output_dict, result, status_ok):
    if status_ok:
        assert copy_dict(input_dict, output_dict) == result
    else:
        with pytest.raises(ValueError):
            copy_dict(input_dict, output_dict)


@pytest.mark.parametrize(
    "input_dict,reference_dict,result",
    [
        # Existing test cases
        (
            {
                "input_str": "orange",
                "input_float": 42.0,
                "input_dict": {"nested_str": "banana", "nested_int": 42},
            },
            {
                "input_str": "apple",
                "input_float": 99.0,
                "input_dict": {"nested_str": "banana", "nested_int": 83},
                "input_default": "default",
            },
            True,
        ),
        ({}, {}, True),
        (
            {"unknown_input": 33, "known_input": "OK"},
            {"known_input": "AOK"},
            False,
        ),
        (
            {"input_str": "orange", "input_float": 42.0},
            {
                "input_str": "orange",
                "input_float": 42.0,
                "input_dict": {"nested_str": "banana", "nested_int": 42},
            },
            True,
        ),
        (
            {"input_str": "orange", "input_float": 42.0},
            {
                "input_string": "orange",
                "input_float": 42.0,
                "input_dict": {"nested_str": "banana", "nested_int": 42},
            },
            False,
        ),
        # New test cases
        (
            {"input_list": [1, 2, 3]},
            {"input_list": [4, 5, 6]},
            True,
        ),
        (
            {"input_str": "orange", "input_float": 42.0, "extra_key": "unexpected"},
            {"input_str": "apple", "input_float": 99.0},
            False,
        ),
    ],
)
def test_is_valid(input_dict, reference_dict, result):
    assert is_valid(input_dict, reference_dict) == result


@pytest.mark.parametrize(
    "path,expected_result",
    [
        ("valid_config.json", (True, {"key": "value"})),
        ("invalid_config.json", (False, {})),
    ],
)
def test_read_config(monkeypatch, path, expected_result):
    def mock_exists(p):
        return p == "valid_config.json"

    def mock_open(p, mode):
        class MockFile:
            def read(self):
                return '{"key": "value"}'

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                pass

        return MockFile()

    # Patch the os.path.exists and open functions using monkeypatch
    monkeypatch.setattr(os.path, "exists", mock_exists)
    monkeypatch.setattr("builtins.open", mock_open)

    # Assert that the function output matches the expected result
    assert read_config(path) == expected_result


@pytest.mark.parametrize(
    "config_dict,input_dict,expected_result",
    [
        (
            {"key": {"default": 10, "sub_weights": {"sub_key": {"default": 20}}}},
            {"key": {"default": 5, "sub_weights": {"sub_key": {"default": 10}}}},
            {"key": {"default": 10, "sub_weights": {"sub_key": {"default": 20}}}},
        ),
        (
            {"key": {"default": 0, "sub_weights": {"sub_key": {"default": 0}}}},
            {"key": {"default": 5, "sub_weights": {"sub_key": {"default": 10}}}},
            {"key": {"default": 0, "sub_weights": {"sub_key": {"default": 0}}}},
        ),
    ],
)
def test_update_defaults(config_dict, input_dict, expected_result):
    assert update_defaults(config_dict, input_dict) == expected_result


@pytest.mark.parametrize(
    "input_dict,expected_priority,expected_sub_priority",
    [
        (
            {"key": {"default": 10, "sub_weights": {"sub_key": {"default": 20}}}},
            {"key": 10},
            {"key": {"sub_key": 20}},
        ),
        (
            {"key": {"default": 0, "sub_weights": {"sub_key": {"default": 0}}}},
            {"key": 0},
            {},
        ),
    ],
)
def test_read_defaults(input_dict, expected_priority, expected_sub_priority):
    assert read_defaults(input_dict) == (expected_priority, expected_sub_priority)


@pytest.mark.parametrize(
    "input_dict,output_dict,key,expected_result",
    [
        (
            {"key": {"sub_key": "value"}},
            {"key": {"sub_key": "default_value"}},
            "key",
            {"key": {"sub_key": "value"}},
        ),
        (
            {"key": "value"},
            {"key": "default_value"},
            "key",
            {"key": "value"},
        ),
    ],
)
def test_copy_values(input_dict, output_dict, key, expected_result):
    assert copy_values(input_dict, output_dict, key) == expected_result


@pytest.mark.parametrize(
    "param_dict,expected_result",
    [
        (
            {"default": 5, "min_val": 0, "max_val": 100, "incr": 1},
            (5, 0, 100, 1),
        ),
        (
            {"default": 5},
            (5, 0, 100, 5),
        ),
    ],
)
def test_get_checkbox_params(param_dict, expected_result):
    assert _get_checkbox_params(param_dict) == expected_result


@pytest.fixture(name="eff_metric")
def efficiency_metrics_fixture():
    eff_metrics = EfficiencyMetrics()
    eff_metrics.set_weight(
        primary_metrics={
            "num_wells": 20,
            "num_unique_owners": 30,
            "avg_elevation_delta": 20,
            "age_range": 10,
            "depth_range": 20,
        }
    )
    return eff_metrics


@pytest.fixture(name="get_model")
def get_model_fixture(get_column_names, eff_metric):
    im_metrics, col_names, filename = get_column_names
    eff_metrics = eff_metric

    # Create the well data object
    wd = WellData(
        data=filename,
        column_names=col_names,
        impact_metrics=im_metrics,
        efficiency_metrics=eff_metrics,
    )

    # Partition the wells as gas/oil
    gas_oil_wells = wd.get_gas_oil_wells
    wd_gas = gas_oil_wells["gas"]

    # Mobilization cost
    mobilization_cost = {1: 120000, 2: 210000, 3: 280000, 4: 350000}
    for n_wells in range(5, len(wd_gas) + 1):
        mobilization_cost[n_wells] = n_wells * 84000

    wd_gas.compute_priority_scores()

    # Formulate the optimization problem
    opt_mdl_inputs = OptModelInputs(
        well_data=wd_gas,
        total_budget=3210000,  # 3.25 million USD
        mobilization_cost=mobilization_cost,
        threshold_distance=10,
        max_wells_per_owner=1,
    )

    opt_mdl_inputs.build_optimization_model()
    opt_campaign = opt_mdl_inputs.solve_model(solver="scip")

    return opt_campaign, opt_mdl_inputs, eff_metrics


@pytest.mark.widgets
def test_user_selection(solara_test, page_session: playwright.sync_api.Page, get_model):
    opt_campaign, opt_mdl_inputs, _ = get_model

    or_wid_class = UserSelection(opt_campaign.clusters_dict, opt_mdl_inputs)

    # Test the structure of the override widget
    assert hasattr(or_wid_class, "wd")
    assert hasattr(or_wid_class, "opt_campaign")
    assert 789 in or_wid_class.well_selected_list
    assert isinstance(or_wid_class.well_selected, WellData)
    assert 789 in or_wid_class.well_selected.data.index
    assert 1 in or_wid_class.cluster_remove_choice
    assert len(or_wid_class.all_wells) == len(opt_mdl_inputs.config.well_data)
    assert isinstance(or_wid_class.remove_widget, SubSelectWidget)
    assert isinstance(or_wid_class.button_remove_confirm, widgets.Button)
    assert isinstance(or_wid_class.add_widget, SelectWidgetAdd)
    assert isinstance(or_wid_class.lock_widget, SubSelectWidget)
    assert hasattr(or_wid_class.add_widget, "re_cluster")
    assert isinstance(or_wid_class.add_widget.re_cluster, widgets.BoundedIntText)

    or_wid_class.display()

    # Assert list is empty initially
    assert not or_wid_class.remove_widget.cluster_widget.selected_list
    assert not or_wid_class.remove_widget.selected_list
    assert isinstance(
        or_wid_class.remove_widget.cluster_widget.button_add, widgets.Button
    )
    assert isinstance(
        or_wid_class.remove_widget.cluster_widget.button_remove, widgets.Button
    )
    assert isinstance(
        or_wid_class.remove_widget.cluster_widget.widget, widgets.Combobox
    )
    assert isinstance(or_wid_class.remove_widget.button_add, widgets.Button)
    assert isinstance(or_wid_class.remove_widget.button_remove, widgets.Button)
    assert isinstance(or_wid_class.remove_widget.widget, widgets.Combobox)

    # Remove project 13
    page_session.get_by_label("Project").fill("13")
    page_session.wait_for_timeout(2000)
    project_remove_button = page_session.locator(
        "text=Select projects to manually remove"
    )
    project_remove_button.click()
    assert or_wid_class.remove_widget.cluster_widget.selected_list == ["13"]
    assert or_wid_class.remove_widget.cluster_widget.widget.options == (
        "1",
        "11",
        "13",
        "19",
    )
    assert or_wid_class.remove_widget.cluster_widget._text == "13"
    cluster = or_wid_class.remove_widget._pass_current_selection()
    assert or_wid_class.remove_widget.widget.options == ()
    page_session.get_by_label("Well").fill("1")
    page_session.wait_for_timeout(2000)
    assert or_wid_class.remove_widget.widget.options == (
        "16079",
        "50038",
        "46413",
        "13528",
    )

    # Remove well 48446, index 789
    page_session.get_by_label("Well").fill("48446")
    well_remove_button = page_session.locator("text=Select wells to manually remove")
    well_remove_button.click()
    page_session.wait_for_timeout(2000)
    assert or_wid_class.remove_widget.selected_list == ["48446"]

    # Remove well 84290, index 829
    page_session.get_by_label("Well").fill("84290")
    well_remove_button.click()
    page_session.wait_for_timeout(2000)
    assert or_wid_class.remove_widget.selected_list == ["48446", "84290"]

    # Unselect recent selection
    project_undo_button = page_session.locator("text=Undo").nth(0)
    well_undo_button = page_session.locator("text=Undo").nth(1)
    well_undo_button.click()
    page_session.wait_for_timeout(2000)
    assert or_wid_class.remove_widget.selected_list == ["48446"]

    # Test the confirm removal button
    assert not hasattr(or_wid_class, "cluster_lock_choice")
    assert not hasattr(or_wid_class, "well_lock_choice")

    or_wid_class.button_remove_confirm.click()

    # Test the re_cluster text box is empty
    assert not or_wid_class.add_widget.re_cluster_dict

    # Add well 94343, index 80
    page_session.get_by_label("Add Well").fill("94343")
    page_session.get_by_label("To Project").fill("1")
    well_add_button = page_session.locator("text=Select wells to manually add")
    well_add_button.click()
    page_session.wait_for_timeout(2000)
    assert or_wid_class.add_widget.selected_list == ["94343"]
    assert or_wid_class.add_widget.re_cluster_dict == {1: [80]}

    # Add the same well twice
    or_wid_class.add_widget._text = "94343"
    with pytest.raises(
        ValueError, match="Choice 94343 already included in list of selections"
    ):
        or_wid_class.add_widget._add(None)

    # Unselect None
    or_wid_class.add_widget._text = ""
    with pytest.raises(ValueError, match="Nothing selected, cannot remove from list"):
        or_wid_class.add_widget._remove(None)

    # Add well 69254, index 600
    page_session.get_by_label("Add Well").fill("69254")
    well_add_button = page_session.locator("text=Select wells to manually add")
    well_add_button.click()
    page_session.wait_for_timeout(2000)
    assert or_wid_class.add_widget.selected_list == ["94343", "69254"]
    assert or_wid_class.add_widget.re_cluster_dict == {1: [80], 6: [600]}

    # Unselect well 69254
    or_wid_class.add_widget._remove("69254")
    assert or_wid_class.add_widget.selected_list == ["94343"]
    assert or_wid_class.add_widget.re_cluster_dict == {1: [80], 6: []}

    # Lock project 19
    page_session.wait_for_timeout(2000)
    page_session.get_by_label("Project").nth(1).fill("19")
    project_lock_button = page_session.locator("text=Select projects to manually lock")
    project_lock_button.click()
    assert or_wid_class.lock_widget.cluster_widget.selected_list == ["19"]
    assert or_wid_class.lock_widget.cluster_widget._text == "19"

    # Add None to the list
    or_wid_class.lock_widget._text = ""
    with pytest.raises(ValueError, match="Nothing selected, cannot add to list"):
        or_wid_class.lock_widget._add(None)

    # Remove well not in the selected list
    or_wid_class.lock_widget._text = "69254"
    with pytest.raises(ValueError, match="Choice 69254 is not in the list"):
        or_wid_class.lock_widget._remove(None)

    # Test the structure of the override widget return
    or_selection = or_wid_class.return_value()
    assert isinstance(or_selection, OverrideSelections)
    assert isinstance(or_selection.remove_widget_return, WidgetReturn)
    assert isinstance(or_selection.add_widget_return, AddWidgetReturn)
    assert isinstance(or_selection.lock_widget_return, WidgetReturn)
    assert hasattr(or_selection.remove_widget_return, "cluster")
    assert hasattr(or_selection.remove_widget_return, "well")
    assert hasattr(or_selection.add_widget_return, "existing_cluster")
    assert hasattr(or_selection.add_widget_return, "new_cluster")
    assert hasattr(or_selection.lock_widget_return, "cluster")
    assert hasattr(or_selection.lock_widget_return, "well")

    assert or_selection.remove_widget_return.cluster == [13]
    assert or_selection.remove_widget_return.well == {1: [789]}
    assert or_selection.add_widget_return.existing_cluster == {11: [80]}
    assert or_selection.add_widget_return.new_cluster == {1: [80], 6: []}
    assert or_selection.lock_widget_return.cluster == [19]
    assert or_selection.lock_widget_return.well == {19: [21, 83, 182, 280, 981]}
