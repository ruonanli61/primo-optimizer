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

# Installed lib
from fast_autocomplete import AutoComplete
from IPython.display import Javascript, display
import ipywidgets as widgets
import pandas as pd

# User defined lib
from primo.utils.raise_exception import raise_exception


class overridewidget:
    """
    Class for displaying a autofill widget on the Jupyter Notebook to allow user select
    wells would like to add to or remove from the optimal solution

    Parameters
    ----------

    well_df : pd.DataFrame
        A data frame of wells, from which users can choose the wells would like to be added
        or removed from the result of optimization problem

    Attributes
    ----------

    widget : widgets.Combobox
        A text widget with autofill feature for user selecting wells would like to be
        included in the well_list

    button : widgets.Button
        A button to confirm and add the selected well to the well_list

    well_list : list
        A list which includes the API well number of wells that would like to be added to
        or removed from the result of the optimization problem
    """

    def __init__(self, well_df: pd.DataFrame, button_description: str):
        well_dict = {str(well_id): {} for well_id in well_df["API Well Number"]}

        words = well_dict
        self.autocomplete = AutoComplete(words=words)
        layout = widgets.Layout(width="auto", height="auto")
        self.widget = widgets.Combobox(
            value="",
            placeholder="Select well",
            description="Well",
            disabled=False,
        )
        self.widget.observe(self._on_change, names="value")

        layout = widgets.Layout(width="auto", height="auto")

        self.button_add = widgets.Button(description=button_description, layout=layout)
        self.button_add.on_click(self._on_button_clicked_add)
        self.button_remove = widgets.Button(description="Undo", layout=layout)
        self.button_remove.on_click(self._on_button_clicked_remove)
        self.well_list = []

    def _on_change(self, data):
        """
        Dynamically update the well option list in the drop down section of the widget
        based on the information user has already typed in
        """

        self.text = data["new"]

        values = self.autocomplete.search(self.text, max_cost=3, size=3)

        # convert nested list to flat list
        values = list(sorted(set(str(item) for sublist in values for item in sublist)))

        # remove previous options from tag `<datalist>` in HTML
        # display(Javascript(""" document.querySelector("datalist").innerHTML = "" """))

        self.widget.options = values

    def _on_button_clicked_add(self, _):
        """
        Add the selected well to the well_list and print the corresponding message in
        the Jupyter Notebook
        """

        if self.text in self.well_list:
            raise_exception(
                f"Well {self.text} has already been added to the override list",
                ValueError,
            )
        else:
            self.well_list.append(self.text)
            print(f"Well {self.text} has been added to the override list.")

    def _on_button_clicked_remove(self, _):
        """
        Remove a selected well from the well_list and print the corresponding message in
        the Jupyter Notebook
        """

        if self.text not in self.well_list:
            raise_exception(
                f"Well {self.text} is not in the list",
                ValueError,
            )
        else:
            self.well_list.remove(self.text)
            print(f"Well {self.text} has been removed from the list.")

    def display(self):
        """
        display the widget and button in the Jupyter Notebook
        """
        buttons = widgets.HBox([self.button_add, self.button_remove])
        display(self.widget, buttons)

    def return_value(self):
        """
        Return the list of selected wells
        """
        return self.well_list


def UserInput(well_df_add: pd.DataFrame, well_df_remove: pd.DataFrame):
    """
    A wrapper to generate the override widget for adding wells and removing wells

    """
    widget_add = overridewidget(well_df_add, "Add the well to the suggested projects")
    widget_remove = overridewidget(
        well_df_remove, "Remove the well from the suggested projects"
    )

    return widget_add, widget_remove


class recalculate:
    def __init__(
        self,
        original_well_list: pd.DataFrame,
        well_add_list: list,
        well_remove_list: list,
        well_df: pd.DataFrame,
        mobilization_costs,
        budget,
        dac_weight,
        dac_budget_fraction,
        max_wells_per_owner,
    ):
        self.original_well_list = original_well_list
        self.well_add_list = well_add_list
        self.well_remove_list = well_remove_list
        self.well_df = well_df
        self.mobilization_costs = mobilization_costs
        self.budget = budget
        self.dac_weight = dac_weight
        self.dac_budget_fraction = dac_budget_fraction
        self.max_wells_per_owner = max_wells_per_owner

        for well_id in self.well_add_list:
            well_id = int(well_id)
            well_add = self.well_df[self.well_df["API Well Number"] == well_id]
            self.original_well_list = pd.concat([self.original_well_list, well_add])

        well_remove_list = [int(well_id) for well_id in self.well_remove_list]
        self.original_well_list["drop"] = self.original_well_list.apply(
            lambda row: "0" if row["API Well Number"] in well_remove_list else "1",
            axis=1,
        )
        self.well_return_df = self.original_well_list[
            self.original_well_list["drop"] == "1"
        ]

    def budget_assess(self):
        total_cost = 0
        for _, groups in self.well_return_df.groupby("Project"):
            n_wells = len(groups)
            campaign_cost = self.mobilization_costs[n_wells]
            total_cost += campaign_cost
        if total_cost > self.budget:
            self.violate_cost = total_cost
        else:
            self.violate_cost = False
        return self.violate_cost

    def dac_assess(self):
        for _, row in self.well_return_df.iterrows():
            well_id = row["API Well Number"]
            if self.dac_weight is not None:
                threshold = self.dac_weight / 100 * self.dac_budget_fraction
                disadvantaged_community_score = row[
                    f"DAC Score [0-{int(self.dac_weight)}]"
                ]
                is_disadvantaged = float(disadvantaged_community_score > threshold)
            else:
                # When the user does not select DAC as a priority factor,
                # all wells are assumed to not be located in a disadvantaged community.
                is_disadvantaged = float(False)
            self.well_return_df.loc[
                self.well_return_df["API Well Number"] == well_id, ["In DAC"]
            ] = is_disadvantaged
        dac_percent = (
            self.well_return_df["In DAC"].sum() / len(self.well_return_df) * 100
        )
        if dac_percent < self.dac_budget_fraction:
            self.violate_dac = dac_percent
        else:
            self.violate_dac = False
        return self.violate_dac

    def operator_assess(self):
        self.violate_operator = {}
        for operator, groups in self.well_return_df.groupby("Operator Name"):
            n_wells = len(groups)
            if n_wells > self.max_wells_per_owner:
                self.violate_operator[operator] = n_wells
        return self.violate_operator
