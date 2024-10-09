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
import copy
import logging
from itertools import combinations
from typing import Dict, List, Optional

# Installed libs
import pandas as pd
from haversine import Unit, haversine_vector

# User-defined libs
from primo.data_parser.data_model import OptInputs
from primo.opt_model.result_parser import Campaign
from primo.utils.clustering_utils import distance_matrix
from primo.utils.raise_exception import raise_exception

LOGGER = logging.getLogger(__name__)


class AssessFeasibility:
    """
    Class for assessing whether the overridden P&A projects adhere to the constraints
    defined in the optimization problem

    Parameters
    ----------
    """

    def __init__(self, opt_inputs, opt_campaign: Dict, wd, plug_list):

        self.opt_inputs = opt_inputs
        self.new_campaign = opt_campaign
        self.wd = wd
        self.plug_list = plug_list
        self.campaign_cost_dict = {}

    def assess_budget(self) -> float:
        """
        Assesses the impact on budget after user overrides and returns the
        amount by which the budget is violated. A 0 or negative value indicates
        that we are still under budget
        """
        total_cost = 0
        for cluster, groups in self.new_campaign.items():
            n_wells = len(groups)
            campaign_cost = self.opt_inputs.get_mobilization_cost[n_wells]
            self.campaign_cost_dict[cluster] = campaign_cost
            total_cost += campaign_cost

        return (total_cost - self.opt_inputs.get_total_budget) * 1e6

    def assess_dac(self) -> float:
        """
        Assess whether the DAC constraint is violated or not and returns
        the amount by which the DAC constraint is violated. A 0 or negative value
        indicates that constraint is satisfied
        """
        opt_inputs = self.opt_inputs.config
        dac_weight = opt_inputs.perc_wells_in_dac
        if dac_weight is None:
            # When the user does not select DAC as a priority factor,
            # this constraint becomes meaningless
            return 0

        disadvantaged_wells = 0

        # count number of wells that exceed threshold based on disadvantaged
        # community score
        disadvantaged_wells = sum(
            self.wd.data.loc[well, "is_disadvantaged"] for well in self.plug_list
        )
        dac_percent = disadvantaged_wells / len(self.plug_list)

        return opt_inputs.perc_wells_in_dac - dac_percent

    def assess_owner_well_count(self) -> Dict:
        """
        Assess whether the owner well count constraint is violated or not.
        Returns list of owners and wells selected for each for whom the owner
        well count constraint is violated
        """
        violated_operators = {}
        for operator, groups in self.wd.data.groupby(self.wd._col_names.operator_name):
            n_wells = len(groups)
            if n_wells > self.opt_inputs.config.max_wells_per_owner:
                violated_operators.setdefault("Owner", []).append(operator)
                violated_operators.setdefault("Number of wells", []).append(n_wells)
                violated_operators.setdefault("Wells", []).append(
                    groups[self.wd._col_names.well_id].to_list()
                )

        return violated_operators

    def assess_distances(self) -> Dict:
        """
        Assess whether the maximum distance between two wells constraint is violated or not
        """
        distance_threshold = self.opt_inputs.config.threshold_distance
        distance_violation = {}
        metric_array = distance_matrix(self.wd, {"distance": 1})
        df_to_array = {
            df_index: array_index
            for array_index, df_index in enumerate(self.wd.data.index)
        }

        for cluster, well_list in self.new_campaign.items():
            for w1, w2 in combinations(well_list, 2):
                well_distance = metric_array[df_to_array[w1], df_to_array[w2]]
                if well_distance > distance_threshold:
                    distance_violation.setdefault("Project", []).append(cluster)
                    distance_violation.setdefault("Well 1", []).append(
                        self.wd.data.loc[w1][self.wd._col_names.well_id]
                    )
                    distance_violation.setdefault("Well 2", []).append(
                        self.wd.data.loc[w2][self.wd._col_names.well_id]
                    )
                    distance_violation.setdefault(
                        "Distance between Well 1 and 2 [Miles]", []
                    ).append(well_distance)

        return distance_violation

    def assess_feasibility(self) -> bool:
        """
        Assesses whether current set of selections is feasible
        """
        if self.assess_budget() > 0:
            return False

        if self.assess_dac() > 0:
            return False

        if self.assess_owner_well_count():
            return False

        if self.assess_distances():
            return False

        return True


class OverrideCampaign:
    """
    Class for constructing new campaigns based on the override results
    and returning infeasibility information.

    Parameters
    ----------
    override_list: list
        A list with well add, well remove, and well lock information

    opt_inputs: OptModelInputs
        Object containing the necessary inputs for the optimization model

    opt_campaign: dict
        A dictionary for the original suggested P&A project
        where keys are cluster numbers and values
        are list of wells for each cluster.
    """

    def __init__(
        self,
        override_list: List,
        opt_inputs,
        opt_campaign: Dict,
        eff_metrics,
    ):
        logging.getLogger("Campaign").setLevel(logging.WARNING)
        opt_campaign_copy = copy.deepcopy(opt_campaign)
        self.new_campaign = opt_campaign_copy
        self.added = override_list[0]
        self.remove = override_list[1]
        self.lock = override_list[2]
        self.opt_inputs = opt_inputs
        self.eff_metrics = eff_metrics

        # form the new projects based on the override list
        # self.new_campaign.modified_project()

        # def modified_project(self):
        # change well cluster
        self._modify_campaign()
        self.plug_list = []
        for cluster, well_list in self.new_campaign.items():
            self.plug_list += well_list
        self.wd = self.opt_inputs.config.well_data._construct_sub_data(self.plug_list)

        self.feasibility = AssessFeasibility(
            self.opt_inputs, self.new_campaign, self.wd, self.plug_list
        )

    def _modify_campaign(self):
        for cluster, well_list in self.added[0].items():
            if cluster in self.new_campaign.keys():
                for well in well_list:
                    if well in self.new_campaign[cluster]:
                        self.new_campaign[cluster].remove(well)

        # add well with new cluster
        for cluster, well_list in self.added[1].items():
            self.new_campaign.setdefault(cluster, []).extend(well_list)

        # remove clusters
        for cluster in self.remove[0]:
            del self.new_campaign[cluster]

        # remove wells
        for cluster, well_list in self.remove[1].items():
            for well in well_list:
                self.new_campaign[cluster].remove(well)

    def violation_info(self):
        violation_info_dict = {}
        if self.feasibility.assess_feasibility() is False:
            violation_info_dict = {"Project Status:": "INFEASIBLE"}
            violate_cost = self.feasibility.assess_budget()
            violate_operator = self.feasibility.assess_owner_well_count()
            violate_distance = self.feasibility.assess_distances()
            violate_dac = self.feasibility.assess_dac()

            if violate_cost > 0:
                msg = f""" After the modification, the total budget is over the limit by ${int(violate_cost)}. Please consider modifying wells you have selected by either using the widget above or by re-running the optimization problem."""
                violation_info_dict[msg] = """"""

            if violate_operator:
                msg = f""" After the modification, the following owners have more than {self.opt_inputs.config.max_wells_per_owner} well(s) 
                                being selected. Please consider modifying wells you have selected by either using the widget above or by re-running
                                the optimization problem."""

                violate_operator_df = pd.DataFrame.from_dict(violate_operator)
                violation_info_dict[msg] = violate_operator_df

            if violate_distance:
                msg = """ After the modification, the following projects have wells are far away from each others. 
                                Please consider modifying wells you have selected by either using the widget above or by 
                                re-running the optimization problem."""

                violate_distance_df = pd.DataFrame.from_dict(violate_distance)
                violation_info_dict[msg] = violate_distance_df

            if violate_dac > 0:
                dac_percent = self.opt_inputs.config.perc_wells_in_dac - violate_dac
                msg = f""" After the modification, {int(dac_percent)}% of well is in DAC. Please consider modifying wells you have selected by either using the widget above or by re-running the optimization problem."""
                violation_info_dict[msg] = """"""
        else:
            violation_info_dict = {"Project Status:": "FEASIBLE"}

        return violation_info_dict

    def override_campaign(self):
        plugging_cost = self.feasibility.campaign_cost_dict
        return Campaign(self.wd, self.new_campaign, plugging_cost)

    def recalculate(self):
        logging.disable(logging.CRITICAL)
        override_campaign = self.override_campaign()
        override_campaign.set_efficiency_weights(self.eff_metrics)
        override_campaign.compute_efficiency_scores()
        print(override_campaign)

    def _re_optimize_dict(self):
        re_optimize_dict = {}
        re_optimize_well_dict = {}
        # for cluster, well_list in self.added[0].items():
        #     if cluster in self.new_campaign.keys():
        #         for well in well_list:
        #             if well in self.new_campaign[cluster]:
        #                 self.new_campaign[cluster].remove(well)

        # # add well with new cluster
        # for cluster, well_list in self.added[1].items():
        #     self.new_campaign.setdefault(cluster, []).extend(well_list)

        # assign 0 to clusters being removed
        for cluster in self.remove[0]:
            re_optimize_dict[cluster] = 0

        # assign 0 to wells being removed
        for cluster, well_list in self.remove[1].items():
            for well in well_list:
                well_dict[well] = 0
            re_optimize_well_dict[cluster] = well_dict

        # assign 1 to wells being locked
        for cluster, well_list in self.lock.items():
            well_dict = re_optimize_dict[cluster]
            for well in well_list:
                well_dict[well] = 1
            re_optimize_well_dict[cluster] = well_dict

        return re_optimize_well_dict


# def re_optimize():


# Retain to be used when implementing the backfill feature
# class BackFill:
#     """
#     Class for assessing whether the overridden P&A projects adhere to the constraints
#     defined in the optimization problem

#     Parameters
#     ----------

#     selected_wells : pd.DataFrame
#         A DataFrame containing wells selected based on solving the optimization problem
#         and/or by manual selections and overrides

#     wells_added : List[str]
#         A list of wells that the user wishes to add to the P&A projects

#     wells_removed : List[str]
#         A list of wells that the user wishes to remove from the P&A projects

#     well_df : pd.DataFrame
#         A DataFrame that includes all candidate wells

#     opt_inputs : OptInputs
#         Input object for the optimization problem

#     dac_weight : int
#         An integer for the weight assigned to the DAC priority factor.

#     Attributes
#     ----------

#     original_wells : pd.DataFrame
#         List of original wells selected for plugging

#     wells_added : List[str]
#         List of wells to be added for plugging

#     wells_removed : List[str]
#         List of wells to be removed from plugging projects

#     well_df : pd.DataFrame
#         Full data associated with all wells

#     opt_inputs : OptInputs
#         The inputs associated with the optimization problem

#     dac_weight : Union[float, None]
#         An integer for the weight assigned to the DAC priority factor.

#     """

#     # pylint: disable=too-many-arguments
#     def __init__(
#         self,
#         override_list: List,
#         opt_inputs: OptModelInputs,
#     ):
#         self.wells = opt_inputs.config.well_data
#         self.campaign = opt_inputs.campaign_candidates
#         self.wells_added = [int(well_id) for well_id in wells_added]
#         self.wells_removed = [int(well_id) for well_id in wells_removed]
#         # self.opt_inputs = opt_inputs
#         self.dac_weight = opt_inputs.config.perc_wells_in_dac

#         # Get list of wells to be plugged (opt results + user overrides)
#         added_wells = well_df[well_df["API Well Number"].isin(self.wells_added)]
#         self._plugged_list = pd.concat([original_wells.copy(), added_wells])

#         # Get list of wells to be removed from plugging
#         self._plugged_list = self._plugged_list[
#             ~self._plugged_list["API Well Number"].isin(self.wells_removed)
#         ]

#     def _add_well(self, well_id: int):
#         """
#         Adds a well in the considerations for plugging
#         """
#         if well_id in self.wells_added:
#             msg = f"Well: {well_id} was already included in plugging list"
#             LOGGER.warning(msg)
#             print(msg)
#             return

#         if well_id in self.wells_removed:
#             raise_exception(
#                 f"Well: {well_id} was already included in removal list", ValueError
#             )

#         self.wells_added.append(well_id)
#         self._plugged_list = pd.concat(
#             [
#                 self._plugged_list,
#                 self.well_df[self.well_df["API Well Number"] == well_id],
#             ]
#         )
#         return

#     def _remove_well(self, well_id: int):
#         """
#         Removes a well from the considerations for plugging
#         """
#         if well_id in self.wells_removed:
#             raise_exception(
#                 f"Well: {well_id} was already included in removal list", ValueError
#             )

#         if well_id in self.wells_added:
#             self.wells_added.remove(well_id)

#         self._plugged_list = self._plugged_list[
#             self._plugged_list["API Well Number"] != well_id
#         ]


#     def backfill(self) -> List[str]:
#         """
#         If current selections (either due to manual overrides or termination of solver
#         prior to optimality)
#         leave room for budget, the method returns a list of candidates that can
#         be added to the plugging projects without violating any constraints

#         Parameters:
#         ----------
#         None

#         Returns:
#         --------
#         List of well API numbers that can be added in the project
#         """
#         additions = []
#         if self.assess_feasibility() is False:
#             # The current well selections already lead to infeasibility (presumably
#             # due to budget being exceeded)
#             # There is no room to add more wells for backfilling
#             return additions

#         # Sort by descending order
#         candidates = self.well_df.sort_values("Priority Score [0-100]", ascending=False)

#         # Only include those candidates not considered for plugging
#         candidates = candidates[
#             ~candidates["API Well Number"].isin(self._plugged_list["API Well Number"])
#         ]

#         # Remove candidates in removal list
#         candidates = candidates[~candidates["API Well Number"].isin(self.wells_removed)]
#         # Stick to those projects that already exist
#         existing_projects = set(self._plugged_list["Project"])
#         candidates = candidates[candidates["Project"].isin(existing_projects)]

#         for well_id in candidates["API Well Number"]:
#             self._add_well(well_id)
#             if self.assess_feasibility():
#                 additions.append(well_id)
#             else:
#                 self._remove_well(well_id)

#         return additions
