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
import logging

# Installed libs
import pandas as pd
from pyomo.core.base.block import BlockData, declare_custom_block
from pyomo.environ import Binary, Constraint, NonNegativeReals, Set, Var

# User-defined libs
from primo.data_parser.default_data import WELL_BASED_METRICS, WELL_PAIR_METRICS
from primo.utils.raise_exception import MissingDataError, raise_exception

LOGGER = logging.getLogger(__name__)


def build_num_unique_owners_model(m):
    """
    Builds constraints and variables to track the number of unique well owners
    in the optimization model.

    Parameters
    ----------
    m : pyomo.environ.Block
        The model block where the unique owner constraints and variables
        will be added.
    """

    cm = m.parent_block().parent_block()
    wd = cm.parent_block().model_inputs.config.well_data
    if wd.col_names.operator_name is None:
        raise_exception("Operator Name column not available", MissingDataError)

    operator_list = (
        wd[wd.col_names.operator_name].loc[list(cm.set_wells)].unique().tolist()
    )

    m.owner_dict = {owner: [] for owner in operator_list}
    for well in cm.set_wells:
        m.owner_dict[wd.data.loc[well, wd.col_names.operator_name]].append(well)
    # {Owner 1: [w1, w2, w3, ....]
    # Key => Owner name, Value => index

    m.set_owners = Set(
        initialize=operator_list,
        doc="set of unique owners in cluster c",
    )

    m.select_owner = Var(
        m.set_owners,
        within=Binary,
        doc="Binary variable to track if an owner's well is selected",
    )

    @m.Constraint(
        cm.set_wells,
        doc="select_owner variable is set to 1 if well is selected",
    )
    def owner_selection_constraint(_, w):
        return (
            cm.select_well[w] <= m.select_owner[wd[wd.col_names.operator_name].loc[w]]
        )

    m.num_owners_chosen = Var(
        within=NonNegativeReals,
        doc="Variable to keep track of total number of unique owners",
    )

    m.num_owners_constraint = Constraint(
        expr=(
            m.num_owners_chosen == sum(m.select_owner[owner] for owner in m.set_owners)
        ),
        doc="Constraint to calculate number of unique owners chosen in a project",
    )

    @m.Constraint(
        m.set_owners,
        doc="Set select_owner variable to 0 if no well of that owner is selected in a project",
    )
    def do_not_select_owner(_, owner):
        return m.select_owner[owner] <= sum(
            m.cluster_model.select_well[w] for w in m.owner_dict[owner]
        )


@declare_custom_block("MaxFormulationBlock")
class MaxFormulationBlockData(BlockData):
    """
    Block for building max-scaling efficiency model
    """

    @property
    def cluster_model(self):
        """
        Returns a pointer to the cluster model
        """
        return self.parent_block().parent_block()

    @property
    def plugging_campaign_model(self):
        """
        Returns a pointer to the plugging campaign model
        """
        return self.parent_block().parent_block().parent_block()

    def compute_metric_score(
        self,
        weight: int,
        metric_data: pd.Series,
        scaling_factor: float,
        metric_type: str,
    ):
        """
        Builds the efficiency expressions for well-based metrics
        """
        # pylint: disable = attribute-defined-outside-init
        self.score = Var(
            domain=NonNegativeReals,
            doc="Score variable for this efficiency metric",
        )
        self.score_upper_bound = Constraint(
            expr=self.score <= weight * self.cluster_model.select_cluster,
            doc="Set score to 0 if cluster is not selected",
        )
        well_vars = self.cluster_model.select_well
        select_cluster = self.cluster_model.select_cluster
        norm_metric_data = metric_data / scaling_factor
        norm_metric_data[norm_metric_data >= 1] = 1

        if metric_type == "well_based":

            @self.Constraint(self.cluster_model.set_wells)
            def calculate_score(blk, w):
                return blk.score <= weight * (
                    select_cluster - norm_metric_data[w] * well_vars[w]
                )

        elif metric_type == "well_pair":

            @self.Constraint(self.cluster_model.set_well_pairs)
            def calculate_score(blk, w1, w2):
                return select_cluster - blk.score / weight >= (
                    norm_metric_data[w1, w2]
                    * (well_vars[w1] + well_vars[w2] - select_cluster)
                )

        elif metric_type == "num_wells":

            @self.Constraint()
            def calculate_score(blk):
                return (
                    blk.score
                    <= weight * sum(well_vars[w] for w in well_vars) / scaling_factor
                )

        elif metric_type == "num_unique_owners":

            build_num_unique_owners_model(self)

            if scaling_factor == 1:

                @self.Constraint()
                def calculate_score(_):
                    return self.num_owners_chosen == self.cluster_model.select_cluster

            else:

                @self.Constraint()
                def calculate_score(blk):
                    return blk.score <= weight * (
                        select_cluster
                        - (self.num_owners_chosen - select_cluster)
                        / (scaling_factor - 1)
                    )


def build_cluster_efficiency_model(eff_blk):
    """
    Builds efficiency model for each cluster

    Parameters
    ----------
    cm : ClusterBlock
        Cluster model object
    """
    # # For reference, this is the model Hierarchy
    # PluggingCampaignModel/Pyomo ConcreteModel
    #     |__ClusterBlock
    #         |__EfficiencyBlock
    #             |__MaxFormulationBlock

    # OptModelInputs's config object that contains zone information
    cm = eff_blk.parent_block()  # cluster model block
    pm = cm.parent_block()  # Plugging campaign model/ConcreteModel
    sf = pm.model_inputs.config  # Block containing scaling factors
    wd = sf.well_data  # WellData object
    eff_metrics = wd.config.efficiency_metrics
    weights = eff_metrics.get_weights
    list_wells = list(cm.set_wells)  # List of wells in this cluster

    # Assess well-based metrics
    for metric in WELL_BASED_METRICS:
        if getattr(weights, metric, 0) == 0:
            # Metric is not selected. So, Skip
            continue

        # Construct Efficiency model for the metric
        # pylint: disable = undefined-variable
        # pylint: disable=protected-access
        col_name = getattr(wd.col_names, getattr(eff_metrics, metric)._required_data)
        setattr(eff_blk, metric, MaxFormulationBlock())
        getattr(eff_blk, metric).compute_metric_score(
            weight=getattr(weights, metric),
            metric_data=wd.data.loc[list_wells, col_name],
            scaling_factor=getattr(sf, "max_" + metric),
            metric_type="well_based",
        )

    pairwise_metrics = cm.parent_block().model_inputs.pairwise_metrics[cm.index()]
    if pairwise_metrics is not None:
        cm.set_well_pairs = Set(initialize=pairwise_metrics.index.to_list())

    for metric in WELL_PAIR_METRICS:
        if getattr(weights, metric, 0) == 0:
            # Metric is not selected, so skip
            continue

        # pylint: disable = undefined-variable
        setattr(eff_blk, metric, MaxFormulationBlock())
        getattr(eff_blk, metric).compute_metric_score(
            weight=getattr(weights, metric),
            metric_data=pairwise_metrics[metric],
            scaling_factor=getattr(sf, "max_" + metric),
            metric_type="well_pair",
        )

    if weights.num_wells > 0:
        # pylint: disable = undefined-variable
        metric = "num_wells"
        setattr(eff_blk, metric, MaxFormulationBlock())
        getattr(eff_blk, metric).compute_metric_score(
            weight=getattr(weights, metric),
            metric_data=pd.Series([0, 0]),
            scaling_factor=getattr(sf, "max_" + metric),
            metric_type=metric,
        )

    if weights.num_unique_owners > 0:
        # pylint: disable = undefined-variable
        metric = "num_unique_owners"
        setattr(eff_blk, metric, MaxFormulationBlock())
        getattr(eff_blk, metric).compute_metric_score(
            weight=getattr(weights, metric),
            metric_data=pd.Series([0, 0]),
            scaling_factor=getattr(sf, "max_" + metric),
            metric_type=metric,
        )
