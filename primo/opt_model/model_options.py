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
import numpy as np
from pyomo.common.config import (
    Bool,
    ConfigDict,
    ConfigValue,
    In,
    IsInstance,
    NonNegativeFloat,
    NonNegativeInt,
    document_kwargs_from_configdict,
)

# User-defined libs
from primo.data_parser.default_data import WELL_BASED_METRICS, WELL_PAIR_METRICS
from primo.data_parser.well_data import WellData
from primo.opt_model.model_with_clustering import PluggingCampaignModel
from primo.utils import get_solver
from primo.utils.clustering_utils import (
    get_pairwise_metrics,
    perform_agglomerative_clustering,
    perform_louvain_clustering,
)
from primo.utils.domain_validators import InRange, validate_mobilization_cost
from primo.utils.raise_exception import raise_exception

LOGGER = logging.getLogger(__name__)


def model_config() -> ConfigDict:
    """
    Returns a Pyomo ConfigDict object that includes all user options
    associated with optimization modeling
    """
    # Container for storing and performing domain validation
    # of the inputs of the optimization model.
    # ConfigValue automatically performs domain validation.
    config = ConfigDict()

    # Essential inputs for the optimization model

    config.declare(
        "well_data",
        ConfigValue(
            domain=IsInstance(WellData),
            doc="WellData object containing the entire dataset",
        ),
    )
    config.declare(
        "total_budget",
        ConfigValue(
            domain=NonNegativeFloat,
            doc="Total budget for plugging [in USD]",
        ),
    )
    config.declare(
        "mobilization_cost",
        ConfigValue(
            domain=validate_mobilization_cost,
            doc="Cost of plugging wells [in USD]",
        ),
    )

    config.declare(
        "shallow_gas_well_cost",
        ConfigValue(
            domain=NonNegativeFloat,
            doc="Cost of plugging a single shallow gas well in $",
        ),
    )

    config.declare(
        "deep_gas_well_cost",
        ConfigValue(
            domain=NonNegativeFloat,
            doc="Cost of plugging a single deep gas well in $",
        ),
    )

    config.declare(
        "shallow_oil_well_cost",
        ConfigValue(
            domain=NonNegativeFloat,
            doc="Cost of plugging a single shallow oil well in $",
        ),
    )

    config.declare(
        "deep_oil_well_cost",
        ConfigValue(
            domain=NonNegativeFloat,
            doc="Cost of plugging a single deep oil well in $",
        ),
    )

    config.declare(
        "beta",
        ConfigValue(
            domain=InRange(0, 1),
            doc="Parameter that defines economies of scale "
            "when plugging multiple wells",
        ),
    )

    # Model type and model nature options
    config.declare(
        "efficiency_formulation",
        ConfigValue(
            default="Max Scaling",
            domain=In(["Max Scaling", "Zone"]),
            doc="Efficiency Formulation",
        ),
    )
    config.declare(
        "objective_weight_impact",
        ConfigValue(
            default=50,
            domain=InRange(0, 100),
            doc="Weight associated with Impact in the objective function",
        ),
    )
    config.declare(
        "num_wells_model_type",
        ConfigValue(
            default="multicommodity",
            domain=In(["multicommodity", "incremental"]),
            doc="Choice of formulation for modeling number of wells",
        ),
    )
    config.declare(
        "model_nature",
        ConfigValue(
            default="linear",
            domain=In(["linear", "quadratic", "aggregated_linear"]),
            doc="Nature of the optimization model: MILP or MIQCQP",
        ),
    )
    config.declare(
        "lazy_constraints",
        ConfigValue(
            default=False,
            domain=Bool,
            doc="If True, some constraints will be added as lazy constraints",
        ),
    )

    # Parameters for optional constraints

    config.declare(
        "perc_wells_in_dac",
        ConfigValue(
            domain=InRange(0, 100),
            doc="Minimum percentage of wells in disadvantaged communities",
        ),
    )
    config.declare(
        "threshold_distance",
        ConfigValue(
            default=10.0,
            domain=NonNegativeFloat,
            doc="Maximum distance [in miles] allowed between wells",
        ),
    )
    config.declare(
        "max_wells_per_owner",
        ConfigValue(
            domain=NonNegativeInt,
            doc="Maximum number of wells per owner",
        ),
    )
    config.declare(
        "max_cost_project",
        ConfigValue(
            domain=NonNegativeFloat,
            doc="Maximum cost per project [in USD]",
        ),
    )
    config.declare(
        "max_num_projects",
        ConfigValue(
            domain=NonNegativeInt,
            doc="Maximum number of projects admissible in a campaign",
        ),
    )

    config.declare(
        "min_wells_in_project",
        ConfigValue(
            domain=NonNegativeInt,
            doc="Minimum number of wells required in a project",
        ),
    )

    config.declare(
        "max_wells_in_project",
        ConfigValue(
            domain=NonNegativeInt,
            doc="Maximum number of wells allowed in a project",
        ),
    )
    config.declare(
        "cluster_method",
        ConfigValue(
            default="Agglomerative",
            domain=In(["Agglomerative", "Louvain"]),
            doc="Method used for clustering the wells",
        ),
    )
    config.declare(
        "threshold_cluster_size",
        ConfigValue(
            default=300,
            domain=NonNegativeInt,
            doc="Maximum size of clusters for Louvain clustering",
        ),
    )
    config.declare(
        "num_nearest_neighbors",
        ConfigValue(
            default=10,
            domain=NonNegativeInt,
            doc=(
                "Number of nearest neighbors to consider adding edges to "
                "while constructing the graph for Louvain clustering"
            ),
        ),
    )
    config.declare(
        "max_resolution",
        ConfigValue(
            default=10,
            domain=NonNegativeFloat,
            doc="Maximum resolution parameter value for Louvain clustering",
        ),
    )
    config.declare(
        "min_budget_usage",
        ConfigValue(
            default=None,
            domain=InRange(0, 100),
            doc="Minimum percentage of the total budget to be used for plugging",
        ),
    )
    config.declare(
        "penalize_unused_budget",
        ConfigValue(
            default=False,
            domain=Bool,
            doc=(
                "If True, unused budget will be penalized in the objective function\n"
                "with suitably chosen weight factor"
            ),
        ),
    )

    # Parameters for computing efficiency metrics

    config.declare(
        "max_num_wells",
        ConfigValue(
            domain=NonNegativeInt,
            doc="Maximum number of wells selected in a project",
        ),
    )
    config.declare(
        "max_dist_to_road",
        ConfigValue(
            domain=NonNegativeFloat,
            doc="Maximum distance to road allowed for selected wells",
        ),
    )
    config.declare(
        "max_elevation_delta",
        ConfigValue(
            domain=NonNegativeFloat,
            doc=(
                "Maximum elevation delta from the closest road "
                "point allowed for selected wells"
            ),
        ),
    )
    config.declare(
        "max_population_density",
        ConfigValue(
            domain=NonNegativeFloat,
            doc="Maximum population density allowed to have near a well",
        ),
    )
    config.declare(
        "max_record_completeness",
        ConfigValue(
            default=1.0,
            domain=NonNegativeFloat,
            doc="Maximum record completeness of a well",
        ),
    )
    config.declare(
        "max_num_unique_owners",
        ConfigValue(
            domain=NonNegativeInt,
            doc="Maximum number of unique owners allowed in a project",
        ),
    )
    config.declare(
        "max_dist_range",
        ConfigValue(
            default=10.0,
            domain=NonNegativeFloat,
            doc="Maximum distance [in miles] allowed between wells",
        ),
    )
    config.declare(
        "max_age_range",
        ConfigValue(
            domain=NonNegativeFloat,
            doc="Maximum age range allowed in a project",
        ),
    )
    config.declare(
        "max_depth_range",
        ConfigValue(
            domain=NonNegativeFloat,
            doc="Maximum depth range allowed in a project",
        ),
    )

    return config


class OptModelInputs:  # pylint: disable=too-many-instance-attributes
    """
    Assembles all the necessary inputs for the optimization model.
    """

    # Using ConfigDict from Pyomo for domain validation.
    CONFIG = model_config()

    @document_kwargs_from_configdict(CONFIG)
    def __init__(self, cluster_mapping=None, **kwargs):
        # pylint: disable=too-many-branches
        # Update the values of all the inputs
        # ConfigDict handles KeyError, other input errors, and domain errors
        LOGGER.info("Processing optimization model inputs.")
        self.config = self.CONFIG(kwargs)

        # Raise an error if the essential inputs are not provided
        wd = self.config.well_data
        if None in [wd, self.config.total_budget, self.config.mobilization_cost]:
            msg = (
                "One or more essential input arguments in [well_data, total_budget, "
                "mobilization_cost] are missing while instantiating the object. "
                "WellData object containing information on all wells, the total budget, "
                "and the mobilization cost are essential inputs for the optimization model. "
            )
            raise_exception(msg, ValueError)

        # Raise an error if priority scores are not calculated.
        if not hasattr(wd.column_names, "priority_score"):
            msg = (
                "Unable to find priority scores in the WellData object. Compute the scores "
                "using the compute_priority_scores method."
            )
            raise_exception(msg, ValueError)

        if self.config.objective_weight_impact < 100:
            if wd.config.efficiency_metrics is None:
                raise_exception(
                    "Weight of efficiency is non-zero. Efficiency metrics object is not specified.",
                    ValueError,
                )
            if wd.config.efficiency_metrics.record_completeness.effective_weight > 0:
                self._compute_record_incompleteness(wd)

        col_names = wd.column_names
        if cluster_mapping is None:
            LOGGER.info("Clustering Data in OptModelInputs")
            # Construct campaign candidates
            # Step 1: Perform clustering, Should distance_threshold be a user argument?
            # Structure: {cluster_1: [index_1, index_2,..], cluster_2: [], ...}
            if self.config.cluster_method == "Agglomerative":
                self.campaign_candidates = perform_agglomerative_clustering(
                    wd, threshold_distance=self.config.threshold_distance
                )
            else:
                self.campaign_candidates = perform_louvain_clustering(
                    wd,
                    threshold_distance=self.config.threshold_distance,
                    threshold_cluster_size=self.config.threshold_cluster_size,
                    nearest_neighbors=self.config.num_nearest_neighbors,
                    max_resolution=self.config.max_resolution,
                )

        else:
            LOGGER.info("Skipping clustering step in OptModelInputs")
            self.campaign_candidates = cluster_mapping
            well_cluster_map = {index: "" for index in wd}
            for cluster, wells in self.campaign_candidates.items():
                for well in wells:
                    well_cluster_map[well] = cluster

            assert "" not in well_cluster_map.values()
            wd.add_new_column_ordered(
                "cluster", "Clusters", list(well_cluster_map.values())
            )

        # Construct owner well count data
        if wd.config.verify_operator_name:
            operator_list = set(wd[col_names.operator_name])
            self.owner_well_count = {owner: [] for owner in operator_list}
            for well in wd:
                # {Owner 1: [(c1, i2), (c1, i3), (c4, i7), ...], ...}
                # Key => Owner name, Tuple[0] => cluster, Tuple[1] => index
                self.owner_well_count[
                    wd.data.loc[well, col_names.operator_name]
                ].append((wd.data.loc[well, col_names.cluster], well))

        # NOTE: Attributes _opt_model and _solver are defined in
        # build_optimization_model and solve_model methods, respectively.
        self.pairwise_metrics = {c: None for c in self.campaign_candidates}
        self._opt_model = None
        self._solver = None
        LOGGER.info("Finished processing optimization model inputs.")

    @property
    def get_total_budget(self):
        """Returns scaled total budget [in million USD]"""
        # Optimization model uses scaled total budget value to avoid numerical issues
        return self.config.total_budget / 1e6

    @property
    def get_mobilization_cost(self):
        """Returns scaled mobilization cost [in million USD]"""
        # Optimization model uses Scaled mobilization costs to avoid numerical issues
        return {
            num_wells: cost / 1e6
            for num_wells, cost in self.config.mobilization_cost.items()
        }

    @property
    def get_max_cost_project(self):
        """Returns scaled maximum cost of the project [in million USD]"""
        if self.config.max_cost_project is None:
            return None

        return self.config.max_cost_project / 1e6

    @property
    def optimization_model(self):
        """Returns the Pyomo optimization model"""
        return self._opt_model

    @property
    def solver(self):
        """Returns the solver object"""
        return self._solver

    def build_optimization_model(self, override_dict=None):
        """Builds the optimization model"""
        LOGGER.info("Beginning to construct the optimization model.")
        self._opt_model = PluggingCampaignModel(self, override_dict)
        LOGGER.info("Completed the construction of the optimization model.")
        return self._opt_model

    def solve_model(self, **kwargs):
        """Solves the optimization"""

        # Adding support for pool search if gurobi_persistent is available
        # To get n-best solutions, pass pool_search_mode = 2 and pool_size = n
        pool_search_mode = kwargs.pop("pool_search_mode", 0)
        pool_size = kwargs.pop("pool_size", 10)

        solver = get_solver(**kwargs)
        self._solver = solver

        # Name attribute is not defined for HiGHS. But it works for all
        # other supported solvers. So, set name as highs, if it does not exist
        solver_name = getattr(solver, "name", "highs")

        if solver_name == "gurobi_persistent":
            # For persistent solvers, model instance need to be set manually
            solver.set_instance(self._opt_model)
            solver.set_gurobi_param("PoolSearchMode", pool_search_mode)
            solver.set_gurobi_param("PoolSolutions", pool_size)

        # Solve the optimization problem
        solver.solve(self._opt_model, tee=kwargs.get("stream_output", True))

        # Return the solution pool, if it is requested
        if solver_name == "gurobi_persistent" and pool_search_mode == 2:
            # Return the solution pool if pool_search_mode is active
            return self._opt_model.get_solution_pool(self._solver)

        # In all other cases, return the optimal campaign
        return self._opt_model.get_optimal_campaign()

    def update_cluster(self, add_widget_return):
        """
        Updates the campaign candidates by changing the cluster numbers for specific wells.

        Parameters
        ---------
        add_widget_return : OverrideAddInfo
            An OverrideAddInfo object which includes information on wells selected to add
            to the existing optimal P&A projects
        """
        existing_clusters = add_widget_return.existing_clusters
        new_clusters = add_widget_return.new_clusters
        wd = self.config.well_data
        col_names = wd.column_names

        if existing_clusters != new_clusters:
            # Remove wells from existing clusters and update owner well counts
            for existing_cluster, existing_wells in existing_clusters.items():
                for well in existing_wells:
                    self.campaign_candidates[existing_cluster].remove(well)
                    self.owner_well_count[
                        wd.data.loc[well, col_names.operator_name]
                    ].remove((existing_cluster, well))

            # Add wells to new clusters and update the well data and owner well counts
            for new_cluster, wells in new_clusters.items():
                for well in wells:
                    self.campaign_candidates[new_cluster].append(well)
                    self.config.well_data.data.loc[well, col_names.cluster] = (
                        new_cluster
                    )
                    self.owner_well_count[
                        wd.data.loc[well, col_names.operator_name]
                    ].append((new_cluster, well))

    @staticmethod
    def _compute_record_incompleteness(wd: WellData):
        """
        Computes the record incompleteness of the given data.
        Higher the score, more of the required data is not available
        for the well.
        Parameters
        ----------
        wd : WellData
            Object containing wells
        """
        data = wd.data[wd.get_flag_columns].sum(axis=1)
        num_columns = max(1, len(wd.get_flag_columns))
        wd.add_new_column_ordered(
            "record_completeness", "Fraction Data Incomplete", data / num_columns
        )

    def compute_efficiency_scaling_factors(self):
        """
        Checks whether scaling factors for efficiency metrics are provided by
        the user or not. If not, computes the scaling factors using the entire
        dataset.

        Parameters
        ----------
        self : OptModelInputs
            OptModelInputs object
        """
        LOGGER.info("Computing scaling factors for efficiency metrics")
        config = self.config
        wd = config.well_data
        eff_metrics = wd.config.efficiency_metrics
        eff_weights = eff_metrics.get_weights

        def set_scaling_factor(metric_name, scale_value):
            """Function for logging warning message"""
            LOGGER.warning(
                f"Scaling factor for the efficiency metric {metric_name} is not "
                f"provided, so it is set to {scale_value}. To modify the "
                f"scaling factor, pass argument max_{metric_name} while instantiating "
                f"the OptModelInputs object."
            )
            setattr(config, "max_" + metric_name, scale_value)

        # Setting a scaling factor for num_wells metric
        if config.max_num_wells is None and eff_weights.num_wells > 0:
            set_scaling_factor("num_wells", 25)

        # Setting a scaling factor for num_unique_owners metric
        if config.max_num_unique_owners is None and eff_weights.num_unique_owners > 0:
            set_scaling_factor("num_unique_owners", 5)

        for metric in WELL_BASED_METRICS:
            if (
                getattr(eff_weights, metric, 0) > 0
                and getattr(config, "max_" + metric) is None
            ):
                # Metric is chosen, but the scaling factor is not specified
                scale_value = wd[getattr(eff_metrics, metric).data_col_name].max()
                if np.isclose(scale_value, 0):
                    LOGGER.warning(
                        f"Scaling factor for {metric} is close to 0. Setting it to 1."
                    )
                    scale_value = 1
                set_scaling_factor(metric, scale_value)

        if sum(getattr(eff_weights, metric, 0) for metric in WELL_PAIR_METRICS) == 0:
            # None of the pairwise metrics are selected, so return
            return

        # Append the pairwise metrics to the model
        for c in self.campaign_candidates:
            self.pairwise_metrics[c] = get_pairwise_metrics(
                wd, self.campaign_candidates[c]
            )

        for metric in WELL_PAIR_METRICS:
            if (
                getattr(eff_weights, metric, 0) > 0
                and getattr(config, "max_" + metric) is None
            ):
                # Metric is chosen, but the scaling factor is not specified
                scale_value = max(
                    self.pairwise_metrics[c][metric].max()
                    for c in self.campaign_candidates
                )
                if np.isclose(scale_value, 0):
                    LOGGER.warning(
                        f"Scaling factor for {metric} is close to 0. Setting it to 1."
                    )
                    scale_value = 1
                set_scaling_factor(metric, scale_value)

    def has_efficiency_scaling_factors(self):
        """
        checks if all efficiency metrics have scaling factors

        Parameters
        ----------
        self : OptModelInputs
            OptModelInputs object

        Returns
        ----------
        Bool : False if scaling factor for any metric does not exist, True otherwise
        """
        config = self.config
        wd = config.well_data
        eff_metrics = wd.config.efficiency_metrics
        for metric in eff_metrics:
            if (
                metric.effective_weight > 0
                and getattr(config, "max_" + metric.name) is not None
            ):
                pass
            else:
                return False
        return True
