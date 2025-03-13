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
from haversine import Unit, haversine


def distance_campaign(
    well_data,
    budget,
    mobilization_costs,
    max_wells_in_project,
    threshold_distance,
    top_wells,
):
    # Assuming gas_wells is a DataFrame with 'Latitude', 'Longitude', and other well information
    data = well_data.data
    cn = well_data.column_names

    # Create a list of coordinates from gas_wells data (using the DataFrame index as identifiers)
    coordinates = list(zip(data[cn.latitude], data[cn.longitude]))
    cluster_map = {}
    index_map = {}
    optimal_campaign = {}
    for idx, _ in enumerate(coordinates):
        index_map[idx] = int(data.index[idx])
    cluster = 1
    # well_fix = {}
    well_plug_list = []
    used_budget = 0
    unused_budget = budget
    plugging_cost = {}

    index = 0  # Initialize the index
    while unused_budget >= mobilization_costs[1]:
        row = top_wells.data.iloc[
            index
        ]  # Access the row by index from the `WellData` object

        if unused_budget < mobilization_costs[1]:
            break
        
        if unused_budget > mobilization_costs[max_wells_in_project]:
            wells_in_project = max_wells_in_project
        else:
            av_n_well = [
                n_wells
                for n_wells, cost in mobilization_costs.items()
                if cost < unused_budget
            ]
            wells_in_project = max(av_n_well)

        # Check if this well index has already been assigned to a cluster
        if not any(
            index in wells for wells in cluster_map.values()
        ):  # Check if the well has already been assigned to a cluster
            ref_lat = row["Latitude"]
            ref_lon = row["Longitude"]
            ref_point = (ref_lat, ref_lon)  # Creating the reference point as a tuple

            distances = (
                []
            )  # List to store the distances and their corresponding coordinates

            # Loop through each point in coordinates and calculate distance
            for idx, point in enumerate(coordinates):
                distance = haversine(
                    ref_point, point, unit=Unit.MILES
                )  # Get distance in miles
                if distance < threshold_distance:
                    distances.append(
                        (distance, index_map[idx])
                    )  # Store only the distance and well index

            # Sort the distances list by the distance value (ascending order)
            distances.sort(
                key=lambda x: x[0]
            )  # Sort by distance (first element of the tuple)

            # Get the 10 closest points (excluding the reference well itself)
            closest_points = [
                well_index for _, well_index in distances[:wells_in_project]
            ]  # Exclude the reference well (1st element)

            # Store in cluster_map (using the original DataFrame index)
            cluster_map[cluster] = closest_points
            optimal_campaign[cluster] = closest_points
            well_plug_list += closest_points
            n_well = len(closest_points)
            cost = mobilization_costs[n_well]
            plugging_cost[cluster] = cost / 1e6
            used_budget = used_budget + cost
            unused_budget = budget - used_budget

        cluster += 1  # Increment the cluster count
        index += 1  # Move to the next row in top_gas_wells

    for index, row in top_wells.data.iterrows():
        # If the well index is not present in any cluster, assign it to cluster 0
        if not any(index in wells for wells in cluster_map.values()):
            if 0 not in cluster_map:
                cluster_map[0] = []
            cluster_map[0].append(index)

    plug_list = []
    for _, well_list in cluster_map.items():
        plug_list += well_list
        # prevent duplication in plug_list
        plug_list = list(set(plug_list))
        well_data_sub = well_data._construct_sub_data(plug_list)
    return well_data_sub, optimal_campaign, plugging_cost, cluster_map
