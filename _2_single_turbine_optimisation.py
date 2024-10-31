from _0_AviemoreSite import *

from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
from py_wake.site import UniformWeibullSite
from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine
from py_wake.utils.gradients import autograd
from py_wake.deficit_models.utils import *

from topfarm import TopFarmProblem
from topfarm.plotting import XYPlotComp
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm.constraint_components.spacing import SpacingConstraint


def evaluate_turbine_single(turbine_data, theta, plot_options=None, optimize=True):
    """
    Evaluate a single wind turbine's placement and potential energy production.

    Parameters:
        turbine_data (dict): Data about the turbine specifications.
        theta (float): Rotation angle for the wind farm grid.
        plot_options (dict, optional): Options to control which plots to display.
        optimize (bool, optional): Whether to run optimization on turbine placement.

    Returns:
        tuple: Annual Energy Production (AEP) with and without wake losses, and the number of turbines.
    """
    # Access the first turbine's data in turbine_data
    turbine_name = list(turbine_data.keys())[0]
    data = turbine_data[turbine_name]

    print(f"Evaluating turbine: {turbine_name}")

    # Extract turbine specifications
    D = data.get('rotor_diameter_m')
    hub_height = data.get('hub_height_m')
    rated_power = data.get('rated_power_kW')
    ws_cutin = data.get('cut_in_wind_speed_ms', 1)
    ws_cutout = data.get('cut_out_wind_speed_ms', None)

    # Initialize wind turbine model
    wt = GenericWindTurbine(
        name=turbine_name,
        diameter=D,
        hub_height=hub_height,
        power_norm=rated_power,
        ws_cutin=ws_cutin,
        ws_cutout=ws_cutout
    )

    # Initialize wind site data
    weibull_direction_data = wind_initialisation(hub_height)
    site = UniformWeibullSite(
        p_wd=weibull_direction_data['Frequency'].tolist(),
        a=weibull_direction_data['Weibull_A'].tolist(),
        k=weibull_direction_data['Weibull_k'].tolist(),
        ti=0.1  # Turbulence intensity
    )

    # Initialize wind farm wake model
    windFarmModel = Bastankhah_PorteAgel_2014(site, wt, k=0.0324555)

    # Create initial wind farm grid with rotation based on theta
    wt_x, wt_y, n_turbines, n_rows, n_columns, total_cabling_length, column_cabling_lengths, row_cabling_length = create_wind_farm_grid(D, theta)

    def topfarm_optimisation():
        """
        Run TopFarm optimization to maximize AEP within boundary constraints.

        Returns:
            tuple: Optimized x and y coordinates.
        """
        boundary_to_be_fixed = [(274591, 814726), (276608, 821842), (277702, 813671), (280428, 820253),
                                (277637, 821425),
                                (278646, 820998), (279816, 818802), (278893, 816509), (278048, 814537),
                                (276752, 813997),
                                (275524, 814419), (276310, 820845), (275760, 818867), (275181, 816879)]
        initial = np.column_stack((wt_x, wt_y))

        def adjust_and_order_boundary(boundary, initial_wt):
            def order_polygon_points(points):
                # Extract x and y coordinates from points
                x_coords = np.array([p[0] for p in points])
                y_coords = np.array([p[1] for p in points])

                # Calculate centroid of the points
                centroid_x = np.mean(x_coords)
                centroid_y = np.mean(y_coords)

                # Compute the angles relative to the centroid for counterclockwise sorting
                angles = np.arctan2(y_coords - centroid_y, x_coords - centroid_x)
                sorted_indices = np.argsort(angles)

                # Return points sorted by the calculated angles
                return [points[i] for i in sorted_indices]

            # Find the point with the minimum y value
            min_point = min(boundary, key=lambda point: point[1])
            min_y = min_point[1]  # Minimum y-coordinate
            min_x = min_point[0]  # Corresponding x-coordinate

            # Shift all points so that the lowest point becomes (0, 0)
            shifted_boundary = [
                (x - min_x, y - min_y) for x, y in boundary
            ]

            shifted_wx_wy = [
                (wx - min_x, wy - min_y) for wx, wy in initial_wt
            ]

            # Order points counterclockwise around the centroid
            ordered_boundary = order_polygon_points(shifted_boundary)

            return ordered_boundary, np.array(shifted_wx_wy), min_x, min_y

        def translate_coordinates_back(optimized_x, optimized_y, min_x, min_y):
            original_x = optimized_x + min_x
            original_y = optimized_y + min_y
            return original_x, original_y

        def aep_fun(x, y):
            aep = windFarmModel(x, y).aep().sum()
            return aep

        # Adjusted and ordered boundary with additional shifts
        boundary, initial, min_x, min_y = adjust_and_order_boundary(boundary_to_be_fixed, initial)

        design_vars = dict(zip('xy', (initial[:, :2]).T))

        daep = windFarmModel.aep_gradients(gradient_method=autograd, wrt_arg=['x', 'y'])

        aep_comp = CostModelComponent(input_keys=['x', 'y'],
                                      n_wt=n_turbines,
                                      cost_function=aep_fun,
                                      cost_gradient_function=daep,
                                      output_keys=("aep", 0),
                                      output_unit="GWh",
                                      maximize=True,
                                      objective=True)

        problem = TopFarmProblem(design_vars=design_vars,
                                 constraints=[XYBoundaryConstraint(boundary, 'polygon'),
                                              SpacingConstraint(3 * D)],
                                 n_wt=n_turbines,
                                 cost_comp=aep_comp,
                                 driver=EasyScipyOptimizeDriver(optimizer='COBYLA', maxiter=4000),
                                 plot_comp=XYPlotComp(),
                                 expected_cost=0.1,
                                 )

        # Optimization
        cost, state, recorder = problem.optimize(disp=True)
        plt.close()

        # Extract x and y coordinates from the optimized state
        optimized_x = state['x']
        optimized_y = state['y']

        # Translate coordinates back to the original reference frame
        original_x, original_y = translate_coordinates_back(optimized_x, optimized_y, min_x, min_y)

        return original_x, original_y

    # Decide whether to optimize or use initial coordinates
    if optimize:
        x, y = topfarm_optimisation()
    else:
        x, y = wt_x, wt_y

    # Save the x and y coordinates to a DataFrame
    optimal_windfarm_coordinates = {
        "Turbine Number": [f"Turbine {i + 1}" for i in range(len(x))],
        "x-coordinate": x,
        "y-coordinate": y
    }

    optimal_windfarm_coordinates = pd.DataFrame(optimal_windfarm_coordinates)

    # Save the DataFrame to a CSV file
    optimal_windfarm_coordinates.to_csv("Results/optimal_windfarm_coordinates.csv", index=False)

    # Calculate the Annual Energy Production (AEP)
    sim_res = windFarmModel(x, y)
    aep = sim_res.aep().sum().item()
    aep_no_wake = sim_res.aep(with_wake_loss=False).sum().item()

    # Define plot options, keeping only the specified plots
    plot_options = plot_options or {
        'power_by_direction': False,
        'flow_map': False,
        'wind_farm_layout': False,
    }

    # Generate plots based on user-defined options
    if plot_options.get('power_by_direction', False):
        plot_power_by_wind_direction(sim_res, wt_idx=21)

    if plot_options.get('flow_map', False):
        plot_flow_map(sim_res, wind_direction=210, wind_speed=12)

    print(n_turbines)
    return aep, aep_no_wake, n_turbines


def plot_power_by_wind_direction(sim_res, wt_idx):
    # Plot the power output across wind directions for a specific turbine
    power = sim_res.Power.sel(wt=wt_idx)
    wd = np.deg2rad(power.wd)  # Convert wind direction to radians for polar plot

    plt.figure()
    ax = plt.subplot(111, polar=True)
    ax.plot(wd, power, label=f'Turbine {wt_idx}')
    ax.set_title(f'Power vs Wind Direction - Turbine {wt_idx}')
    ax.set_xlabel('Wind Direction (degrees)')
    plt.legend()
    plt.show()


def plot_flow_map(sim_res, wind_direction=210, wind_speed=12):
    # Plot the flow map for the wind farm at a specific wind direction and wind speed
    flow_map = sim_res.flow_map(wd=wind_direction, ws=wind_speed)
    flow_map.plot_wake_map()
    plt.title(f'Flow Map at {wind_direction} degrees and {wind_speed} m/s')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.savefig('Figures/FlowMapOpt.png')  # Save the figure
    plt.show()


# _____ Initialise Parameters and Run Evaluation _____
theta = np.radians(210)
turbine_data = load_turbine_data("Data/Best_Turbine.xlsx")  # Function to load turbine data

# Choose which plots to show
plot_options = {
    'power_by_direction': False,
    'flow_map': True,
}

aep, aep_no_wake, n_turbines = evaluate_turbine_single(turbine_data, theta, plot_options, optimize=True)
print(f"AEP: {aep}, AEP without wake loss: {aep_no_wake}, Number of Turbines: {n_turbines}")

