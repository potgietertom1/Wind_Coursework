from _0_AviemoreSite import *

from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
from py_wake.site import UniformWeibullSite
from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine


# Function to evaluate AEP based on turbine diameter and hub height
def evaluate(D, hub_height, theta):
    # Initialise wind turbine
    wt = GenericWindTurbine(
        name="Test_3MW",
        diameter=D,
        hub_height=hub_height,
        power_norm=3000,
        ws_cutin=3,
        ws_cutout=25
    )

    # Initialise wind data
    weibull_direction_data = wind_initialisation(hub_height)
    site = UniformWeibullSite(
        p_wd=weibull_direction_data['Frequency'].tolist(),
        a=weibull_direction_data['Weibull_A'].tolist(),
        k=weibull_direction_data['Weibull_k'].tolist(),
        ti=0.1
    )

    # Initialise wind farm wake model
    windFarmModel = Bastankhah_PorteAgel_2014(site, wt, k=0.0324555)

    # Rotate the wind farm grid based on the current angle
    wt_x, wt_y, n_turbines, n_rows, n_columns, total_cabling_length, column_cabling_lengths, row_cabling_length = create_wind_farm_grid(D, theta)
    print(total_cabling_length, column_cabling_lengths, row_cabling_length)

    # Calculate the Annual Energy Production (AEP)
    sim_res = windFarmModel(wt_x, wt_y)
    aep = sim_res.aep().sum().item()
    aep_no_wake = sim_res.aep(with_wake_loss=False).sum().item()

    # Calculate the percentage loss due to wake effects
    loss_to_wake_effects = (aep_no_wake - aep) / aep * 100 if aep > 0 else None

    def plot_flow_map(sim_res, wind_direction=210, wind_speed=12):
        # Plot the flow map for the wind farm at a specific wind direction and wind speed
        flow_map = sim_res.flow_map(wd=wind_direction, ws=wind_speed)
        flow_map.plot_wake_map()
        plt.title(f'Flow Map at {wind_direction} degrees and {wind_speed} m/s')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.savefig('Figures/WakeMap')  # Save the figure
        plt.show()

    # plot_flow_map(sim_res, wind_direction=210, wind_speed=12)

    return sim_res, aep, n_turbines, loss_to_wake_effects


# --- Plotting Functions ---

# Plot 1: Number of Turbines with Diameter
def plot_number_of_turbines_and_aep_with_diameter(diameters, hub_height, theta):
    # Calculate number of turbines, AEP, and wake loss for each diameter using the evaluate function
    n_turbines = []
    aep_values = []
    wake_losses = []

    for D in diameters:
        sim_res, aep, n_turbine, loss_to_wake_effects = evaluate(D, hub_height, theta)
        n_turbines.append(n_turbine)
        aep_values.append(aep)
        wake_losses.append(loss_to_wake_effects)

    # Plot number of turbines with diameter
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot number of turbines
    ax1.plot(diameters, n_turbines, marker='o', color='b', label='Number of Turbines')
    ax1.set_xlabel('Diameter (m)')
    ax1.set_ylabel('Number of Turbines', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid()

    # Create a secondary y-axis for AEP
    ax2 = ax1.twinx()
    ax2.plot(diameters, aep_values, marker='x', color='r', label='AEP')
    ax2.set_ylabel('Annual Energy Production (GWh)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Add wake loss percentage as bubbles above each AEP data point
    for i, (D, aep, loss) in enumerate(zip(diameters, aep_values, wake_losses)):
        if loss is not None:  # Only annotate if a valid wake loss value exists
            ax2.annotate(f'{loss:.1f}%',
                         (D, aep),
                         textcoords="offset points",
                         xytext=(0, 10),  # Position text slightly above the point
                         ha='center',
                         color='darkred',
                         fontsize=9)

    # Title and save the plot
    plt.title('Number of Turbines, AEP, and Wake Loss with Diameter')
    fig.tight_layout()
    plt.savefig('Figures/D-n_turbines_and_AEP_with_wake_loss.png')  # Save the figure
    plt.show()

# Plot 2: Power Output with Hub Height
def plot_power_output_with_hub_height(hub_heights, theta):
    power_outputs = []
    for hub_height in hub_heights:
        aep, _ = evaluate(D, hub_height, theta)
        power_outputs.append(aep)

    plt.figure(figsize=(10, 6))
    plt.plot(hub_heights, power_outputs, marker='o')
    plt.title('Power Output with Hub Height')
    plt.xlabel('Hub Height (m)')
    plt.ylabel('Annual Energy Production (AEP) [Wh]')
    plt.grid()
    plt.savefig('Figures/AEP-HubHeight.png')  # Save the figure
    plt.show()


# Plot 3: Power Output with Theta
def plot_power_output_with_theta(hub_height, thetas):
    power_outputs = []
    for theta in thetas:
        aep, _ = evaluate(D, hub_height, np.radians(theta))
        power_outputs.append(aep)

    plt.figure(figsize=(10, 6))
    plt.plot(thetas, power_outputs, marker='o')
    plt.title('Power Output with Theta')
    plt.xlabel('Theta (degrees)')
    plt.ylabel('Annual Energy Production (AEP) [GWh]')
    plt.grid()
    plt.savefig('Figures/AEP-Theta.png')  # Save the figure
    plt.show()


def plot_weibull_distributions(turbine_data):
    """
    Plot the Weibull distributions of wind speeds for each turbine.
    """
    plt.figure(figsize=(10, 6))

    # Loop through each turbine to calculate and plot its Weibull distribution
    for turbine_name, data in turbine_data.items():
        # Get the Weibull parameters for the turbine
        hub_height = data.get('hub_height_m')
        weibull_direction_data = wind_initialisation(hub_height)  # Adjust this function to get Weibull params

        # Average shape and scale values (we're assuming these are provided for each direction and averaging)
        k_values = weibull_direction_data['Weibull_k'].values
        a_values = weibull_direction_data['Weibull_A'].values
        k_mean = np.mean(k_values)
        a_mean = np.mean(a_values)

        # Define a range of wind speeds to plot the Weibull PDF
        wind_speeds = np.linspace(0, 30, 100)
        weibull_pdf = weibull_min.pdf(wind_speeds, k_mean, scale=a_mean)

        # Plot the Weibull PDF for the turbine
        plt.plot(wind_speeds, weibull_pdf, label=f"{turbine_name} (Hub Height: {hub_height}m)")

    # Add plot details
    plt.title("Weibull Distribution of Wind Speeds for Each Turbine")
    plt.xlabel("Wind Speed (m/s)")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid()
    plt.show()

def plot_weibull_distribution_with_hub_height(hub_heights):
    plt.figure(figsize=(10, 6))

    # Iterate over each hub height and calculate Weibull distribution
    for hub_height in hub_heights:
        # Get Weibull parameters for the current hub height
        weibull_data = wind_initialisation(hub_height)
        A = weibull_data['Weibull_A'].mean()  # Scale parameter
        k = weibull_data['Weibull_k'].mean()  # Shape parameter

        # Generate wind speeds for the Weibull distribution curve
        wind_speeds = np.linspace(0, 30, 300)
        weibull_pdf = weibull_min.pdf(wind_speeds, k, scale=A)

        # Plot the Weibull distribution for this hub height
        plt.plot(wind_speeds, weibull_pdf, label=f'Hub Height: {hub_height} m')

    # Add labels and legend
    plt.title('Weibull Distribution for Different Hub Heights')
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid()
    plt.savefig('Figures/Weibull_Distribution_HubHeights.png')  # Save the figure
    plt.show()


def plot_wake_maps_for_diameters(D1, D2, hub_height, theta, wind_direction=210, wind_speed=12):
    # Initialize wind turbines with different diameters
    wt1 = GenericWindTurbine(
        name="Test_3MW_D1",
        diameter=D1,
        hub_height=hub_height,
        power_norm=3000,
        ws_cutin=3,
        ws_cutout=25
    )

    wt2 = GenericWindTurbine(
        name="Test_3MW_D2",
        diameter=D2,
        hub_height=hub_height,
        power_norm=3000,
        ws_cutin=3,
        ws_cutout=25
    )

    # Initialize wind data for site
    weibull_direction_data = wind_initialisation(hub_height)
    site = UniformWeibullSite(
        p_wd=weibull_direction_data['Frequency'].tolist(),
        a=weibull_direction_data['Weibull_A'].tolist(),
        k=weibull_direction_data['Weibull_k'].tolist(),
        ti=0.1
    )

    # Initialize wake models for both turbine configurations
    windFarmModel1 = Bastankhah_PorteAgel_2014(site, wt1, k=0.0324555)
    windFarmModel2 = Bastankhah_PorteAgel_2014(site, wt2, k=0.0324555)

    # Get turbine positions for each configuration
    wt_x1, wt_y1, _, _, _, _ = create_wind_farm_grid(D1, theta)
    wt_x2, wt_y2, _, _, _, _ = create_wind_farm_grid(D2, theta)

    # Simulate AEP for each configuration to get wake maps
    sim_res1 = windFarmModel1(wt_x1, wt_y1)
    sim_res2 = windFarmModel2(wt_x2, wt_y2)

    # Generate flow maps for the specified wind direction and speed
    flow_map1 = sim_res1.flow_map(wd=wind_direction, ws=wind_speed)
    flow_map2 = sim_res2.flow_map(wd=wind_direction, ws=wind_speed)

    # Plot wake maps side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot the first wake map for D=100
    flow_map1.plot_wake_map(ax=ax1)
    ax1.set_title(f'Wake Map for Diameter {D1} m at {wind_direction} degrees, {wind_speed} m/s')
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')

    # Plot the second wake map for D=50
    flow_map2.plot_wake_map(ax=ax2)
    ax2.set_title(f'Wake Map for Diameter {D2} m at {wind_direction} degrees, {wind_speed} m/s')
    ax2.set_xlabel('x [m]')
    ax2.set_ylabel('y [m]')

    plt.tight_layout()
    plt.savefig('Figures/Side_by_Side_Wake_Maps.png')  # Save the figure
    plt.show()

def plot_power_output_with_thetas(D, hub_height, thetas, labels=None):
    """
    Plot the power output (AEP) of a single turbine at various wind farm rotation angles (theta),
    and then plot wake maps for the optimal, worst, and max AEP directions side by side.
    """
    # Set the seaborn style
    sns.set(style='whitegrid')

    power_outputs = []
    total_aep = []  # To store total AEP for all turbines
    capacity_factors = []  # To store capacity factors
    rated_power = 3150  # Rated power in kW

    # Calculate power output metrics for each theta
    for theta in thetas:
        sim_res, aep, _, n_turbines = evaluate(D, hub_height, theta)
        capacity_factor = (aep / (rated_power / 1000000 * n_turbines * 8760)) * 100  # Calculate capacity factor
        power_outputs.append(aep / n_turbines)  # AEP per turbine
        total_aep.append(aep)  # Store total AEP for all turbines
        capacity_factors.append(capacity_factor)  # Store capacity factor

    # Plot AEP per turbine over theta
    plt.figure(figsize=(10, 6))
    plt.plot(np.degrees(thetas), power_outputs, marker='o', label='AEP per Turbine', color='blue')

    # Find indices of the max and min power outputs per turbine
    max_index = np.argmax(power_outputs)
    min_index = np.argmin(power_outputs)

    # Annotate max capacity factor per turbine
    plt.annotate(f'Max Capacity Factor: {capacity_factors[max_index]:.1f}%',
                 (np.degrees(thetas[max_index]), power_outputs[max_index]),
                 textcoords="offset points",
                 xytext=(0, 10),
                 ha='center', color='green',
                 bbox=dict(facecolor='white', edgecolor='green', boxstyle='round,pad=0.3'))

    # Annotate min capacity factor per turbine
    plt.annotate(f'Min Capacity Factor: {capacity_factors[min_index]:.1f}%',
                 (np.degrees(thetas[min_index]), power_outputs[min_index]),
                 textcoords="offset points",
                 xytext=(0, -15),
                 ha='center', color='red',
                 bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.3'))

    # Calculate the index of the maximum total AEP
    ultimate_max_index = np.argmax(total_aep)

    # Annotate the ultimate max AEP (not per turbine)
    plt.annotate(f'Ultimate Max AEP: {total_aep[ultimate_max_index]:.1f} GWh',
                 (np.degrees(thetas[ultimate_max_index]), power_outputs[ultimate_max_index]),
                 textcoords="offset points",
                 xytext=(0, 10),
                 ha='center', color='orange',
                 bbox=dict(facecolor='white', edgecolor='orange', boxstyle='round,pad=0.3'))

    plt.title('Power Output Per Turbine with Theta', fontsize=16)
    plt.xlabel('Theta (degrees)', fontsize=14)
    plt.ylabel('Annual Energy Production (AEP) [GWh]', fontsize=14)
    plt.xlim(min(np.degrees(thetas)), max(np.degrees(thetas)))  # Set x-axis limits
    plt.ylim(min(power_outputs) * 0.9, max(power_outputs) * 1.1)  # Set y-axis limits slightly above max total AEP
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)  # Dotted grid for better visibility
    plt.legend(fontsize=12)
    plt.savefig('Figures/AEP per Turbine - Theta.png')  # Save the figure
    plt.show()
    plt.close()


    theta_max = thetas[max_index]
    theta_min = thetas[min_index]
    theta_ultimate_AEP = thetas[ultimate_max_index]

    # Directions and labels for subplots
    critical_thetas = [theta_max, theta_min, theta_ultimate_AEP]
    default_labels = ['Max Capacity Factor', 'Min Capacity Factor', 'Max Ultimate AEP']

    # Set up figure with equal width for each plot
    fig, axs = plt.subplots(1, 3, figsize=(14, 6))  # 1 row, 2 columns

    for i, (theta, ax) in enumerate(zip(critical_thetas, axs)):
        # Evaluate the turbine at the current critical theta to get flow map
        sim_res, _, _, _ = evaluate(D, hub_height, theta)

        # Plot the wake map for each subplot
        flow_map = sim_res.flow_map(wd=210, ws=12)
        im = flow_map.plot_wake_map(ax=ax)  # Plot without no_color_bar

        # Remove the color bar for the first plot
        if i < len(critical_thetas) - 1:
            for collection in ax.collections:
                collection.colorbar.remove()  # Remove color bar for this subplot

        # Set title for each subplot
        title = labels[i] if labels and i < len(labels) else default_labels[i]
        ax.set_title(f'{title} (Theta: {np.degrees(theta):.1f}°)', fontsize=14)

        # Remove x and y tick labels and set equal aspect ratio
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')


    # Adjust layout and save the final figure
    plt.tight_layout()
    plt.savefig('Figures/KeyDirectionWakeMaps.png')
    plt.show()


# --- Main Execution ---

# Parameters
theta = np.radians(210)
D = 100  # Fixed diameter for certain plots
h = 100
hub_heights = np.arange(60, 151, 10)  # Hub heights from 60 to 120
diameters = np.arange(50, 151, 10)  # Diameters from 50 to 150
thetas = np.arange(0, 361, 180)  # Theta from 0 to 360 degrees (in degrees)

# Generate plots
# plot_number_of_turbines_and_aep_with_diameter(diameters, h, theta)
# plot_power_output_with_hub_height(hub_heights, theta)
# plot_power_output_with_theta(h,thetas)
# plot_weibull_distribution_with_hub_height(hub_heights)
# plot_power_output_with_thetas(D,h,thetas)


# plot_wake_maps_for_diameters(D1=150, D2=50, hub_height=100, theta=np.radians(210))
evaluate(108,100,theta)