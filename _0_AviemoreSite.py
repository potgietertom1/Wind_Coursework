import numpy as np
from matplotlib.path import Path
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import weibull_min
import seaborn as sns

# _____ Elevation Constants _____
elevation_mountain = 595
elevation_weather_station = 220

# _____ Hellman Scaling Stuff _____
h_ref  = 10
a = 0.34

# Starting bounding box coordinates found from GIS work
starting_x = np.array([274591, 276608, 277702, 280428])
starting_y = np.array([814726, 821842, 813671, 820253])

# Financial Constants
cable_cost = 80             # GBP per m (50 for cable 30 for installation)
CFD_strike_price = 50       # Assuming a CFD strike price of 50 GBP/MWh
discount_rate = 0.08        # Assuming an 8% discount rate
project_lifetime = 20       # Assuming a project lifetime of 20 years
inflation_rate = 0.02       # 2% inflation a year
depreciation_rate = 0.005   # 0.5% depreciation a year


def load_turbine_data(filepath):
    """Load turbine data from an Excel file and return a dictionary of turbine information."""

    # Load the Excel file without headers
    df = pd.read_excel(filepath, header=None)

    # Transpose the dataframe and set the first row as the column headers
    df_transposed = df.set_index(0).transpose()

    # Initialize a dictionary to store turbine data
    turbines_dict = {}

    # Extract unique turbine names, dropping any missing values
    turbine_names = df_transposed["Turbine Name"].dropna().unique()
    print("Turbine Names:", turbine_names)

    # Iterate through each turbine name and collect its data
    for idx, turbine_name in enumerate(turbine_names):
        # Fetch the row corresponding to this turbine
        turbine_data = df_transposed.iloc[idx]

        # Store turbine data in the dictionary
        turbines_dict[turbine_name] = {
            "manufacturer": turbine_data["Manufacturer"],
            "rated_power_kW": turbine_data["Rated Power (kW)"],
            "rotor_diameter_m": turbine_data["Rotor Diameter (m)"],
            "swept_area_m2": turbine_data.get("Swept Area (m^2)", None),
            "hub_height_m": turbine_data["Hub Height"],
            "cut_in_wind_speed_ms": turbine_data["Cut-in Wind Speed (m/s)"],
            "rated_wind_speed_ms": turbine_data["Rated Wind Speed (m/s)"],
            "cut_out_wind_speed_ms": turbine_data["Cut-out Wind Speed (m/s)"],
            "number_of_blades": turbine_data.get("Number of Blades", None),
            "grid_frequency_Hz": turbine_data.get("Grid Frequency (Hz)", None),
            "cost_per_turbine": turbine_data.get("Cost", None)
        }

    return turbines_dict

def wind_initialisation(hub_height):
    """
    This function processes wind data from an Excel file, converts wind speed from knots to m/s,
    adjusts the wind speed to account for the mountain height plus the hub height, calculates Weibull parameters
    (scale A and shape k), and calculates the frequency of each wind direction.

    Parameters:
    - hub_height (float): The hub height of the wind turbine (in meters).
    - file_path (str): The path to the Excel file containing wind data.
    - h_ref (float): The reference height used in the calibration (default is 10 meters).
    - a (float): Power law exponent used for wind speed adjustment.

    Returns:
    - pd.DataFrame: A DataFrame containing Weibull parameters and frequency of wind directions.
    """

    # Load the Excel file
    wind_data = pd.read_excel('Data/Wind_Data.xlsx')

    # Conversion factor from knots to meters per second
    knots_to_mps = 0.514444

    # Add a new column for wind speed in meters per second
    wind_data['Wind_Speed_mps'] = wind_data['Wind - Mean Speed (knots)'] * knots_to_mps

    # Replace 360° with 0° in the wind direction column
    wind_data['Wind - Mean Direction'] = wind_data['Wind - Mean Direction'].replace(360, 0)

    # Adjust wind speed using mountain height + hub height calibration formula
    h = elevation_mountain - elevation_weather_station + hub_height
    wind_data['Wind_Speed_mps_adjusted'] = wind_data['Wind_Speed_mps'] * (h / h_ref) ** a

    # Clean data by removing NaN values
    wind_cleaned = wind_data.dropna()

    # Function to calculate Weibull parameters (Shape k and Scale A) for wind speed data
    def fit_weibull(speed_data):
        params = weibull_min.fit(speed_data, floc=0)
        scale_param = params[2]  # Scale parameter (A)
        shape_param = params[0]  # Shape parameter (k)
        return scale_param, shape_param

    # Calculate Weibull parameters for each wind direction
    weibull_params = wind_cleaned.groupby('Wind - Mean Direction')['Wind_Speed_mps_adjusted'].apply(
        fit_weibull).reset_index()
    weibull_params.columns = ['Wind - Mean Direction', 'Weibull Parameters']

    # Separate scale (A) and shape (k) parameters into distinct columns
    weibull_params[['Weibull_A', 'Weibull_k']] = pd.DataFrame(weibull_params['Weibull Parameters'].tolist(),
                                                              index=weibull_params.index)
    weibull_params = weibull_params.drop(columns='Weibull Parameters')

    # Calculate the frequency of each wind direction
    direction_frequency = wind_cleaned['Wind - Mean Direction'].value_counts(normalize=True).reset_index()
    direction_frequency.columns = ['Wind - Mean Direction', 'Frequency']

    # Merge Weibull parameters with wind direction frequencies
    weibull_direction_data = pd.merge(weibull_params, direction_frequency, on='Wind - Mean Direction')

    return weibull_direction_data

def create_wind_farm_grid(D, theta):
    # Compute the center of the bounding box for rotation
    center_x = np.mean(starting_x)
    center_y = np.mean(starting_y)

    # Function to rotate points by a given angle theta around a center point
    def rotate_points(x, y, center_x, center_y, theta):
        # Shift points to the origin for rotation
        x_shifted = x - center_x
        y_shifted = y - center_y

        # Apply rotation matrix
        x_rot = x_shifted * np.cos(theta) - y_shifted * np.sin(theta)
        y_rot = x_shifted * np.sin(theta) + y_shifted * np.cos(theta)

        # Shift points back after rotation
        x_final = x_rot + center_x
        y_final = y_rot + center_y

        return x_final, y_final

    # Ensure that the coordinates are properly ordered to form a polygon
    def order_polygon_points(x_coords, y_coords):
        # Calculate the centroid of the points
        centroid_x = np.mean(x_coords)
        centroid_y = np.mean(y_coords)

        # Compute the angles of the points relative to the centroid
        angles = np.arctan2(y_coords - centroid_y, x_coords - centroid_x)

        # Sort points by angle in counterclockwise order
        sorted_indices = np.argsort(angles)
        return x_coords[sorted_indices], y_coords[sorted_indices]

    # Order the polygon points correctly
    ordered_x, ordered_y = order_polygon_points(starting_x, starting_y)

    # Calculate the correct spacing of the turbines
    spacing_x = 7 * D  # front-to-back spacing is 7D
    spacing_y = 5 * D  # side-to-side spacing is 5D

    # Increase grid size to overfill, ensuring coverage of the top-left region too
    extra_space_factor = 3  # Factor to overfill the grid
    width = np.max(ordered_x) - np.min(ordered_x)
    height = np.max(ordered_y) - np.min(ordered_y)
    num_x_overfill = int((width / spacing_x) * extra_space_factor)  # Overfill in x direction
    num_y_overfill = int((height / spacing_y) * extra_space_factor)  # Overfill in y direction

    # Generate an overfilled grid centered around the centroid of the polygon
    turbine_x_overfill, turbine_y_overfill = np.meshgrid(
        np.arange(-num_x_overfill // 3, num_x_overfill) * spacing_x,
        np.arange(-num_y_overfill // 3, num_y_overfill) * spacing_y
    )

    # Flatten the grid for easier manipulation
    turbine_x_overfill = turbine_x_overfill.flatten()
    turbine_y_overfill = turbine_y_overfill.flatten()

    # Center the overfilled turbines around the center of the bounding box
    turbine_x_overfill += np.min(ordered_x)
    turbine_y_overfill += np.min(ordered_y)

    # Rotate the overfilled turbine positions to face 210 degrees into the wind
    turbine_x_overfill_rot, turbine_y_overfill_rot = rotate_points(
        turbine_x_overfill, turbine_y_overfill, center_x, center_y, theta
    )

    # Create a polygon using the properly ordered coordinates
    bounding_polygon = Path(np.column_stack((ordered_x, ordered_y)))

    # Combine the rotated turbine positions into an array for filtering
    turbine_positions = np.column_stack((turbine_x_overfill_rot, turbine_y_overfill_rot))

    # Check which turbines lie inside the polygon
    inside_polygon = bounding_polygon.contains_points(turbine_positions)

    # Filter turbines that are inside the polygon
    turbine_x_final_overfill = turbine_x_overfill_rot[inside_polygon]
    turbine_y_final_overfill = turbine_y_overfill_rot[inside_polygon]

    # Calculate the number of turbines
    num_turbines = len(turbine_x_final_overfill)

    # Rotate the coordinate system to align with the original grid before rotation
    theta_inv = -theta  # Inverse of rotation angle to align back to grid reference frame
    turbine_x_rot_back, turbine_y_rot_back = rotate_points(
        turbine_x_final_overfill, turbine_y_final_overfill, center_x, center_y, theta_inv
    )

    # Count unique x and y values in the rotated reference frame
    unique_x_rot = np.unique(np.round(turbine_x_rot_back / spacing_x))  # Grouping based on spacing
    unique_y_rot = np.unique(np.round(turbine_y_rot_back / spacing_y))  # Grouping based on spacing
    num_columns = len(unique_x_rot)
    num_rows = len(unique_y_rot)

    turbines_per_column = {}

    for x_value in unique_x_rot:
        indices_in_column = np.isclose(np.round(turbine_x_rot_back / spacing_x), x_value)
        turbines_in_column = np.sum(indices_in_column)
        turbines_per_column[x_value] = turbines_in_column


    # Cable Length Calculations

    # Length per column: (turbines_per_column - 1) * 5D
    column_cabling_lengths = {column: (count - 1) * 5 * D for column, count in turbines_per_column.items()}

    # Total length for columns
    total_column_cabling_length = sum(column_cabling_lengths.values())

    # Length to attach the rows: (num_rows - 1) * 7D
    row_cabling_length = (num_columns - 1) * 7 * D

    # Total cabling length
    total_cabling_length = total_column_cabling_length + row_cabling_length


    # Plot the updated layout with strict polygon filtering
    plt.figure(figsize=(10, 10))
    plt.plot(np.append(ordered_x, ordered_x[0]), np.append(ordered_y, ordered_y[0]), 'r-', label='Site Boundary')
    plt.scatter(turbine_x_final_overfill, turbine_y_final_overfill, c='b', label='Wind Turbine')
    plt.title(f'Wind Turbine Farm Layout (D={D:.0f}m, Facing {np.degrees(theta):.0f}°)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('Figures/SitePlotD100T210.png')  # Save the figure
    plt.show()

    return turbine_x_final_overfill, turbine_y_final_overfill, num_turbines, num_rows, num_columns, total_cabling_length, column_cabling_lengths, row_cabling_length


