import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import weibull_min
from windrose import WindroseAxes


# Load the Excel file
file_path = '../Data/Wind_Data.xlsx'
wind_data = pd.read_excel(file_path)

# Conversion factor from knots to meters per second
knots_to_mps = 0.514444
# Add a new column for wind speed in meters per second
wind_data['Wind - Mean Speed (mps)'] = wind_data['Wind - Mean Speed (knots)'] * knots_to_mps
# Replace 360° with 0° in the wind direction column
wind_data['Wind - Mean Direction'] = wind_data['Wind - Mean Direction'].replace(360, 0)


wind_cleaned = wind_data.dropna()
wind_speed = wind_cleaned['Wind - Mean Speed (mps)'].values
wind_direction = wind_cleaned['Wind - Mean Direction'].values

wind_grouped = wind_cleaned.groupby('Wind - Mean Direction').agg(
    avg_speed=('Wind - Mean Speed (mps)', 'mean'),
    frequency=('Wind - Mean Speed (mps)', 'size')  # Count occurrences for each direction
)

def weibull():
    # Include all wind speed data (including zeros)
    wind_speed_mps = wind_data['Wind - Mean Speed (mps)']

    # Remove non-finite values (NaN, Inf)
    wind_speed_mps = wind_speed_mps[np.isfinite(wind_speed_mps)]

    # Fit the Weibull distribution to the wind speed data, including zeros
    shape, loc, scale = weibull_min.fit(wind_speed_mps, floc=0)  # Fix location at 0

    # Generate points for the Weibull distribution
    x = np.linspace(0, max(wind_speed_mps), 100)
    weibull_pdf = weibull_min.pdf(x, shape, loc, scale)

    # Plot the histogram of the wind speed data
    plt.hist(wind_speed_mps, bins=30, density=True, alpha=0.6, color='g', label='Wind Speed Data')

    # Plot the fitted Weibull distribution
    plt.plot(x, weibull_pdf, 'r-', lw=2, label=f'Weibull fit\nShape: {shape:.2f}, Scale: {scale:.2f}')

    # Add labels and legend
    plt.title('Weibull Distribution of Wind Speed (Including Zeros)')
    plt.xlabel('Wind Speed (mps)')
    plt.ylabel('Density')
    plt.legend()
    print(x, loc)
    # Show plot
    plt.savefig('..\Figures/Weibull.png')  # Save the figure

    plt.show()
    return


def plot_wind_rose(wind_speed, wind_direction):
    ax = WindroseAxes.from_ax()
    # Use the `bar` method to plot wind direction vs wind speed
    ax.bar(wind_direction, wind_speed, normed=True, opening=0.8, edgecolor='white')
    ax.set_legend(title="Wind speed (m/s)")

    # Add labels and title
    plt.title('Wind Rose: Wind Speed and Direction Frequencies')
    plt.savefig('..\Figures/WindRose.png')  # Save the figure

    plt.show()

def double_wind_rose():
    directions = np.radians(wind_grouped.index)  # Convert degrees to radians for the plot
    avg_speeds = wind_grouped['avg_speed'].values
    frequencies = wind_grouped['frequency'].values

    # Create the first plot (Direction vs Average Speed)
    plt.figure(figsize=(10, 5))

    # Radial plot for average wind speed
    plt.subplot(1, 2, 1, polar=True)
    plt.plot(directions, avg_speeds, marker='o', linestyle='-', color='b')
    plt.fill(directions, avg_speeds, color='b', alpha=0.3)
    plt.title('Wind Direction vs. Average Speed')
    plt.gca().set_theta_zero_location('N')  # Set 0 degrees at the top (North)
    plt.gca().set_theta_direction(-1)  # Make the plot clockwise (North-East-South-West)
    plt.xlabel('Direction (°)')
    plt.ylabel('Avg Wind Speed (m/s)')

    # Create the second plot (Direction vs Frequency)
    plt.subplot(1, 2, 2, polar=True)
    plt.plot(directions, frequencies, marker='o', linestyle='-', color='g')
    plt.fill(directions, frequencies, color='g', alpha=0.3)
    plt.title('Wind Direction vs. Frequency')
    plt.gca().set_theta_zero_location('N')  # Set 0 degrees at the top (North)
    plt.gca().set_theta_direction(-1)  # Make the plot clockwise (North-East-South-West)
    plt.xlabel('Direction (°)')
    plt.ylabel('Frequency')

    # Display the plots
    plt.tight_layout()
    plt.savefig('..\Figures/Double_Wind_Rose.png')  # Save the figure
    plt.show()

# Call functions to plot the Weibull distribution and wind rose
weibull()
double_wind_rose()
plot_wind_rose(wind_speed, wind_direction)
