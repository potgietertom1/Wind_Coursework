from _0_AviemoreSite import *

from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
from py_wake.site import UniformWeibullSite
from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine

from Tools.financial_calculations import wind_farm_calculations


def evaluate_turbines(turbine_data, theta):
    best_LCOE = float('inf')
    best_turbine = None
    turbine_results = {}

    for turbine_name, data in turbine_data.items():
        print(f"Evaluating turbine: {turbine_name}")

        # Extract data for the turbine
        D = data['rotor_diameter_m']
        hub_height = data['hub_height_m']
        rated_power = data['rated_power_kW']
        ws_cutin = data.get('cut_in_wind_speed_ms', 1)
        ws_cutout = data.get('cut_out_wind_speed_ms')
        cost = data['cost_per_turbine']

        # Initialize turbine and wind data
        wt = GenericWindTurbine(
            name=turbine_name,
            diameter=D,
            hub_height=hub_height,
            power_norm=rated_power,
            ws_cutin=ws_cutin,
            ws_cutout=ws_cutout
        )
        weibull_direction_data = wind_initialisation(hub_height)
        site = UniformWeibullSite(
            p_wd=weibull_direction_data['Frequency'].tolist(),
            a=weibull_direction_data['Weibull_A'].tolist(),
            k=weibull_direction_data['Weibull_k'].tolist(),
            ti=0.1
        )
        windFarmModel = Bastankhah_PorteAgel_2014(site, wt, k=0.0324555)

        # Set up wind farm layout and calculate AEP
        wt_x, wt_y, n_turbines, n_rows, n_columns, total_cabling_length, column_cabling_lengths, row_cabling_length = create_wind_farm_grid(D, theta)
        sim_res = windFarmModel(wt_x, wt_y)
        aep = sim_res.aep().sum().item()
        aep_no_wake = sim_res.aep(with_wake_loss=False).sum().item()

        # Calculate wake loss and capacity factor
        loss_to_wake_effects = (aep_no_wake - aep) / aep * 100 if aep > 0 else 0
        capacity_factor = (aep / (rated_power / 1000000 * n_turbines * 8760)) * 100

        # Perform financial calculations
        financial_results = wind_farm_calculations(
            cost_per_turbine=cost,
            n_turbines=n_turbines,
            AEP=aep,
            CFD_strike_price=CFD_strike_price,
            discount_rate=discount_rate,
            project_lifetime=project_lifetime,
            inflation_rate=inflation_rate,
            depreciation_rate=depreciation_rate,
            cost_per_meter_of_cable=cable_cost,
            total_cabling_length=total_cabling_length
        )

        # Store results
        turbine_results[turbine_name] = {
            'Rotor Diameter (m)': D,
            'Hub Height (m)': hub_height,
            'Rated Power (kW)': rated_power,
            'Cost per Turbine (GBP)': cost,
            'Number of Turbines': n_turbines,
            'AEP (GWh)': aep,
            'AEP (GWh)(No Wake)': aep_no_wake,
            'Loss to Wake Effects (%)': loss_to_wake_effects,
            'Capacity Factor (%)': capacity_factor,
            **financial_results
        }

        # Update best turbine if current LCOE is lower
        lcoe = financial_results["LCOE With Depreciation (GBP/MWh)"]
        if lcoe < best_LCOE:
            best_LCOE = lcoe
            best_turbine = turbine_name

    # Convert results to a DataFrame and save to CSV
    results_df = pd.DataFrame.from_dict(turbine_results, orient='index')
    results_df.to_csv('Results/turbine_results.csv', index_label='Turbine Name')

    return turbine_results, best_turbine, best_LCOE


# _____ Initialise Parameters and Run Evaluation _____
theta = np.radians(210)
turbine_data = load_turbine_data("Data/Turbine_Data.xlsx")  # Assume turbine data loading function
turbine_results, best_turbine, best_LCOE = evaluate_turbines(turbine_data, theta)

# Print results
print("\nResults for all turbines:")
for turbine_name, results in turbine_results.items():
    print(f"Turbine: {turbine_name}, Results: {results}")

print(f"\nTurbine with lowest LCOE: {best_turbine} with LCOE: {best_LCOE:.2f} (GBP/MWh)")


