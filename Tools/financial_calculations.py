import numpy_financial as npf

def wind_farm_calculations(cost_per_turbine, n_turbines, AEP, CFD_strike_price, discount_rate, project_lifetime,
                           inflation_rate, depreciation_rate, cost_per_meter_of_cable, total_cabling_length):
    """
    Perform financial calculations for a wind farm project.

    Parameters:
        cost_per_turbine (float): Cost of each turbine (GBP).
        n_turbines (int): Number of turbines.
        AEP (float): Annual energy production (MWh).
        CFD_strike_price (float): Contract for difference strike price (GBP/MWh).
        discount_rate (float): Discount rate for NPV calculations.
        project_lifetime (int): Lifetime of the wind farm (years).
        inflation_rate (float): Annual inflation rate.
        depreciation_rate (float): Annual depreciation rate for energy production.
        cost_per_meter_of_cable (float): Cost of cable per meter (GBP).
        total_cabling_length (float): Total length of cables required (meters).

    Returns:
        dict: Contains results like capital costs, LCOE, NPV, IRR, payback period, etc.
    """
    # Capital cost of the wind farm
    capital_cost = cost_per_turbine * n_turbines  # Total turbine capital cost

    # Cable cost (added here)
    cable_cost = cost_per_meter_of_cable * total_cabling_length

    # Breakdown of capital costs (with cable cost added)
    support_structure_cost = 0.553 * capital_cost
    BOS_capex = 0.278 * capital_cost
    soft_cost = 0.151 * capital_cost

    # Total CAPEX, including cable cost
    total_capital_cost = capital_cost/0.553 + cable_cost

    OM = 0.26/0.74*total_capital_cost

    # Operations & Maintenance Costs (2% of total capital cost annually)
    annual_OM_costs = 0.02 * total_capital_cost

    # Without inflation
    OM_costs_without_inflation = [annual_OM_costs for _ in range(project_lifetime)]

    # With inflation
    OM_costs_with_inflation = [annual_OM_costs * ((1 + inflation_rate) ** (year - 1)) for year in
                               range(1, project_lifetime + 1)]

    # Decommissioning Cost (15% of total capital cost at end of life)
    decommissioning_cost = 0.15 * total_capital_cost

    # Total energy output (without and with depreciation)
    total_energy_output_without_depreciation = AEP * project_lifetime * 1000  # in MWh
    total_energy_output_with_depreciation = sum(
        AEP * (1 - depreciation_rate) ** (year - 1) * 1000 for year in range(1, project_lifetime + 1))

    # Total revenue per year
    annual_revenue = CFD_strike_price * AEP * 1000  # GBP/year

    # Cash flows (without and with inflation)
    cash_flows_without_inflation = [-total_capital_cost] + [(annual_revenue - om_cost) for om_cost in
                                                            OM_costs_without_inflation]
    cash_flows_with_inflation = [-total_capital_cost] + [(annual_revenue - om_cost) for om_cost in
                                                         OM_costs_with_inflation]

    # Subtract decommissioning cost in the final year
    cash_flows_without_inflation[-1] -= decommissioning_cost
    cash_flows_with_inflation[-1] -= decommissioning_cost

    # NPV calculation
    npv_without_inflation = sum(
        cf / ((1 + discount_rate) ** year) for year, cf in enumerate(cash_flows_without_inflation))
    npv_with_inflation = sum(cf / ((1 + discount_rate) ** year) for year, cf in enumerate(cash_flows_with_inflation))

    # IRR calculation
    irr_without_inflation = npf.irr(cash_flows_without_inflation)
    irr_with_inflation = npf.irr(cash_flows_with_inflation)

    # Payback period calculation
    cumulative_cash_flow = 0
    payback_period_without_inflation = None
    for year, cash_flow in enumerate(cash_flows_without_inflation[1:], start=1):
        cumulative_cash_flow += cash_flow
        if cumulative_cash_flow >= total_capital_cost:
            payback_period_without_inflation = year
            break

    cumulative_cash_flow = 0
    payback_period_with_inflation = None
    for year, cash_flow in enumerate(cash_flows_with_inflation[1:], start=1):
        cumulative_cash_flow += cash_flow
        if cumulative_cash_flow >= total_capital_cost:
            payback_period_with_inflation = year
            break

    # LCOE calculation (Levelized Cost of Energy)
    total_lifetime_costs_without_inflation = total_capital_cost + sum(OM_costs_without_inflation) + decommissioning_cost
    total_lifetime_costs_with_inflation = total_capital_cost + sum(OM_costs_with_inflation) + decommissioning_cost

    lcoe_without_depreciation = total_lifetime_costs_without_inflation / total_energy_output_without_depreciation  # GBP/MWh
    lcoe_with_depreciation = total_lifetime_costs_with_inflation / total_energy_output_with_depreciation  # GBP/MWh

    # Results dictionary
    results = {
        "Total Capital Cost (GBP)": total_capital_cost,
        "Cable Cost (GBP)": cable_cost,
        "LCOE Without Depreciation (GBP/MWh)": lcoe_without_depreciation,
        "LCOE With Depreciation (GBP/MWh)": lcoe_with_depreciation,
        "Payback Period Without Inflation (years)": payback_period_without_inflation,
        "Payback Period With Inflation (years)": payback_period_with_inflation,
        "NPV Without Inflation (GBP)": npv_without_inflation,
        "NPV With Inflation (GBP)": npv_with_inflation,
        "IRR Without Inflation (%)": irr_without_inflation * 100,
        "IRR With Inflation (%)": irr_with_inflation * 100
    }

    return results
