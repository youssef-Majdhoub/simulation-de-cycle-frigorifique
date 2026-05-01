from CoolProp.CoolProp import PropsSI
import pandas as pd
import numpy as np


class compressor1:
    # isentropic compressor
    def __init__(
        self,
        fluid,
        T_in,
        P_in,
        P_max,
        T_out,
        steps,
        simulation_limits={"T_min": 200, "T_max": 1000, "P_min": 1e5},
        max_penality_leakage=0.25,
        max_penality_friction=0.25,
    ):
        self.fluid = fluid
        self.T_in = T_in
        self.P_in = P_in
        self.P_max = P_max
        self.T_out = T_out
        self.steps = steps
        self.p_step = (P_max - P_in) / steps
        self.pressure_data = np.zeros(steps + 1)
        self.temperature_data = np.zeros(steps + 1)
        self.h_data = np.zeros(steps + 1)
        self.density_data = np.zeros(steps + 1)
        self.viscosity_data = np.zeros(steps + 1)
        self.W_comp = np.zeros(steps + 1)
        self.entropy_data = np.zeros(steps + 1)
        self.pressure_data[0] = P_in
        self.temperature_data[0] = T_in
        self.h_data[0] = PropsSI("H", "T", T_in, "P", P_in, fluid)
        self.entropy_data[0] = PropsSI("S", "T", T_in, "P", P_in, fluid)
        self.density_data[0] = PropsSI("D", "T", T_in, "P", P_in, fluid)
        self.viscosity_data[0] = PropsSI("V", "T", T_in, "P", P_in, fluid)
        self.max_penality_leakage = max_penality_leakage
        self.max_penality_friction = max_penality_friction
        self.simulation_limits = simulation_limits
        T_crit = PropsSI("TCRIT", fluid)
        self.rho_min = PropsSI(
            "D", "T", simulation_limits["T_min"], "P", simulation_limits["P_min"], fluid
        )
        self.rho_max = PropsSI(
            "D", "T", min(simulation_limits["T_max"], T_crit), "Q", 1, fluid
        )
        self.mu_min = PropsSI(
            "V", "T", simulation_limits["T_min"], "P", simulation_limits["P_min"], fluid
        )
        self.mu_max = PropsSI(
            "V", "T", min(simulation_limits["T_max"], T_crit), "Q", 1, fluid
        )

    def step(self, n):
        new_p = self.pressure_data[n - 1] + self.p_step
        new_S = PropsSI("S", "P", new_p, "T", self.temperature_data[n - 1], self.fluid)
        new_H = PropsSI("H", "P", new_p, "T", self.temperature_data[n - 1], self.fluid)
        new_ro = PropsSI("D", "P", new_p, "T", self.temperature_data[n - 1], self.fluid)
        new_mu = PropsSI("V", "P", new_p, "T", self.temperature_data[n - 1], self.fluid)
        leakage_penality = (
            self.max_penality_leakage
            * (self.rho_max - new_ro)
            / (self.rho_max - self.rho_min)
        )
        friction_penality = (
            self.max_penality_friction
            * (new_mu - self.mu_min)
            / (self.mu_max - self.mu_min)
        )
        coeff = 1 - leakage_penality - friction_penality
        W = (new_H - self.h_data[n - 1]) / coeff
        new_H = self.h_data[n - 1] + W
        new_T = PropsSI(
            "T", "P", new_p, "H", new_H, self.fluid
        )  # Update temperature based on new enthalpy
        new_S = PropsSI(
            "S", "P", new_p, "T", new_T, self.fluid
        )  # Update entropy based on new temperature
        return new_p, new_T, new_S, new_ro, new_mu, new_H, W

    def simulate(self):
        for n in range(1, self.steps + 1):
            new_p, new_T, new_S, new_ro, new_mu, new_H, W = self.step(n)
            if new_T >= self.T_out or new_p >= self.P_max:
                print(f"Simulation stopped at step {n} due to limits.")
                self.real_steps = n
                self.pressure_data[n] = new_p
                self.temperature_data[n] = new_T
                self.h_data[n] = new_H
                self.density_data[n] = new_ro
                self.viscosity_data[n] = new_mu
                self.entropy_data[n] = new_S
                self.W_comp[n] = W
                self.pressure_data = self.pressure_data[: n + 1]
                self.temperature_data = self.temperature_data[: n + 1]
                self.h_data = self.h_data[: n + 1]
                self.density_data = self.density_data[: n + 1]
                self.viscosity_data = self.viscosity_data[: n + 1]
                self.entropy_data = self.entropy_data[: n + 1]
                self.W_comp = self.W_comp[: n + 1]
                break
            else:
                self.pressure_data[n] = new_p
                self.temperature_data[n] = new_T
                self.h_data[n] = new_H
                self.density_data[n] = new_ro
                self.viscosity_data[n] = new_mu
                self.entropy_data[n] = new_S
                self.W_comp[n] = W
        return self.real_steps
