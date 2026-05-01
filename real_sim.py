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


class compressor2:
    # isothermic compressor: regime de stabilite
    # thermique du comression all excess heat is dumped outside the system
    def __init__(
        self,
        fluid,
        T_in,
        P_in,
        P_out,
        steps,
        simulation_limits={"T_min": 200, "T_max": 1000, "P_min": 1e5},
    ):
        self.fluid = fluid
        self.T_in = T_in
        self.P_in = P_in
        self.P_out = P_out
        self.steps = steps
        self.p_step = (self.P_out - self.P_in) / self.steps
        self.pressure_data = np.zeros(steps + 1)
        self.h_data = np.zeros(steps + 1)
        self.W_comp = np.zeros(steps + 1)
        self.entropy_data = np.zeros(steps + 1)
        self.rejectted_heat = np.zeros(steps + 1)
        self.gas_fraction = np.ones(steps + 1)
        self.pressure_data[0] = P_in
        self.h_data[0] = PropsSI("H", "T", T_in, "P", P_in, fluid)
        self.entropy_data[0] = PropsSI("S", "T", T_in, "P", P_in, fluid)

    def step(self, n):
        new_p = self.pressure_data[n - 1] + self.p_step
        new_H = PropsSI("H", "T", self.T_in, "P", new_p, self.fluid)
        new_s = PropsSI("S", "T", self.T_in, "P", new_p, self.fluid)
        rejectted_heat = -(new_s - self.entropy_data[n - 1]) * self.T_in
        W = new_H - self.h_data[n - 1] + rejectted_heat
        Q = PropsSI("Q", "T", self.T_in, "P", new_p, self.fluid)
        if Q < 0:
            Q = 1.0
        return new_p, new_H, new_s, W, rejectted_heat, Q

    def simulate(self):
        for n in range(1, self.steps + 1):
            new_p, new_H, new_s, W, rejectted_heat, Q = self.step(n)
            self.pressure_data[n] = new_p
            self.h_data[n] = new_H
            self.entropy_data[n] = new_s
            self.W_comp[n] = W
            self.rejectted_heat[n] = rejectted_heat
            self.gas_fraction[n] = Q
        return self.steps


class condensor:
    # the transformation is isothermal and with saturated fluid along the process
    def __init__(
        self,
        fluid,
        T_in,
        P_in,
        Q_in,
        steps,
        max_p_loss_per_step=0.1,
        simulation_limits={"T_min": 200, "T_max": 1000, "P_min": 1e5},
    ):
        self.fluid = fluid
        self.T_in = T_in
        self.P_in = P_in
        self.Q_in = Q_in
        self.steps = steps
        self.max_p_loss_per_step = max_p_loss_per_step
        self.Q_step = Q_in / steps
        self.pressure_loss = np.zeros(steps + 1)
        self.h_data = np.zeros(steps + 1)
        self.entropy_data = np.zeros(steps + 1)
        self.gas_fraction = np.zeros(steps + 1)
        self.density_data = np.zeros(steps + 1)
        self.viscosity_data = np.zeros(steps + 1)
        self.under_pressure = 0
        self.culmulative_pressure_loss = 0
        self.h_data[0] = PropsSI("H", "T", T_in, "P", P_in, fluid)
        self.entropy_data[0] = PropsSI("S", "T", T_in, "P", P_in, fluid)
        self.gas_fraction[0] = Q_in
        self.density_data[0] = PropsSI("D", "T", T_in, "P", P_in, fluid)
        self.viscosity_data[0] = PropsSI("V", "T", T_in, "P", P_in, fluid)
        self.simulation_limits = simulation_limits
        T_crit = PropsSI("TCRIT", fluid)
        self.rho_min = PropsSI(
            "D", "T", simulation_limits["T_max"], "P", simulation_limits["P_min"], fluid
        )
        self.rho_max = PropsSI(
            "D", "T", min(simulation_limits["T_min"], T_crit), "Q", 0, fluid
        )
        self.mu_min = PropsSI(
            "V", "T", simulation_limits["T_max"], "P", simulation_limits["P_min"], fluid
        )
        self.mu_max = PropsSI(
            "V", "T", min(simulation_limits["T_min"], T_crit), "Q", 0, fluid
        )

    def step(self, n):
        new_Q = self.gas_fraction[n - 1] - self.Q_step
        new_H = PropsSI("H", "T", self.T_in, "Q", new_Q, self.fluid)
        new_s = PropsSI("S", "T", self.T_in, "Q", new_Q, self.fluid)
        new_ro = PropsSI("D", "T", self.T_in, "Q", new_Q, self.fluid)
        new_mu = PropsSI("V", "T", self.T_in, "Q", new_Q, self.fluid)
        coeff = (new_mu - self.mu_min) / (self.mu_max - self.mu_min) + (
            self.rho_max - new_ro
        ) / (self.rho_max - self.rho_min)
        pressure_loss = self.max_p_loss_per_step * self.P_in * self.Q_step * coeff
        return new_H, new_s, new_Q, new_ro, new_mu, pressure_loss

    def simulate(self):
        p_cond = PropsSI("P", "T", self.T_in, "Q", 0, self.fluid)
        for n in range(1, self.steps + 1):
            new_H, new_s, new_Q, new_ro, new_mu, pressure_loss = self.step(n)
            self.h_data[n] = new_H
            self.entropy_data[n] = new_s
            self.gas_fraction[n] = new_Q
            self.density_data[n] = new_ro
            self.viscosity_data[n] = new_mu
            self.pressure_loss[n] = pressure_loss
            self.culmulative_pressure_loss += pressure_loss
            if self.P_in - self.culmulative_pressure_loss < p_cond:
                self.under_pressure += 1
        return self.under_pressure, self.culmulative_pressure_loss


class expander:
    # isenthalpic expander
    def __init__(self, fluid, H_in, P_in, P_out):
        self.fluid = fluid
        self.H_in = H_in
        self.P_in = P_in
        self.P_out = P_out
        T_out = PropsSI("T", "H", H_in, "P", P_out, fluid)
        Q_out = PropsSI("Q", "H", H_in, "P", P_out, fluid)


class evaporator1:
    def __init__(self, fluid, Q_in, T_in, steps):
        self.fluid = fluid
        self.Q_in = Q_in
        self.T_in = T_in
        self.steps = steps
        self.q_step = (1 - Q_in) / steps
        self.h_data = np.zeros(steps + 1)
        self.entropy_data = np.zeros(steps + 1)
        self.heat_sucked_data = np.zeros(steps + 1)
        self.gas_fraction = np.zeros(steps + 1)
        self.h_data[0] = PropsSI("H", "T", T_in, "Q", Q_in, fluid)
        self.entropy_data[0] = PropsSI("S", "T", T_in, "Q", Q_in, fluid)
        self.gas_fraction[0] = Q_in

    def step(self, n):
        new_Q = self.gas_fraction[n - 1] + self.q_step
        new_H = PropsSI("H", "T", self.T_in, "Q", new_Q, self.fluid)
        new_s = PropsSI("S", "T", self.T_in, "Q", new_Q, self.fluid)
        heat = new_H - self.h_data[n - 1]
        new_s = PropsSI("S", "T", self.T_in, "Q", new_Q, self.fluid)
        return new_H, new_s, new_Q, heat, new_s

    def simulate(self):
        for n in range(1, self.steps + 1):
            new_H, new_s, new_Q, heat, new_s = self.step(n)
            self.h_data[n] = new_H
            self.entropy_data[n] = new_s
            self.gas_fraction[n] = new_Q
            self.heat_sucked_data[n] = heat
            self.entropy_data[n] = new_s
        return self.steps


class evaporator2:
    # we are still sucking heat to heat our gaz
    def __init__(self, fluid, T_in, T_out, P_in, steps):
        self.fluid = fluid
        self.T_in = T_in
        self.T_out = T_out
        self.P_in = P_in
        self.steps = steps
        self.T_step = (T_out - T_in) / steps
        self.temperature_data = np.zeros(steps + 1)
        self.h_data = np.zeros(steps + 1)
        self.entropy_data = np.zeros(steps + 1)
        self.heat_sucked_data = np.zeros(steps + 1)
        self.temperature_data[0] = T_in
        self.h_data[0] = PropsSI("H", "T", T_in, "P", P_in, fluid)
        self.entropy_data[0] = PropsSI("S", "T", T_in, "P", P_in, fluid)

    def step(self, n):
        new_T = self.temperature_data[n - 1] + self.T_step
        new_H = PropsSI("H", "T", new_T, "P", self.P_in, self.fluid)
        new_s = PropsSI("S", "T", new_T, "P", self.P_in, self.fluid)
        heat = new_H - self.h_data[n - 1]
        return new_T, new_H, new_s, heat

    def simulate(self):
        for n in range(1, self.steps + 1):
            new_T, new_H, new_s, heat = self.step(n)
            self.temperature_data[n] = new_T
            self.h_data[n] = new_H
            self.entropy_data[n] = new_s
            self.heat_sucked_data[n] = heat
        return np.sum(self.heat_sucked_data)
