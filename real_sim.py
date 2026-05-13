import os

from CoolProp.CoolProp import PropsSI, PhaseSI
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib as pl


class compressor1:
    # corrected isentropic compressor
    def __init__(
        self,
        fluid,
        T_in,
        P_in,
        P_max,
        T_out,
        steps,
        simulation_limits={"T_min": 200, "T_max": 1000, "P_min": 1e5},
        max_penality_leakage=0.1,
        max_penality_friction=0.1,
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
        new_H = PropsSI("H", "P", new_p, "S", self.entropy_data[n - 1], self.fluid)
        Q = PropsSI("Q", "P", new_p, "S", self.entropy_data[n - 1], self.fluid)
        if Q != 1 and Q != 0:
            new_ro = 1 / (Q * PropsSI("D", "P", new_p, "Q", 1, self.fluid)) + 1 / (
                (1 - Q) * PropsSI("D", "P", new_p, "Q", 0, self.fluid)
            )
            new_ro = 1 / new_ro
            new_mu = Q * PropsSI("V", "P", new_p, "Q", 1, self.fluid) + (
                1 - Q
            ) * PropsSI("V", "P", new_p, "Q", 0, self.fluid)
        else:
            new_ro = PropsSI("D", "P", new_p, "S", self.entropy_data[n - 1], self.fluid)
            new_mu = PropsSI("V", "P", new_p, "S", self.entropy_data[n - 1], self.fluid)
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
        if W < 0:
            print(f"Negative work at step {n}, this is a bug.")
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
                self.real_steps = n
                break
            else:
                self.pressure_data[n] = new_p
                self.temperature_data[n] = new_T
                self.h_data[n] = new_H
                self.density_data[n] = new_ro
                self.viscosity_data[n] = new_mu
                self.entropy_data[n] = new_S
                self.W_comp[n] = W
                self.real_steps = n
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
        self.pressure_data[0] = P_in
        self.h_data[0] = PropsSI("H", "T", T_in, "P", P_in, fluid)
        self.entropy_data[0] = PropsSI("S", "T", T_in, "P", P_in, fluid)

    def step(self, n):
        new_p = self.pressure_data[n - 1] + self.p_step
        new_H = PropsSI("H", "T", self.T_in, "P", new_p, self.fluid)
        new_s = PropsSI("S", "T", self.T_in, "P", new_p, self.fluid)
        rejectted_heat = -(new_s - self.entropy_data[n - 1]) * self.T_in
        W = new_H - self.h_data[n - 1] + rejectted_heat
        return new_p, new_H, new_s, W, rejectted_heat

    def simulate(self):
        for n in range(1, self.steps + 1):
            new_p, new_H, new_s, W, rejectted_heat = self.step(n)
            self.pressure_data[n] = new_p
            self.h_data[n] = new_H
            self.entropy_data[n] = new_s
            self.W_comp[n] = W
            self.rejectted_heat[n] = rejectted_heat
        return self.steps


class condensor:
    # the transformation is isothermal and with saturated fluid along the process
    def __init__(
        self,
        fluid,
        T_in,
        P_in,
        last_w,
        delta_p,
        steps,
        max_p_loss_per_step=0.1,
        simulation_limits={"T_min": 200, "T_max": 1000, "P_min": 1e5},
    ):
        self.fluid = fluid
        self.T_in = T_in
        self.P_in = P_in
        self.Q_in = 1
        self.last_w = last_w
        self.delta_p = delta_p
        self.steps = steps
        self.max_p_loss_per_step = max_p_loss_per_step
        self.Q_step = self.Q_in / steps
        self.pressure_loss = np.zeros(steps + 1)
        self.h_data = np.zeros(steps + 1)
        self.entropy_data = np.zeros(steps + 1)
        self.gas_fraction = np.ones(steps + 1)
        self.w_comp = np.zeros(steps + 1)
        self.density_data = np.zeros(steps + 1)
        self.viscosity_data = np.zeros(steps + 1)
        self.culmulative_pressure_loss = 0
        self.h_data[0] = PropsSI("H", "T", T_in, "Q", self.Q_in, fluid)
        self.entropy_data[0] = PropsSI("S", "T", T_in, "Q", self.Q_in, fluid)
        self.gas_fraction[0] = 1
        self.density_data[0] = PropsSI("D", "T", T_in, "Q", self.Q_in, fluid)
        self.viscosity_data[0] = PropsSI("V", "T", T_in, "Q", self.Q_in, fluid)
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
        if new_Q < 0:
            new_Q = 0
        new_H = PropsSI("H", "T", self.T_in, "Q", new_Q, self.fluid)
        new_s = PropsSI("S", "T", self.T_in, "Q", new_Q, self.fluid)
        if new_Q != 1 and new_Q != 0:
            new_ro = 1 / (
                new_Q * PropsSI("D", "T", self.T_in, "Q", 1, self.fluid)
            ) + 1 / ((1 - new_Q) * PropsSI("D", "T", self.T_in, "Q", 0, self.fluid))
            new_ro = 1 / new_ro
            new_mu = new_Q * PropsSI("V", "T", self.T_in, "Q", 1, self.fluid) + (
                1 - new_Q
            ) * PropsSI("V", "T", self.T_in, "Q", 0, self.fluid)
        else:
            new_ro = PropsSI("D", "T", self.T_in, "Q", new_Q, self.fluid)
            new_mu = PropsSI("V", "T", self.T_in, "Q", new_Q, self.fluid)
        coeff = (new_mu - self.mu_min) / (self.mu_max - self.mu_min) + (
            self.rho_max - new_ro
        ) / (self.rho_max - self.rho_min)
        pressure_loss = self.max_p_loss_per_step * self.P_in * self.Q_step * coeff
        return new_H, new_s, new_Q, new_ro, new_mu, pressure_loss

    def simulate(self):
        for n in range(1, self.steps + 1):
            new_H, new_s, new_Q, new_ro, new_mu, pressure_loss = self.step(n)
            self.h_data[n] = new_H
            self.entropy_data[n] = new_s
            self.gas_fraction[n] = new_Q
            self.density_data[n] = new_ro
            self.viscosity_data[n] = new_mu
            self.pressure_loss[n] = pressure_loss
            self.culmulative_pressure_loss += pressure_loss
            self.w_comp[n] = self.last_w * pressure_loss / self.delta_p
            if self.gas_fraction[n] == 0:
                print(f"Condensation completed at step {n}.")
                self.pressure_loss = self.pressure_loss[: n + 1]
                self.h_data = self.h_data[: n + 1]
                self.entropy_data = self.entropy_data[: n + 1]
                self.gas_fraction = self.gas_fraction[: n + 1]
                self.density_data = self.density_data[: n + 1]
                self.viscosity_data = self.viscosity_data[: n + 1]
                self.w_comp = self.w_comp[: n + 1]
                self.real_steps = n
                return (
                    np.sum(self.w_comp),
                    self.culmulative_pressure_loss,
                    self.real_steps,
                )
        return np.sum(self.w_comp), self.culmulative_pressure_loss, self.steps


class expander:
    # isenthalpic expander
    def __init__(self, fluid, H_in, P_in, P_out):
        self.fluid = fluid
        self.H_in = H_in
        self.P_in = P_in
        self.P_out = P_out
        self.T_out = PropsSI("T", "H", H_in, "P", P_out, fluid)
        stat = PhaseSI("H", H_in, "P", P_out, fluid)
        print(f"Expander output phase: {stat}")
        if stat == "twophase":
            self.Q_out = PropsSI("Q", "H", H_in, "P", P_out, fluid)
        elif stat == "liquid":
            self.Q_out = 0
        elif stat == "gas":
            self.Q_out = 1
        self.S_out = PropsSI("S", "H", H_in, "P", P_out, fluid)


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
        if new_Q >= 1.0:
            new_Q = 1.0
        new_H = PropsSI("H", "T", self.T_in, "Q", new_Q, self.fluid)
        new_s = PropsSI("S", "T", self.T_in, "Q", new_Q, self.fluid)
        heat = new_H - self.h_data[n - 1]
        return new_H, new_s, new_Q, heat

    def simulate(self):
        for n in range(1, self.steps + 1):
            new_H, new_s, new_Q, heat = self.step(n)
            self.h_data[n] = new_H
            self.entropy_data[n] = new_s
            self.gas_fraction[n] = new_Q
            self.heat_sucked_data[n] = heat
            if self.gas_fraction[n] == 1.0:
                self.real_steps = n
                self.h_data = self.h_data[: n + 1]
                self.entropy_data = self.entropy_data[: n + 1]
                self.gas_fraction = self.gas_fraction[: n + 1]
                self.heat_sucked_data = self.heat_sucked_data[: n + 1]
                return self.real_steps
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
        self.h_data[0] = PropsSI("H", "T", T_in, "Q", 1, fluid)
        self.entropy_data[0] = PropsSI("S", "T", T_in, "Q", 1, fluid)

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


# starting cycle simulation
# --- GLOBAL CYCLE TRACKING BY COMPONENT INDEX ---
# Index map:
# 0: Compressor 1 (Low stage)
# 1: Compressor 2 (Isothermal/High stage)
# 2: Condenser
# 3: Expander (Valve)
# 4: Evaporator 1 (Latent)
# 5: Evaporator 2 (Sensible)
class cycle:
    # all are evaluated once the COP is calculated
    def __init__(
        self,
        fluid,
        T_hot,
        T_cold,
        delta_T,
        P_in,
        com1_steps,
        com2_steps,
        cond_steps,
        evap1_steps,
        evap2_steps,
    ):
        self.fluid = fluid
        self.T_hot = T_hot
        self.T_cold = T_cold
        self.delta_T = delta_T  # the over heating/obercoolong delta
        self.P_in = P_in  # at compressor1 inlet
        self.com1_steps = com1_steps
        self.com2_steps = com2_steps
        self.cond_steps = cond_steps
        self.evap1_steps = evap1_steps
        self.evap2_steps = evap2_steps
        # every thing should be tracked
        self.pressure_data = []
        self.temperature_data = []
        self.process_data = []  # records which process is done each index
        self.entopy_data = []
        self.h_data = []
        self.W_comp = []
        self.heat_sucked_data = []

    def simulate(self):
        # compression1
        T1_out = self.T_hot + self.delta_T
        p1_max = PropsSI("P", "T", T1_out, "Q", 1.0, self.fluid)
        comp1 = compressor1(
            self.fluid, self.T_cold, self.P_in, p1_max, T1_out, self.com1_steps
        )
        real_steps = comp1.simulate()
        next = 0
        if real_steps < self.com1_steps:
            print(
                "Compressor 1 did not reach the desired output pressure,digging into possible reasons."
            )
            if comp1.temperature_data[-1] >= T1_out:
                print(
                    """The compressor reached the desired output temperature but not the pressure, 
                    this is the start of compression2"""
                )
                next = 1
            else:
                print(
                    """The compressor did not reach the desired output temperature, there is a bug"""
                )
                return
        else:
            print(
                "Compressor 1 reached the desired output pressure checking for final temperature"
            )
            if comp1.temperature_data[-1] >= T1_out:
                print("""The compressor reached the desired output temperature,
                    this is the start of condensation""")
                next = 2
            else:
                print("""The compressor did not reach the desired output temperature,
                    starting cold condensation""")
                next = 2
        self.pressure_data.extend(comp1.pressure_data)
        self.temperature_data.extend(comp1.temperature_data)
        self.h_data.extend(comp1.h_data)
        self.entopy_data.extend(comp1.entropy_data)
        self.process_data.extend([0] * (real_steps + 1))
        self.W_comp.extend(comp1.W_comp)
        self.heat_sucked_data.extend([0] * (real_steps + 1))
        if next == 1:
            # start compression2
            # collecting intial conditions for compression2
            T2_in = comp1.temperature_data[-1]
            P2_in = comp1.pressure_data[-1]
            comp2 = compressor2(
                self.fluid, T2_in, P2_in, p1_max, self.com2_steps
            )  # isothermal compression
            comp2.simulate()
            self.pressure_data.extend(comp2.pressure_data)
            self.temperature_data.extend([T2_in] * (comp2.steps + 1))
            self.h_data.extend(comp2.h_data)
            self.entopy_data.extend(comp2.entropy_data)
            self.process_data.extend([1] * (comp2.steps + 1))
            self.W_comp.extend(comp2.W_comp)
            self.heat_sucked_data.extend([0] * (comp2.steps + 1))
        # start condensation
        # collecting intial conditions for condensation
        T_cond_in = self.temperature_data[-1]
        P_cond_in = self.pressure_data[-1]
        delta_p = self.pressure_data[-1] - self.pressure_data[-2]
        w_last = comp2.W_comp[-1] if next == 1 else comp1.W_comp[-1]

        cond = condensor(
            self.fluid, T_cond_in, P_cond_in, w_last, delta_p, self.cond_steps
        )  # isothermal condensation
        cond.simulate()
        self.pressure_data.extend([P_cond_in] * (cond.steps + 1))
        self.temperature_data.extend([T_cond_in] * (cond.steps + 1))
        self.h_data.extend(cond.h_data)
        self.entopy_data.extend(cond.entropy_data)
        self.W_comp.extend(cond.w_comp)
        self.process_data.extend([2] * (cond.steps + 1))
        self.heat_sucked_data.extend([0] * (cond.steps + 1))
        # start expansion
        # collecting intial conditions for expansion
        H_exp_in = self.h_data[-1]
        P_exp_in = self.pressure_data[-1]
        P_exp_out = PropsSI("P", "T", self.T_cold - self.delta_T, "Q", 0, self.fluid)
        expan = expander(self.fluid, H_exp_in, P_exp_in, P_exp_out)
        self.pressure_data.append(P_exp_out)
        self.temperature_data.append(expan.T_out)
        self.h_data.append(H_exp_in)
        self.entopy_data.append(expan.S_out)
        self.process_data.append(3)
        self.W_comp.append(0)
        self.heat_sucked_data.append(0)
        # start evaporation1
        # collecting intial conditions for evaporation1
        T_evap1_in = self.temperature_data[-1]
        Q_in = expan.Q_out
        evap1 = evaporator1(self.fluid, Q_in, T_evap1_in, self.evap1_steps)
        steps = evap1.simulate()
        self.pressure_data.extend([P_exp_out] * (steps + 1))
        self.temperature_data.extend([T_evap1_in] * (steps + 1))
        self.h_data.extend(evap1.h_data)
        self.entopy_data.extend(evap1.entropy_data)
        self.process_data.extend([4] * (steps + 1))
        self.W_comp.extend([0] * (steps + 1))
        self.heat_sucked_data.extend(evap1.heat_sucked_data)
        # start evaporation2
        # collecting intial conditions for evaporation2
        T_evap2_in = self.temperature_data[-1]
        P_evap2_in = self.pressure_data[-1]
        T_evap2_out = self.T_cold
        evap2 = evaporator2(
            self.fluid, T_evap2_in, T_evap2_out, P_evap2_in, self.evap2_steps
        )
        evap2.simulate()
        self.pressure_data.extend([P_evap2_in] * (evap2.steps + 1))
        self.temperature_data.extend(evap2.temperature_data)
        self.h_data.extend(evap2.h_data)
        self.entopy_data.extend(evap2.entropy_data)
        self.process_data.extend([5] * (evap2.steps + 1))
        self.W_comp.extend([0] * (evap2.steps + 1))
        self.heat_sucked_data.extend(evap2.heat_sucked_data)
        self.COP = np.sum(self.heat_sucked_data) / np.sum(self.W_comp)
        return self.COP

    def save_data(self, org=""):
        # creating an ID for the cycle based on it's key parameters
        ID = f"{self.fluid}_T_hot{int(self.T_hot)}_T_cold{int(self.T_cold)}_delta_T{int(self.delta_T)}"
        ID += f"_P_in{int(self.P_in)}_com1_steps{self.com1_steps}_com2_steps{self.com2_steps}"
        ID += f"_cond_steps{self.cond_steps}_evap1_steps{self.evap1_steps}_evap2_steps{self.evap2_steps}"
        path = os.path.join(org, f"cycle_simulation_data_{ID}.xlsx")
        df = pd.DataFrame(
            {
                "Pressure": self.pressure_data,
                "Temperature": self.temperature_data,
                "Enthalpy": self.h_data,
                "Entropy": self.entopy_data,
                "Process": self.process_data,
                "Work": self.W_comp,
                "Heat_Sucked": self.heat_sucked_data,
            }
        )
        df.to_excel(path, index=False)


class Analysis:
    def __init__(self, fluid, T_hot, T_cold, P_in, steps=-1):
        self.fluid = fluid
        self.T_hot = T_hot
        self.T_cold = T_cold
        self.P_in = P_in
        self.steps = steps
        # delta_T range depends on the fridge ability to transfare heat effeciently
        self.delta_T = np.arange(1, 20, 1)
        if self.steps == -1:
            self.steps = int(
                self.convergence_analysis()[0]
            )  # get the number of steps needed for convergence

        self.cycle_sim = cycle(
            self.fluid,
            self.T_hot,
            self.T_cold,
            self.delta_T[0],
            self.P_in,
            self.steps,
            self.steps,
            self.steps,
            self.steps,
            self.steps,
        )
        self.cycle_sim.simulate()

    def convergence_analysis(self, T_hot=[], T_cold=[]):
        com1_steps = np.arange(10, 1000, 10)
        com2_steps = np.arange(10, 1000, 10)
        cond_steps = np.arange(10, 1000, 10)
        evap1_steps = np.arange(10, 1000, 10)
        evap2_steps = np.arange(10, 1000, 10)
        if len(T_hot) == 0:
            T_hot = np.array([self.T_hot + 5 * i for i in range(5)])
        if len(T_cold) == 0:
            T_cold = np.array([self.T_cold - 5 * i for i in range(5)])
        COP = np.zeros(com1_steps.size)
        n_conve = np.zeros(len(T_hot))
        for j in range(len(T_hot)):
            point = 0
            for i in range(com1_steps.size):
                cycle_sim = cycle(
                    self.fluid,
                    T_hot[j],
                    T_cold[j],
                    self.delta_T[0],
                    self.P_in,
                    com1_steps[i],
                    com2_steps[i],
                    cond_steps[i],
                    evap1_steps[i],
                    evap2_steps[i],
                )
                COP[i] = cycle_sim.simulate()
                print(
                    f"error at step {com1_steps[i]}: {abs(COP[i] - COP[i - 1]) if i > 0 else 'N/A'}"
                )
                if i > 0 and abs(COP[i] - COP[i - 1]) < 0.01 and point == 0:
                    print(
                        f"Convergence reached at step {com1_steps[i]} with COP {COP[i]}"
                    )
                    point = 1
                    COP[i:] = COP[
                        i
                    ]  # fill the rest of the array with the converged value
                    n_conve[j] = com1_steps[i]
                    break
        plt.plot(T_hot - T_cold, n_conve, "o-")
        plt.xlabel("Temperature difference (K)")
        plt.ylabel("Number of steps")
        plt.title("Convergence analysis of the cycle simulation")
        plt.show()
        return n_conve

    def analyze_pressure(self, pressure_array=[], process_array=[]):
        if len(pressure_array) == 0:
            pressure_array = self.cycle_sim.pressure_data
        if len(process_array) == 0:
            process_array = self.cycle_sim.process_data
        plt.plot(process_array, pressure_array, "o-")
        plt.xlabel("Process index")
        plt.ylabel("Pressure (Pa)")
        plt.title("Pressure evolution through the cycle")
        plt.show()

    def analyze_temperature(self, temperature_array=[], process_array=[]):
        if len(temperature_array) == 0:
            temperature_array = self.cycle_sim.temperature_data
        if len(process_array) == 0:
            process_array = self.cycle_sim.process_data
        plt.plot(process_array, temperature_array, "o-")
        plt.xlabel("Process index")
        plt.ylabel("Temperature (K)")
        plt.title("Temperature evolution through the cycle")
        plt.show()

    def analyze_entropy(self, entropy_array=[], process_array=[]):
        if len(entropy_array) == 0:
            entropy_array = self.cycle_sim.entopy_data
        if len(process_array) == 0:
            process_array = self.cycle_sim.process_data
        plt.plot(process_array, entropy_array, "o-")
        plt.xlabel("Process index")
        plt.ylabel("Entropy (J/kg.K)")
        plt.title("Entropy evolution through the cycle")
        plt.show()

    def analyze_work(self, work_array=[], process_array=[]):
        if len(work_array) == 0:
            work_array = self.cycle_sim.W_comp
        if len(process_array) == 0:
            process_array = self.cycle_sim.process_data
        plt.plot(process_array, work_array, "o-")
        plt.xlabel("Process index")
        plt.ylabel("Work (J/kg)")
        plt.title("Work evolution through the cycle")
        plt.show()

    def analyze_heat_sucked(self, heat_array=[], process_array=[]):
        if len(heat_array) == 0:
            heat_array = self.cycle_sim.heat_sucked_data
        if len(process_array) == 0:
            process_array = self.cycle_sim.process_data
        plt.plot(process_array, heat_array, "o-")
        plt.xlabel("Process index")
        plt.ylabel("Heat Sucked (J/kg)")
        plt.title("Heat Sucked evolution through the cycle")
        plt.show()


# convergence analysis
# 1-Tcold=cte
"""T_cold = np.array([268.15 for _ in range(26)])
T_hot = np.array([290.15 + i for i in range(26)])
analysis = Analysis("R134a", T_hot[0], T_cold[0], 1e5)
analysis.convergence_analysis(T_hot, T_cold)"""
# 2-Thot=cte
"""T_cold = np.array([260.15 + i for i in range(26)])
T_hot = np.array([308.15 for _ in range(26)])
analysis = Analysis("R134a", T_hot[0], T_cold[0], 1e5)
analysis.convergence_analysis(T_hot, T_cold)"""
# 3-differente fluids
"""fluids = ["R134a", "R12", "R22", "R717"]
T_cold = 268.15
T_hot = 308.15
for fluid in fluids:
    print(f"Analyzing cycle with fluid {fluid}")
    analysis = Analysis(fluid, T_hot, T_cold, 1e5)"""
# starting Thot=308.15K,Tcold variable analysis
"""base_dir = pl.Path(__file__).resolve().parent
path = os.path.join(base_dir, "cycle_simulation_data_Thot")
print(f"Saving data to {path}")
os.makedirs(path, exist_ok=True)
T_cold = np.array([260.15 + i for i in range(26)])
T_hot = np.array([308.15 for _ in range(26)])
COP = []
for i in range(26):
    analysis = Analysis("R134a", T_hot[i], T_cold[i], 1e5, 250)
    analysis.cycle_sim.save_data(path)
    COP.append(analysis.cycle_sim.COP)
# making a data frame for the COP results
df = pd.DataFrame({"T_cold": T_cold, "T_hot": T_hot, "COP": COP})
df.to_excel(os.path.join(path, "COP_results.xlsx"), index=False)
# starting Tcold=268.15K,Thot variable analysis
base_dir = pl.Path(__file__).resolve().parent
path = os.path.join(base_dir, "cycle_simulation_data_Tcold")
os.makedirs(path, exist_ok=True)
T_cold = np.array([268.15 for _ in range(26)])
T_hot = np.array([290.15 + i for i in range(26)])
COP = []
for i in range(26):
    analysis = Analysis("R134a", T_hot[i], T_cold[i], 1e5, 250)
    analysis.cycle_sim.save_data(path)
    COP.append(analysis.cycle_sim.COP)
# making a data frame for the COP results
df = pd.DataFrame({"T_cold": T_cold, "T_hot": T_hot, "COP": COP})
df.to_excel(os.path.join(path, "COP_results.xlsx"), index=False)
# fixied teampretures Tcold=268.15, Thot=308.15, fluids variable analysis
base_dir = pl.Path(__file__).resolve().parent
path = os.path.join(base_dir, "cycle_simulation_data_fluids")
os.makedirs(path, exist_ok=True)
fluids = ["R134a", "R12", "R22", "R717"]
T_cold = 268.15
T_hot = 308.15
COP = []
for fluid in fluids:
    analysis = Analysis(fluid, T_hot, T_cold, 1e5, 250)
    analysis.cycle_sim.save_data(path)
    COP.append(analysis.cycle_sim.COP)
# making a data frame for the COP results
df = pd.DataFrame(
    {
        "Fluid": fluids,
        "T_cold": [T_cold] * len(fluids),
        "T_hot": [T_hot] * len(fluids),
        "COP": COP,
    }
)
df.to_excel(os.path.join(path, "COP_results.xlsx"), index=False)"""
# testing the analysis class analysis mehtods
analysis = Analysis("R134a", 308.15, 268.15, 1e5, 250)
analysis.analyze_pressure()
analysis.analyze_temperature()
analysis.analyze_entropy()
analysis.analyze_work()
analysis.analyze_heat_sucked()
