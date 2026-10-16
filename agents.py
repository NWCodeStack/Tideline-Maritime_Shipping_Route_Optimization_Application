"""
agents.py - behavioural profiles for the decision-makers in the simulation.

Two kinds of agent shape every routing decision:

* a **VesselProfile**  - the operator of the ship (Container line, Oil-tanker owner)
* a **CargoOwnerProfile** - whoever owns the goods on board (High-Margin / Low-Margin retailer, energy trader)

``RoutingAgent`` fuses one of each into the handful of numbers the routing engine needs
(delay penalty per hour, insurance sensitivity, toll weight, speed options, avoidance threshold).

Profile matrix (what each persona cares about)
----------------------------------------------
                       time pressure   toll tolerance   war-risk sensitivity   walks away from conflict zone
Container ship         very high       pays to save     high                   when premium > 3x
Oil tanker             low             neutral          extreme                when premium > 4x (rarely has a choice)
High-margin retailer   very high       sponsors speed   -                      -
Low-margin retailer    low             hates tolls      -                      -
Energy trader          moderate        neutral          -                      -

All dollar figures are illustrative assumptions for the demo, not market quotes.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

# War-risk premium charged per unit of geopolitical risk, per day of exposure, as a fraction of insured value.
WAR_RISK_RATE_PER_RISK_DAY = 0.0015


def premium_multiple(peak_risk: float) -> float:
    """How many times the normal insurance premium a lane costs at its riskiest point.

    1.0x on a calm lane, ~1.9x at Hormuz-like baseline risk (0.4), ~3x at risk 0.43, ~12x at risk 1.0.
    Quadratic so that insurers react sharply once a lane stops being merely "tense".
    """
    return 1.0 + 11.0 * float(peak_risk) ** 2


def draft_for_load(vessel_type: str, cargo_weight_t: float) -> float:
    """Approximate draft (m) of a ship carrying ``cargo_weight_t`` - same mapping the ML data used."""
    from ml_models import HULL
    h = HULL[vessel_type]
    lo, hi = h["cargo"]
    load = min(max((cargo_weight_t - lo) / (hi - lo), 0.0), 1.0)
    return round(h["draft"][0] + (h["draft"][1] - h["draft"][0]) * load, 1)


# --------------------------------------------------------------------------------------------------
# Profiles
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class VesselProfile:
    key: str
    label: str
    vessel_type: str                      # "Container" | "Tanker"  (matches the ML model's categories)
    speed_options_kn: tuple               # speeds the ship may choose per lane (slow-steam ... full speed)
    daily_asset_cost_usd: float           # charter + crew + capital cost per day at sea
    hull_value_usd: float
    war_risk_aversion: float              # multiplies the insurer's war-risk charge (1 = typical)
    max_premium_multiple: float           # refuses lanes whose insurance premium exceeds this multiple
    toll_tolerance: float                 # <1 operator happily pays tolls to keep schedule, >1 resists
    description: str = ""


@dataclass(frozen=True)
class CargoOwnerProfile:
    key: str
    label: str
    cargo: str
    cargo_weight_t: float
    cargo_value_usd: float
    late_penalty_usd_per_hr: float        # stock-out / contract penalties / capital tied up, per hour of delay
    toll_weight: float                    # <1: owner sponsors fast routing; >1: owner insists on toll avoidance
    description: str = ""


VESSELS: dict[str, VesselProfile] = {
    "container": VesselProfile(
        key="container", label="Container ship", vessel_type="Container",
        speed_options_kn=(16.0, 19.0, 22.0), daily_asset_cost_usd=85_000, hull_value_usd=150_000_000,
        war_risk_aversion=1.0, max_premium_multiple=3.0, toll_tolerance=0.6,
        description="Time-critical liner service: heavy daily asset cost, pays tolls to hold the schedule, "
                    "refuses conflict areas once insurance spikes beyond 3x.",
    ),
    "tanker": VesselProfile(
        key="tanker", label="Oil tanker", vessel_type="Tanker",
        speed_options_kn=(11.0, 12.5, 14.0), daily_asset_cost_usd=45_000, hull_value_usd=80_000_000,
        war_risk_aversion=4.0, max_premium_multiple=4.0, toll_tolerance=1.0,
        description="Slow and heavy: huge cargo value and extreme war-risk premiums make every risky mile "
                    "expensive, while time matters far less.",
    ),
    # The fuel model was trained on two hull classes; bulk carriers borrow the tanker hull (full-bodied, slow)
    # and LNG carriers the container hull (fine-lined, fast). Fuel figures for them are labelled as estimates.
    "bulk": VesselProfile(
        key="bulk", label="Bulk carrier", vessel_type="Tanker",
        speed_options_kn=(11.0, 12.5, 14.0), daily_asset_cost_usd=28_000, hull_value_usd=35_000_000,
        war_risk_aversion=1.5, max_premium_multiple=3.5, toll_tolerance=1.3,
        description="Capesize-style dry bulk: cheap to run, low-value cargo, avoids tolls where it can. "
                    "Fuel estimated with the tanker hull class.",
    ),
    "lng": VesselProfile(
        key="lng", label="LNG carrier", vessel_type="Container",
        speed_options_kn=(16.0, 18.0, 19.5), daily_asset_cost_usd=95_000, hull_value_usd=220_000_000,
        war_risk_aversion=3.0, max_premium_multiple=3.0, toll_tolerance=0.9,
        description="High-value gas carrier on a tight delivery window; very war-risk averse. "
                    "Fuel estimated with the container hull class.",
    ),
}

CARGO_OWNERS: dict[str, CargoOwnerProfile] = {
    "high_margin": CargoOwnerProfile(
        key="high_margin", label="High-margin retailer (Tech / Fashion)", cargo="Apple electronics",
        cargo_weight_t=70_000, cargo_value_usd=250_000_000, late_penalty_usd_per_hr=9_000, toll_weight=0.35,
        description="Launch windows and stock-outs make lateness brutal; will sponsor faster routing and tolls.",
    ),
    "low_margin": CargoOwnerProfile(
        key="low_margin", label="Low-margin retailer (Bulk / IKEA)", cargo="IKEA flat-pack furniture",
        cargo_weight_t=95_000, cargo_value_usd=40_000_000, late_penalty_usd_per_hr=600, toll_weight=2.5,
        description="Razor-thin margins: every dollar of toll hurts more than a few extra days at sea.",
    ),
    "energy_trader": CargoOwnerProfile(
        key="energy_trader", label="Energy trader", cargo="Crude oil", cargo_weight_t=150_000,
        cargo_value_usd=78_000_000, late_penalty_usd_per_hr=1_200, toll_weight=1.0,
        description="Pays mostly the cost of capital tied up in the cargo while it is afloat.",
    ),
    "bulk_trader": CargoOwnerProfile(
        key="bulk_trader", label="Dry-bulk trader", cargo="Iron ore", cargo_weight_t=150_000,
        cargo_value_usd=18_000_000, late_penalty_usd_per_hr=300, toll_weight=1.3,
        description="Low value per tonne: lateness is cheap, tolls are not.",
    ),
    "lng_buyer": CargoOwnerProfile(
        key="lng_buyer", label="LNG utility buyer", cargo="Liquefied natural gas", cargo_weight_t=75_000,
        cargo_value_usd=70_000_000, late_penalty_usd_per_hr=2_500, toll_weight=0.8,
        description="Winter supply contract: late cargoes are expensive to replace on the spot market.",
    ),
}

# Which cargo owners make sense for each vessel (cargo weight must fit the hull class used by the fuel model).
VESSEL_OWNERS: dict[str, tuple] = {
    "container": ("high_margin", "low_margin"),
    "tanker": ("energy_trader",),
    "bulk": ("bulk_trader",),
    "lng": ("lng_buyer",),
}


@dataclass(frozen=True)
class RoutingAgent:
    """One vessel operator carrying one owner's cargo - the unit the routing engine optimises for."""
    vessel: VesselProfile
    owner: CargoOwnerProfile

    @property
    def name(self) -> str:
        return f"{self.vessel.label} - {self.owner.cargo}"

    @property
    def vessel_type(self) -> str:
        return self.vessel.vessel_type

    @property
    def cargo_weight_t(self) -> float:
        return self.owner.cargo_weight_t

    @property
    def draft_m(self) -> float:
        return draft_for_load(self.vessel.vessel_type, self.owner.cargo_weight_t)

    @property
    def speed_options_kn(self) -> tuple:
        return self.vessel.speed_options_kn

    @property
    def delay_penalty_usd_per_hr(self) -> float:
        """Cost of one more hour in transit: the ship's own daily cost plus the cargo owner's lateness penalty."""
        return self.vessel.daily_asset_cost_usd / 24.0 + self.owner.late_penalty_usd_per_hr

    @property
    def insurance_sensitivity_usd(self) -> float:
        """USD charged per unit of geopolitical risk per day of exposure (hull + cargo insured value)."""
        insured = self.vessel.hull_value_usd + self.owner.cargo_value_usd
        return insured * WAR_RISK_RATE_PER_RISK_DAY * self.vessel.war_risk_aversion

    @property
    def toll_weight(self) -> float:
        """How heavily a dollar of toll counts in the objective (operator tolerance x owner attitude)."""
        return self.vessel.toll_tolerance * self.owner.toll_weight

    @property
    def max_premium_multiple(self) -> float:
        return self.vessel.max_premium_multiple


def make_agent(vessel_key: str, owner_key: str, cruise_speed_kn: float | None = None) -> RoutingAgent:
    """``cruise_speed_kn`` pins the ship to one speed instead of letting the optimiser pick per lane."""
    vessel = VESSELS[vessel_key]
    if cruise_speed_kn:
        vessel = replace(vessel, speed_options_kn=(float(cruise_speed_kn),))
    return RoutingAgent(vessel, CARGO_OWNERS[owner_key])


# The two ships compared side by side in the app, plus the alternative owner the user can swap in.
DEFAULT_CONTAINER = make_agent("container", "high_margin")
DEFAULT_TANKER = make_agent("tanker", "energy_trader")


def profile_matrix() -> pd.DataFrame:
    """The behavioural matrices as a table (one column per persona) for display in the UI / README."""
    rows: dict[str, dict] = {}
    for v in VESSELS.values():
        a = make_agent(v.key, VESSEL_OWNERS[v.key][0])
        rows[v.label] = {
            "Kind": "Vessel operator",
            "Speed options (kn)": " / ".join(f"{s:g}" for s in v.speed_options_kn),
            "Asset cost ($/day)": f"{v.daily_asset_cost_usd:,.0f}",
            "War-risk aversion": f"{v.war_risk_aversion:g}x",
            "Avoids lanes above": f"{v.max_premium_multiple:g}x premium",
            "Toll tolerance": f"{v.toll_tolerance:g} (lower = pays)",
            "Insurance $/risk-day (typical)": f"{a.insurance_sensitivity_usd:,.0f}",
        }
    for o in CARGO_OWNERS.values():
        rows[o.label] = {
            "Kind": "Cargo owner",
            "Cargo": o.cargo,
            "Cargo value ($)": f"{o.cargo_value_usd:,.0f}",
            "Late penalty ($/hr)": f"{o.late_penalty_usd_per_hr:,.0f}",
            "Toll weight": f"{o.toll_weight:g} (lower = sponsors)",
        }
    return pd.DataFrame(rows).fillna("-")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print(profile_matrix().T.to_string())
    for ag in (DEFAULT_CONTAINER, DEFAULT_TANKER, make_agent("container", "low_margin")):
        print(f"\n{ag.name}: draft {ag.draft_m} m, delay ${ag.delay_penalty_usd_per_hr:,.0f}/h, "
              f"insurance ${ag.insurance_sensitivity_usd:,.0f}/risk-day, toll weight {ag.toll_weight:.2f}")
