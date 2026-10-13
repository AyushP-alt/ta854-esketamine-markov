"""Illustrative cohort Markov model for esketamine nasal spray in TRD.

Illustrative model using published TA854 inputs; not a replication.
Comparator is placebo nasal spray plus a newly started oral antidepressant,
the comparison in the company submission for NICE TA854.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


CYCLES_PER_YEAR = 13  # 4-week cycles
CYCLE_YEARS = 1.0 / CYCLES_PER_YEAR
CYCLE_WEEKS = 4.0
SUSTAIN1_EXPOSURE_WEEKS = 17.7

# Clinical states reported in the trace. The engine splits depressed and
# responding so an induction failure is not put back on esketamine.
DEP, RESP, DEAD = 0, 1, 2
STATE_NAMES = ["depressed", "responding", "dead"]
# FDEP: depressed, never responded at induction (not eligible again).
# RDEP: depressed after an induction response (eligible if they re-respond).
# RELIG: responding and eligible for esketamine.
# RINEL: responding after a later transition from the induction-failure pool.
FDEP, RDEP, RELIG, RINEL, DEAD_X = 0, 1, 2, 3, 4


def cycle_prob_from_cumulative(cumulative: float, exposure_weeks: float = SUSTAIN1_EXPOSURE_WEEKS) -> float:
    """Constant-hazard cycle probability from a cumulative incidence."""
    if not 0.0 <= cumulative < 1.0:
        raise ValueError(f"cumulative incidence must be in [0, 1): {cumulative}")
    return 1.0 - (1.0 - cumulative) ** (CYCLE_WEEKS / exposure_weeks)


# SUSTAIN-1 stable remitters, 26.7% and 45.3% over 17.7 weeks (Daly et al. 2019).
P_RELAPSE_ESK = cycle_prob_from_cumulative(0.267)
P_RELAPSE_COMP = cycle_prob_from_cumulative(0.453)


@dataclass
class Inputs:
    """All clinical and cost inputs. See INPUT_LOG for sources."""

    horizon_years: float = 5.0
    discount_rate: float = 0.035
    start_age: float = 46.0
    # ONS national life tables, England, 2018-2020, age 46, sex-weighted
    # toward the SUSTAIN-1 sample (66% female). Approximate qx, not a
    # copied cell from the spreadsheet. Same risk in both arms (committee
    # preferred no treatment effect on mortality).
    annual_qx: float = 0.0016

    # Induction-cycle probability of leaving depression for response/remission.
    # TRANSFORM-2 day-28 response (Popova et al. 2019; NICE TA854 committee papers).
    p_response_esk: float = 0.693
    p_response_comp: float = 0.520
    # Later-cycle re-response, both arms. Illustrative, below the acute
    # TRANSFORM-2 rate and in the range of STAR*D later steps, which the
    # company used for subsequent lines. Equal across arms so this gap is
    # not a source of incremental benefit.
    p_reresponse: float = 0.08

    # 4-week relapse while the esketamine effect is fully on.
    # Derived from SUSTAIN-1 stable-remitter cumulative relapse and the
    # reported median maintenance exposure of 17.7 weeks (Daly et al. 2019):
    # esketamine 26.7%, placebo spray 45.3%. Constant-hazard conversion
    # p = 1 - (1 - F)^(4/17.7). Stored values are that expression, not a
    # hand-rounded substitute. The company model instead used 2.43% and
    # 3.56% per 4 weeks; those are the recovery/recurrence figures in the
    # committee papers, and they are kept as a scenario switch.
    p_relapse_esk: float = P_RELAPSE_ESK
    p_relapse_comp: float = P_RELAPSE_COMP

    # Drug and administration costs stop after this many 4-week cycles
    # (9 cycles = 36 weeks, the duration the company used as a stopping point
    # for many people in stable remission).
    treatment_cycles: int = 9
    # After stopping, fraction of the remaining incremental effect lost
    # each cycle. 0 = company-style maintained effect. 1 = immediate loss.
    waning_per_cycle: float = 0.15

    # Utilities. Depressed: 0.417, TRANSFORM-2 baseline mapped EQ-5D-3L
    # (TA854 committee papers). Responding: weighted average of the
    # committee-paper response (0.764) and remission (0.866) utilities,
    # weights 0.525/0.693 and (0.693-0.525)/0.693 from TRANSFORM-2 remission
    # among responders. Collapsed state, so this weight is a simplification.
    u_depressed: float = 0.417
    u_responding: float = 0.841

    # Health-state costs per 4-week cycle, TA854 committee papers (document 1-2),
    # chart review of 295 UK TRD patients. Response and remission were both £164.
    c_depressed: float = 980.0
    c_responding: float = 164.0

    # List price, BNF online April 2022, quoted in TA854 section 2.3.
    # 84 mg = three 28 mg devices. Excludes VAT. No PAS (confidential).
    price_per_session: float = 489.0
    # Per supervised session. Not a published TA unit cost.
    # Company modelled about 6 patients per nurse; ERG preferred 1:1.
    # £25 is a 6:1 illustration; £150 is a 1:1 illustration (see README).
    admin_per_session: float = 25.0
    # Newly started oral antidepressant, both arms. Illustrative BNF
    # order of magnitude for generic sertraline; cancels in the ICER.
    oral_ad_per_cycle: float = 4.0

    # Non-responders stop esketamine after the induction cycle (SPC:
    # evidence of benefit assessed at the end of induction).
    stop_if_no_response: bool = True


INPUT_LOG = [
    {
        "input": "Population",
        "value": "Adults with treatment-resistant depression; start age 46",
        "source": "TA854 scope; SUSTAIN-1 mean age 46.3 (Daly et al. JAMA Psychiatry 2019)",
    },
    {
        "input": "Comparator",
        "value": "Placebo nasal spray + newly started oral antidepressant",
        "source": "Company submission comparison in TA854 (TRANSFORM-2 design)",
    },
    {
        "input": "Perspective, discount, cycle",
        "value": "NHS, 3.5% costs and QALYs, 4-week cycle, 5-year horizon",
        "source": "NICE reference case; company originally used 5 years (TA854 section 3.24)",
    },
    {
        "input": "Induction response, esketamine",
        "value": "0.693 at day 28",
        "source": "TRANSFORM-2, Popova et al. Am J Psychiatry 2019; cited in TA854 committee papers (69%)",
    },
    {
        "input": "Induction response, comparator",
        "value": "0.520 at day 28",
        "source": "TRANSFORM-2, Popova et al. 2019; TA854 committee papers (52%)",
    },
    {
        "input": "Induction remission (used only to weight the collapsed utility)",
        "value": "0.525 vs 0.310",
        "source": "TRANSFORM-2, Popova et al. 2019; TA854 committee papers (53% vs 31%)",
    },
    {
        "input": "4-week relapse, esketamine arm",
        "value": f"{P_RELAPSE_ESK:.6f}",
        "source": "Derived from SUSTAIN-1 stable remitters: 26.7% relapsed, median exposure 17.7 weeks (Daly et al. JAMA Psychiatry 2019). p = 1-(1-0.267)^(4/17.7). Company model used 2.43% recurrence instead.",
    },
    {
        "input": "4-week relapse, comparator arm",
        "value": f"{P_RELAPSE_COMP:.6f}",
        "source": "Derived from SUSTAIN-1 stable remitters on placebo spray: 45.3% relapsed, same exposure assumption. Company model used 3.56% recurrence, and STAR*D for the comparator (TA854 section 3.21).",
    },
    {
        "input": "Re-response after cycle 1",
        "value": "0.08 both arms",
        "source": "Illustrative. Below TRANSFORM-2 and in the range of STAR*D later-step response, which TA854 used for subsequent lines. Not a copied transition.",
    },
    {
        "input": "Utility, depressed",
        "value": "0.417",
        "source": "TRANSFORM-2 baseline, EQ-5D-5L mapped to 3L, TA854 committee papers",
    },
    {
        "input": "Utility, response / remission",
        "value": "0.764 / 0.866; collapsed base case 0.841",
        "source": "TA854 committee papers. Weighting is a simplification of this 3-state model.",
    },
    {
        "input": "Health-state cost per 4-week cycle",
        "value": "Depressed £980; responding £164",
        "source": "TA854 committee papers, document 1-2, UK chart review (n=295)",
    },
    {
        "input": "Esketamine list price",
        "value": "£163 / £326 / £489 for 28 / 56 / 84 mg",
        "source": "BNF online, April 2022, quoted in TA854 section 2.3. Base case uses 84 mg.",
    },
    {
        "input": "Session schedule",
        "value": "8, then 4, then 2 sessions per 4-week cycle",
        "source": "SmPC / TA854: twice weekly weeks 1-4, weekly weeks 5-8, every 1-2 weeks thereafter. Fortnightly used from cycle 3.",
    },
    {
        "input": "Administration cost per session",
        "value": "£25 company-style; £150 committee-style",
        "source": "Derived from the 6:1 vs 1:1 nurse-ratio dispute in TA854 sections 3.29 and 3.35. Not a published unit cost.",
    },
    {
        "input": "Background mortality",
        "value": "annual qx 0.0016 at age 46, constant",
        "source": "Approximation to ONS national life tables, England, 2018-2020, age 46, weighted to SUSTAIN-1 sex mix. Confirm against the ONS qx spreadsheet before any real use.",
    },
    {
        "input": "Treatment duration and waning",
        "value": "Base: stop costs after 9 cycles; 15% of incremental effect lost per cycle after stopping",
        "source": "Scenario settings, not estimates. Company assumed substantial stopping by 9-24 months and maintained benefit. Committee disputed both (TA854 sections 3.24, 3.29-3.31).",
    },
]


def cycle_death_prob(annual_qx: float) -> float:
    return 1.0 - (1.0 - annual_qx) ** CYCLE_YEARS


def sessions_in_cycle(cycle: int) -> int:
    """SmPC schedule, counted per 4-week cycle."""
    if cycle == 0:
        return 8
    if cycle == 1:
        return 4
    return 2


def effect_remaining(cycle: int, treatment_cycles: int, waning_per_cycle: float) -> float:
    """1 while on treatment, then geometric decay toward the comparator."""
    if cycle < treatment_cycles:
        return 1.0
    cycles_off = cycle - treatment_cycles + 1
    retain = max(0.0, 1.0 - waning_per_cycle)
    return retain ** cycles_off


def arm_rates(cycle: int, esk: bool, inp: Inputs, waning_start_shift: int = 0) -> tuple[float, float]:
    """Return (p_to_responding, p_relapse) before mortality is applied.

    waning_start_shift of -1 uses the next cycle's effect, which is how the
    Excel trace applies waning one transition early.
    """
    if cycle == 0:
        p_resp = inp.p_response_esk if esk else inp.p_response_comp
    else:
        p_resp = inp.p_reresponse
    p_rel = inp.p_relapse_comp
    if esk:
        w = effect_remaining(cycle - waning_start_shift, inp.treatment_cycles, inp.waning_per_cycle)
        # Relapse advantage applies only while the effect remains.
        p_rel = inp.p_relapse_comp + (inp.p_relapse_esk - inp.p_relapse_comp) * w
        # Re-response is equal by default. Induction difference is cycle 0 only,
        # and cycle 0 is always on treatment.
    return p_resp, p_rel


def transition_matrix(
    p_respond: float,
    p_relapse_eligible: float,
    p_relapse_ineligible: float,
    p_death: float,
    induction: bool,
) -> np.ndarray:
    """Rows sum to 1. Death is a competing risk on every transition.

    The split remembers whether a response began at induction. RELIG relapses
    at p_relapse_eligible. RINEL relapses at p_relapse_ineligible, because
    those patients failed induction and never received esketamine.
    """
    stay_alive = 1.0 - p_death
    respond_to = RELIG if induction else RINEL
    P = np.zeros((5, 5))
    P[FDEP, respond_to] = p_respond * stay_alive
    P[FDEP, FDEP] = (1.0 - p_respond) * stay_alive
    P[FDEP, DEAD_X] = p_death
    P[RDEP, RELIG] = p_respond * stay_alive
    P[RDEP, RDEP] = (1.0 - p_respond) * stay_alive
    P[RDEP, DEAD_X] = p_death
    P[RELIG, RDEP] = p_relapse_eligible * stay_alive
    P[RELIG, RELIG] = (1.0 - p_relapse_eligible) * stay_alive
    P[RELIG, DEAD_X] = p_death
    P[RINEL, FDEP] = p_relapse_ineligible * stay_alive
    P[RINEL, RINEL] = (1.0 - p_relapse_ineligible) * stay_alive
    P[RINEL, DEAD_X] = p_death
    P[DEAD_X, DEAD_X] = 1.0
    return P


def _clinical(state: np.ndarray) -> tuple[float, float, float]:
    depressed = float(state[FDEP] + state[RDEP])
    responding = float(state[RELIG] + state[RINEL])
    dead = float(state[DEAD_X])
    return depressed, responding, dead


def discount_factor(cycle: int, rate: float) -> float:
    """Mid-cycle discount, consistent with the half-cycle correction."""
    year = (cycle + 0.5) * CYCLE_YEARS
    return 1.0 / (1.0 + rate) ** year


def run_arm(
    esk: bool,
    inp: Inputs,
    drop_last_half_cycle: bool = False,
    waning_start_shift: int = 0,
    half_cycle_costs: bool = True,
) -> dict:
    n = int(round(inp.horizon_years * CYCLES_PER_YEAR))
    p_death = cycle_death_prob(inp.annual_qx)
    state = np.zeros(5)
    state[FDEP] = 1.0
    rows = []
    cost_d = qaly_d = ly_d = 0.0
    cost_u = qaly_u = ly_u = 0.0
    drug_cost = admin_cost = state_cost = oral_cost = 0.0

    for t in range(n):
        p_resp, p_rel = arm_rates(t, esk, inp, waning_start_shift=waning_start_shift)
        # RINEL failed induction, so it keeps the comparator relapse rate.
        P = transition_matrix(
            p_resp,
            p_relapse_eligible=p_rel,
            p_relapse_ineligible=inp.p_relapse_comp,
            p_death=p_death,
            induction=(t == 0),
        )
        if not np.allclose(P.sum(axis=1), 1.0):
            raise RuntimeError(f"Row sums not 1 at cycle {t}")
        nxt = state @ P
        # Excel's last row sets HalfQALY = S/2, dropping the post-transition state.
        if drop_last_half_cycle and t == n - 1:
            occ = 0.5 * state
        else:
            occ = 0.5 * (state + nxt)  # half-cycle correction
        if not (drop_last_half_cycle and t == n - 1) and abs(occ.sum() - 1.0) > 1e-8:
            raise RuntimeError(f"Occupancy not 1 at cycle {t}: {occ.sum()}")

        dep, resp, _dead = _clinical(occ)
        # Excel half-cycles QALYs only. State and oral costs use the cycle-start stock.
        cost_dep, cost_resp, _ = _clinical(occ if half_cycle_costs else state)
        df = discount_factor(t, inp.discount_rate)
        q_cycle = (dep * inp.u_depressed + resp * inp.u_responding) * CYCLE_YEARS
        ly_cycle = (dep + resp) * CYCLE_YEARS
        hs_cost = cost_dep * inp.c_depressed + cost_resp * inp.c_responding
        oral = inp.oral_ad_per_cycle * (cost_dep + cost_resp)

        # Cycle 0 treats the whole living cohort. After that, only people
        # who responded at induction and are responding now stay on drug.
        eligible = float(state[RELIG])
        alive = float(state[FDEP] + state[RDEP] + state[RELIG] + state[RINEL])
        on_tx = 0.0
        if esk and t < inp.treatment_cycles:
            if t == 0 or not inp.stop_if_no_response:
                on_tx = alive
            else:
                on_tx = eligible
        if esk and inp.stop_if_no_response and t > 0 and on_tx - eligible > 1e-9:
            raise RuntimeError(f"Induction failures were treated at cycle {t}")
        sess = sessions_in_cycle(t) * on_tx
        drug = sess * inp.price_per_session
        admin = sess * inp.admin_per_session
        cyc_cost = hs_cost + oral + drug + admin

        cost_d += cyc_cost * df
        qaly_d += q_cycle * df
        ly_d += ly_cycle * df
        cost_u += cyc_cost
        qaly_u += q_cycle
        ly_u += ly_cycle
        drug_cost += drug * df
        admin_cost += admin * df
        state_cost += hs_cost * df
        oral_cost += oral * df

        dep_s, resp_s, dead_s = _clinical(state)
        rows.append(
            {
                "cycle": t,
                "depressed": dep_s,
                "responding": resp_s,
                "dead": dead_s,
                "on_treatment": on_tx,
                "eligible": eligible,
                "p_respond": p_resp,
                "p_relapse": p_rel,
                "effect": effect_remaining(t - waning_start_shift, inp.treatment_cycles, inp.waning_per_cycle) if esk else 0.0,
                "occ_depressed": dep,
                "occ_responding": resp,
                "cycle_cost": cyc_cost,
                "cycle_qaly": q_cycle,
                "disc_cost": cyc_cost * df,
                "disc_qaly": q_cycle * df,
            }
        )
        state = nxt

    return {
        "cost": cost_d,
        "qaly": qaly_d,
        "ly": ly_d,
        "cost_undiscounted": cost_u,
        "qaly_undiscounted": qaly_u,
        "ly_undiscounted": ly_u,
        "drug_cost": drug_cost,
        "admin_cost": admin_cost,
        "state_cost": state_cost,
        "oral_cost": oral_cost,
        "trace": pd.DataFrame(rows),
        "final_state_sum": float(state.sum()),
    }


def incremental(
    inp: Inputs,
    drop_last_half_cycle: bool = False,
    waning_start_shift: int = 0,
    half_cycle_costs: bool = True,
) -> dict:
    esk = run_arm(
        True, inp,
        drop_last_half_cycle=drop_last_half_cycle,
        waning_start_shift=waning_start_shift,
        half_cycle_costs=half_cycle_costs,
    )
    comp = run_arm(
        False, inp,
        drop_last_half_cycle=drop_last_half_cycle,
        waning_start_shift=waning_start_shift,
        half_cycle_costs=half_cycle_costs,
    )
    inc_c = esk["cost"] - comp["cost"]
    inc_q = esk["qaly"] - comp["qaly"]
    icer = inc_c / inc_q if abs(inc_q) > 1e-12 else math.inf
    return {
        "esketamine": esk,
        "comparator": comp,
        "inc_cost": inc_c,
        "inc_qaly": inc_q,
        "icer": icer,
        "inputs": inp,
        "drop_last_half_cycle": drop_last_half_cycle,
        "waning_start_shift": waning_start_shift,
        "half_cycle_costs": half_cycle_costs,
    }


class _Run:
    """Thin result so a switch loop can read r.icer."""

    def __init__(self, result: dict):
        self.icer = result["icer"]
        self.inc_cost = result["inc_cost"]
        self.inc_qaly = result["inc_qaly"]
        self.result = result


def run_model(
    inputs: Inputs,
    drop_last_half_cycle: bool = False,
    waning_start_shift: int = 0,
    half_cycle_costs: bool = True,
) -> _Run:
    return _Run(
        incremental(
            inputs,
            drop_last_half_cycle=drop_last_half_cycle,
            waning_start_shift=waning_start_shift,
            half_cycle_costs=half_cycle_costs,
        )
    )


def checks(result: dict) -> list[str]:
    notes = []
    for name in ("esketamine", "comparator"):
        arm = result[name]
        notes.append(
            f"{name}: final state sum {arm['final_state_sum']:.6f}; "
            f"discounted QALYs {arm['qaly']:.3f} vs life-years {arm['ly']:.3f}"
        )
        if arm["qaly"] >= arm["ly"] + 1e-9:
            notes.append(f"FAIL: {name} QALYs exceed life-years")
        if abs(arm["final_state_sum"] - 1.0) > 1e-6:
            notes.append(f"FAIL: {name} state probabilities do not sum to 1")
        if not np.allclose(arm["trace"][["depressed", "responding", "dead"]].sum(axis=1), 1.0):
            notes.append(f"FAIL: {name} cycle occupancy does not sum to 1")
    esk = result["esketamine"]["trace"]
    if result["inputs"].stop_if_no_response:
        window = esk[(esk["cycle"] > 0) & (esk["cycle"] < result["inputs"].treatment_cycles)]
        if not np.allclose(window["on_treatment"], window["eligible"]):
            notes.append("FAIL: esketamine restarts treatment for induction non-responders")
        if (window["eligible"] - window["responding"] > 1e-9).any():
            notes.append("FAIL: eligible cohort exceeds the responding state")
    return notes


def one_way(base: Inputs, spec: list[tuple]) -> pd.DataFrame:
    """spec items: (label, attribute, low, high)."""
    base_icer = incremental(base)["icer"]
    rows = []
    for label, attr, low, high in spec:
        for tag, value in (("low", low), ("high", high)):
            inp = Inputs(**{**asdict(base), attr: value})
            res = incremental(inp)
            rows.append(
                {
                    "parameter": label,
                    "bound": tag,
                    "value": value,
                    "inc_cost": res["inc_cost"],
                    "inc_qaly": res["inc_qaly"],
                    "icer": res["icer"],
                    "delta_icer": res["icer"] - base_icer,
                }
            )
    return pd.DataFrame(rows)


def tornado_spans(owsa: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for param, g in owsa.groupby("parameter", sort=False):
        lo = g.loc[g["bound"] == "low"].iloc[0]
        hi = g.loc[g["bound"] == "high"].iloc[0]
        rows.append(
            {
                "parameter": param,
                "icer_low": lo["icer"],
                "icer_high": hi["icer"],
                "span": abs(hi["icer"] - lo["icer"]),
                "value_low": lo["value"],
                "value_high": hi["value"],
            }
        )
    return pd.DataFrame(rows).sort_values("span", ascending=False)


def plot_tornado(spans: pd.DataFrame, base_icer: float, path: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    y = np.arange(len(spans))
    for i, row in enumerate(spans.itertuples(index=False)):
        left = min(row.icer_low, row.icer_high)
        width = abs(row.icer_high - row.icer_low)
        ax.barh(i, width, left=left, color="#1f4e79", height=0.62)
        ax.plot([row.icer_low, row.icer_high], [i, i], color="#1f4e79", lw=0)
    ax.axvline(base_icer, color="#b45309", lw=1.4, label=f"Base case £{base_icer:,.0f}")
    ax.axvspan(20000, 30000, color="#f5d0a9", alpha=0.45, label="£20,000–£30,000")
    labels = []
    for row in spans.itertuples(index=False):
        labels.append(f"{row.parameter}\n({row.value_low:g} to {row.value_high:g})")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("ICER (£ per QALY)")
    ax.set_title("One-way sensitivity: illustrative TA854 cohort model")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="x", color="#e5e7eb")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_trace(result: dict, path: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    for name, color in (("esketamine", "#1f4e79"), ("comparator", "#9a3412")):
        tr = result[name]["trace"]
        ax.plot(tr["cycle"] * 4 / 52, tr["responding"], color=color, label=f"{name}, responding")
        ax.plot(tr["cycle"] * 4 / 52, tr["depressed"], color=color, ls="--", label=f"{name}, depressed")
    ax.set_xlabel("Years")
    ax.set_ylabel("State probability")
    ax.set_title("State occupancy; the whole cohort starts depressed")
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.set_ylim(0, 1)
    ax.grid(color="#e5e7eb")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def scenario_table(scenarios: dict[str, Inputs]) -> pd.DataFrame:
    rows = []
    for name, inp in scenarios.items():
        res = incremental(inp)
        rows.append(
            {
                "scenario": name,
                "treatment_cycles": inp.treatment_cycles,
                "waning_per_cycle": inp.waning_per_cycle,
                "admin_per_session": inp.admin_per_session,
                "p_relapse_esk": inp.p_relapse_esk,
                "p_relapse_comp": inp.p_relapse_comp,
                "price_per_session": inp.price_per_session,
                "inc_cost": res["inc_cost"],
                "inc_qaly": res["inc_qaly"],
                "icer": res["icer"],
                "esk_cost": res["esketamine"]["cost"],
                "comp_cost": res["comparator"]["cost"],
                "esk_qaly": res["esketamine"]["qaly"],
                "comp_qaly": res["comparator"]["qaly"],
            }
        )
    return pd.DataFrame(rows)
