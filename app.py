"""Interactive decision-support dashboard for the illustrative TA854 model."""

from __future__ import annotations

from dataclasses import asdict, replace
import math

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import model
from dashboard_logic import economic_summary, find_nmb_zero_by_search, split_diagnostics, validate_probability


REFERENCE_PRICE = model.Inputs().price_per_session
WTP_CONTEXTS = {
    "Historical TA854 context · £20,000–£30,000/QALY": (20_000, 30_000, 25_000),
    "Current standard NICE context · £25,000–£35,000/QALY": (25_000, 35_000, 30_000),
}


def make_trace_figure(result: dict, horizon_years: float) -> go.Figure:
    """Plot model cycle-start occupancy for each reported clinical state."""
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09,
        subplot_titles=("Depressed", "Responding", "Dead"),
    )
    state_colors = {"depressed": "#355C7D", "responding": "#2A7F78", "dead": "#8A94A6"}
    strategies = (("esketamine", "Esketamine + oral antidepressant", "solid"),
                  ("comparator", "Comparator + oral antidepressant", "dash"))
    x_max = max(0.0, horizon_years)
    for key, label, dash in strategies:
        trace = result[key]["trace"]
        years = trace["cycle"] / model.CYCLES_PER_YEAR
        for row, state in enumerate(("depressed", "responding", "dead"), start=1):
            fig.add_trace(
                go.Scatter(
                    x=years,
                    y=trace[state],
                    mode="lines",
                    name=f"{label} · {state.title()}",
                    legendgroup=key,
                    line={"color": state_colors[state], "dash": dash, "width": 2.4},
                    hovertemplate="Year %{x:.2f}<br>%{y:.1%} of cohort<extra>%{fullData.name}</extra>",
                ),
                row=row,
                col=1,
            )
            fig.update_yaxes(title_text="Cohort share", tickformat=".0%", range=[0, 1], row=row, col=1)
    fig.update_xaxes(title_text="Model time (years; 13 four-week cycles per year)", range=[0, x_max], row=3, col=1)
    fig.update_layout(
        height=700,
        margin={"l": 20, "r": 20, "t": 65, "b": 25},
        legend={"orientation": "h", "y": -0.15},
        hovermode="x unified",
    )
    fig.update_annotations(font={"size": 13, "color": "#233044"})
    return fig


def make_ce_plane(points: list[dict], wtp: float) -> go.Figure:
    """Plot deterministic incremental outcomes and the selected WTP line."""
    xs = [float(p["inc_qaly"]) for p in points]
    ys = [float(p["inc_cost"]) for p in points]
    max_x = max([abs(x) for x in xs] + [0.03])
    max_y = max([abs(y) for y in ys] + [max_x * wtp, 2000.0])
    x_pad = max_x * 0.25
    x_min, x_max = min(0.0, min(xs)) - x_pad, max(0.0, max(xs)) + x_pad
    max_y = max(max_y, abs(wtp * x_min), abs(wtp * x_max))
    line_x = [x_min, x_max]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=line_x,
        y=[wtp * x for x in line_x],
        mode="lines",
        name=f"Selected WTP · £{wtp:,.0f}/QALY",
        line={"color": "#D18A3B", "dash": "dash", "width": 2},
        hovertemplate="WTP reference<br>ΔQALY %{x:.3f}<br>ΔCost £%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=xs,
        y=ys,
        mode="markers+text",
        text=[p["label"] for p in points],
        textposition="top center",
        name="Deterministic model runs",
        marker={"size": 11, "color": ["#244C66" if p.get("kind") == "base" else "#2A7F78" for p in points]},
        customdata=[[p["label"]] for p in points],
        hovertemplate="%{customdata[0]}<br>Incremental QALYs %{x:.4f}<br>Incremental cost £%{y:,.0f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="#AAB3BF", line_width=1)
    fig.add_vline(x=0, line_color="#AAB3BF", line_width=1)
    quadrant_style = {"showarrow": False, "font": {"size": 10, "color": "#64748B"}}
    fig.add_annotation(x=0.98, y=0.98, xref="paper", yref="paper", text="More effective · more costly", xanchor="right", **quadrant_style)
    fig.add_annotation(x=0.02, y=0.98, xref="paper", yref="paper", text="Less effective · more costly", xanchor="left", **quadrant_style)
    fig.add_annotation(x=0.98, y=0.02, xref="paper", yref="paper", text="More effective · less costly", xanchor="right", **quadrant_style)
    fig.add_annotation(x=0.02, y=0.02, xref="paper", yref="paper", text="Less effective · less costly", xanchor="left", **quadrant_style)
    fig.update_layout(
        height=430,
        margin={"l": 20, "r": 20, "t": 25, "b": 35},
        xaxis_title="Incremental QALYs · esketamine minus comparator",
        yaxis_title="Incremental cost · £",
        hovermode="closest",
        legend={"orientation": "h", "y": -0.2},
    )
    fig.update_xaxes(range=[x_min, x_max], zeroline=False)
    fig.update_yaxes(range=[-max_y * 0.12, max_y * 1.12], zeroline=False)
    return fig


def _input_from_controls(
    horizon: int,
    price: float,
    admin: float,
    response_esk: float,
    relapse_esk: float,
    relapse_comp: float,
    treatment_cycles: int,
    waning: float,
) -> model.Inputs:
    return replace(
        model.Inputs(),
        horizon_years=float(horizon),
        price_per_session=float(price),
        admin_per_session=float(admin),
        p_response_esk=validate_probability(response_esk, "Esketamine induction response probability"),
        p_relapse_esk=validate_probability(relapse_esk, "Esketamine relapse probability"),
        p_relapse_comp=validate_probability(relapse_comp, "Comparator relapse probability"),
        treatment_cycles=int(treatment_cycles),
        waning_per_cycle=validate_probability(waning, "Waning per cycle"),
    )


def _run_summary_row(name: str, result: dict, wtp: float) -> dict:
    summary = economic_summary(result, wtp)
    return {
        "Scenario": name,
        "Incremental cost (£)": summary["inc_cost"],
        "Incremental QALYs": summary["inc_qaly"],
        "ICER (£/QALY)": summary["icer_display"],
        "NMB (£)": summary["nmb"],
    }


def _input_cache_key(inputs: model.Inputs) -> tuple:
    """Hash model inputs independently of WTP, which is an interpretation setting."""
    return tuple(asdict(inputs).items())


@st.cache_data(show_spinner=False)
def _cached_incremental(input_items: tuple) -> dict:
    """Reuse deterministic engine outputs when only the WTP control changes."""
    return model.incremental(model.Inputs(**dict(input_items)))


@st.cache_data(show_spinner=False)
def _cached_one_way(input_items: tuple, spec_items: tuple) -> pd.DataFrame:
    """Cache model-generated one-way endpoints; WTP does not affect these runs."""
    return model.one_way(model.Inputs(**dict(input_items)), list(spec_items))


st.set_page_config(page_title="Esketamine value & access explorer", layout="wide")
st.markdown(
    """
    <style>
      .stApp, [data-testid="stAppViewContainer"] { background: #F5F7FA; color: #1F2937; }
      [data-testid="stHeader"] { background: #F5F7FA; }
      .block-container { max-width: 1440px; padding-top: 1.35rem; padding-bottom: 3rem; }
      h1, h2, h3, h4 { color: #19364D; letter-spacing: -0.02em; }
      [data-testid="stMarkdownContainer"] p,
      [data-testid="stMarkdownContainer"] li,
      [data-testid="stMarkdownContainer"] blockquote { color: #334155; }
      [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: #586779 !important; opacity: 1; }
      [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p,
      [data-testid="stWidgetLabel"] label { color: #334155 !important; opacity: 1; }
      [data-testid="stTooltipIcon"] { color: #586779 !important; opacity: 1; }
      [data-testid="stTooltipIcon"] svg { fill: #586779; color: #586779; opacity: 1; }
      [data-testid="stSidebar"] { background: #FFF7F7; }
      [data-testid="stTabs"] [data-baseweb="tab-list"] { gap: .45rem; border-bottom: 1px solid #E1E7ED; }
      [data-testid="stTabs"] button[data-baseweb="tab"] { color: #8F1826 !important; background: #FFF0F1; border: 1px solid #E5A5AC; border-radius: 6px 6px 0 0; font-weight: 650; padding: .45rem .9rem; opacity: 1; }
      [data-testid="stTabs"] button[data-baseweb="tab"] p { color: #8F1826 !important; opacity: 1; }
      [data-testid="stTabs"] button[data-baseweb="tab"]:hover { color: #7F1D1D !important; background: #FFE1E4; border-color: #C85A66; }
      [data-testid="stTabs"] button[data-baseweb="tab"]:hover p { color: #7F1D1D !important; }
      [data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] { color: #FFFFFF !important; background: #B42332; border-color: #B42332; border-bottom: 3px solid #7F1D1D; }
      [data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] p { color: #FFFFFF !important; }
      [data-testid="stTabs"] button[data-baseweb="tab"]:focus-visible { outline: 3px solid #1D4ED8; outline-offset: 2px; }
      [data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #E1E7ED; border-radius: 8px; padding: 14px 16px; }
      [data-testid="stMetricValue"], [data-testid="stMetricValue"] div,
      [data-testid="stMetricValue"] span { color: #172B3A !important; opacity: 1; }
      [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] div,
      [data-testid="stMetricLabel"] p { color: #586779 !important; opacity: 1; }
      [data-testid="stMetricDelta"], [data-testid="stMetricDelta"] div,
      [data-testid="stMetricDelta"] span { color: #334155 !important; opacity: 1; }
      [data-testid="stMetric"]:hover { background: #FFFFFF; }
      .eyebrow { color: #527083; font-size: .78rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
      .note { color: #586779; font-size: .92rem; }
      .disclaimer { background: #EDF2F6; border-left: 4px solid #527083; padding: 12px 16px; border-radius: 4px; color: #334155; }
      [data-testid="stAppViewContainer"] button:focus-visible,
      [data-testid="stAppViewContainer"] input:focus-visible,
      [data-testid="stAppViewContainer"] [role="combobox"]:focus-visible { outline: 3px solid #1D4ED8; outline-offset: 2px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="eyebrow">Illustrative health economics · NICE TA854 context</div>', unsafe_allow_html=True)
st.title("Esketamine | Value & Access Explorer")
st.markdown(
    '<div class="disclaimer"><strong>Illustrative decision-support model.</strong> '
    'This is not the manufacturer model, ERG model, or an official NICE estimate. '
    'It uses historical list-price assumptions and does not model the confidential patient access scheme. '
    'Model-based cost-effectiveness is not a coverage or reimbursement decision.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Decision context")
    horizon = st.selectbox("Time horizon", [1, 5, 10], index=1, format_func=lambda y: f"{y} years")
    st.caption("The model accrues 13 four-week cycles per year: 13, 65, or 130 cycles. Costs and QALYs cover every selected cycle using mid-cycle discounting. Trace points show cycle-start occupancy, so the final point precedes the horizon endpoint.")
    wtp_context_label = st.selectbox("WTP policy context", list(WTP_CONTEXTS), index=0)
    wtp_min, wtp_max, wtp_default = WTP_CONTEXTS[wtp_context_label]
    wtp = st.slider(
        "WTP threshold (£/QALY)",
        min_value=wtp_min,
        max_value=wtp_max,
        value=wtp_default,
        step=1_000,
        format="£%d",
    )
    wtp_context = wtp_context_label.split(" · ")[0]
    st.caption("The current NICE standard range is separate policy context and does not retrospectively change the original TA854 appraisal." if wtp_context.startswith("Historical") else "Current NICE standard range, shown as separate policy context rather than a retrospective change to TA854.")

tab_overview, tab_price, tab_sensitivity, tab_evidence = st.tabs(
    ["Value Overview", "Price & Access Explorer", "Sensitivity Analysis", "Model & Evidence"]
)

with tab_price:
    st.subheader("Scenario inputs")
    st.markdown("Adjust the model’s actual per-session price and documented clinical assumptions. These controls drive the same live model run used in the overview and charts.")
    reference_price = model.Inputs().price_per_session
    discount_pct = st.slider("Discount from historical 84 mg list price", 0, 100, 0, 1, format="%d%%")
    effective_price = reference_price * (1.0 - discount_pct / 100.0)
    ref_col, eff_col, disc_col = st.columns(3)
    ref_col.metric("Reference acquisition price", f"£{reference_price:,.2f} / session")
    eff_col.metric("Effective acquisition price", f"£{effective_price:,.2f} / session")
    disc_col.metric("List-price discount", f"{discount_pct}%")
    st.caption("The £489 reference is the model’s historical BNF price for an 84 mg session (TA854, April 2022), not a verified current or confidential net price. Drug acquisition and supervised administration are separate model inputs.")

    defaults = model.Inputs()
    relapse_presets = {
        "SUSTAIN-1 derived base case": (defaults.p_relapse_esk, defaults.p_relapse_comp),
        "Company recurrence scenario": (0.0243, 0.0356),
        "Higher esketamine relapse illustration": (0.100, defaults.p_relapse_comp),
    }
    relapse_sources = {
        "SUSTAIN-1 derived base case": "Derived from 26.7% versus 45.3% cumulative relapse over 17.7 weeks using a constant-hazard conversion; this is the Python default.",
        "Company recurrence scenario": "Company recurrence probabilities cited in the TA854 committee papers: 2.43% versus 3.56% per four weeks. This is an alternative source assumption, not the Python base case.",
        "Higher esketamine relapse illustration": "Raises the esketamine relapse probability to 0.10 as a stand-in for the selection-bias concern discussed in TA854. This is an illustrative scenario value, not an observed trial estimate.",
    }
    clinical_col, persistence_col = st.columns(2)
    with clinical_col:
        response_esk = st.slider(
            "Esketamine induction response probability",
            min_value=0.0,
            max_value=1.0,
            value=defaults.p_response_esk,
            step=0.001,
            help="Maps only to Inputs.p_response_esk. The base value, 0.693, is the TRANSFORM-2 day-28 response estimate.",
        )
        relapse_choice = st.selectbox("Relapse assumption set", list(relapse_presets), index=0)
        relapse_esk, relapse_comp = relapse_presets[relapse_choice]
        st.caption(relapse_sources[relapse_choice])
        st.caption(f"Current four-week relapse probabilities: esketamine {relapse_esk:.6f}; comparator {relapse_comp:.6f}.")
    with persistence_col:
        waning_choice = st.selectbox(
            "Post-treatment relapse-benefit waning",
            ["Base case · 15% lost per cycle", "Maintained benefit · 0% lost", "Faster waning · 50% lost"],
            index=0,
        )
        waning = {
            "Base case · 15% lost per cycle": defaults.waning_per_cycle,
            "Maintained benefit · 0% lost": 0.0,
            "Faster waning · 50% lost": 0.50,
        }[waning_choice]
        st.caption("Waning changes how quickly the lower esketamine relapse probability moves toward the comparator rate after treatment stops. The 15% base case is an illustrative modelling choice; 0% represents a maintained-effect scenario and 50% a faster-waning scenario.")

    delivery_col, admin_col = st.columns(2)
    with delivery_col:
        treatment_cycles = st.slider("Treatment cost duration (four-week cycles)", min_value=1, max_value=24, value=defaults.treatment_cycles, step=1)
        st.caption("Drug and administration costs stop after this many cycles, subject to the model’s treatment-eligibility rules.")
    with admin_col:
        admin = st.number_input("Administration cost per session (£)", min_value=0.0, max_value=500.0, value=defaults.admin_per_session, step=5.0)
        st.caption("Separate from acquisition price. £25 is the model’s 6:1 nurse-ratio illustration; £150 is the 1:1 scenario cited in the README.")

base_input = model.Inputs()
base_result = _cached_incremental(_input_cache_key(base_input))
current_input = _input_from_controls(
    horizon, effective_price, admin, response_esk, relapse_esk, relapse_comp, treatment_cycles, waning
)
current_result = _cached_incremental(_input_cache_key(current_input))
summary = economic_summary(current_result, wtp)
base_changed = any(asdict(current_input)[k] != asdict(base_input)[k] for k in asdict(base_input))
base_nmb = economic_summary(base_result, wtp)["nmb"]

with tab_overview:
    st.caption(f"Current run: {horizon}-year horizon · {len(current_result['esketamine']['trace'])} four-week cycles · {wtp_context} · £{wtp:,.0f}/QALY")
    st.subheader("Executive value summary")
    st.caption("Current controls · esketamine compared with placebo nasal spray, both with newly started oral antidepressant")
    c1, c2, c3 = st.columns(3)
    c1.metric("Esketamine total cost", f"£{current_result['esketamine']['cost']:,.0f}", help="Discounted total cost, NHS perspective")
    c2.metric("Comparator total cost", f"£{current_result['comparator']['cost']:,.0f}", help="Discounted total cost, NHS perspective")
    c3.metric("Incremental cost", f"£{summary['inc_cost']:,.0f}", help="Esketamine cost minus comparator cost")
    c4, c5, c6 = st.columns(3)
    c4.metric("Esketamine total QALYs", f"{current_result['esketamine']['qaly']:.3f}")
    c5.metric("Comparator total QALYs", f"{current_result['comparator']['qaly']:.3f}")
    c6.metric("Incremental QALYs", f"{summary['inc_qaly']:.3f}", help="Esketamine QALYs minus comparator QALYs")
    c7, c8 = st.columns(2)
    c7.metric("ICER", summary["icer_display"], help="Dominance and near-zero incremental effects are handled explicitly")
    c8.metric("Incremental NMB", f"£{summary['nmb']:,.0f}", help="WTP × incremental QALYs − incremental cost")
    st.info(summary["interpretation"])
    if base_changed:
        st.caption(f"Current selected scenario is compared with the live Python default when relevant. Default base-case NMB at the same threshold: £{base_nmb:,.0f}.")

    with st.expander("Arm-level results and cost breakdown", expanded=False):
        arm_table = pd.DataFrame([
            {"Strategy": "Esketamine + oral antidepressant", "Discounted cost (£)": current_result["esketamine"]["cost"], "Discounted QALYs": current_result["esketamine"]["qaly"], "Discounted life-years": current_result["esketamine"]["ly"], "Drug cost (£)": current_result["esketamine"]["drug_cost"], "Administration (£)": current_result["esketamine"]["admin_cost"], "Health-state (£)": current_result["esketamine"]["state_cost"], "Oral AD (£)": current_result["esketamine"]["oral_cost"]},
            {"Strategy": "Comparator + oral antidepressant", "Discounted cost (£)": current_result["comparator"]["cost"], "Discounted QALYs": current_result["comparator"]["qaly"], "Discounted life-years": current_result["comparator"]["ly"], "Drug cost (£)": current_result["comparator"]["drug_cost"], "Administration (£)": current_result["comparator"]["admin_cost"], "Health-state (£)": current_result["comparator"]["state_cost"], "Oral AD (£)": current_result["comparator"]["oral_cost"]},
        ])
        st.dataframe(arm_table.style.format({c: "£{:,.0f}" for c in arm_table.columns if "(£)" in c} | {"Discounted QALYs": "{:.3f}", "Discounted life-years": "{:.3f}"}), width="stretch", hide_index=True)

    st.subheader("Cohort health-state trace")
    st.caption("Cycle-start cohort proportions from the model trace. These are not the half-cycle occupancies used for QALYs. Lines distinguish strategy (solid = esketamine; dashed = comparator).")
    st.plotly_chart(make_trace_figure(current_result, horizon), width="stretch", config={"displayModeBar": False})

    st.subheader("Cost-effectiveness plane")
    st.caption("Deterministic model runs only. A single point is not a probability distribution or probabilistic sensitivity analysis.")
    plane_points = [{"label": f"Selected controls · {horizon} years", "inc_cost": summary["inc_cost"], "inc_qaly": summary["inc_qaly"], "kind": "selected"}]
    if base_changed:
        plane_points.insert(0, {"label": "Current Python default · 5 years", "inc_cost": base_result["inc_cost"], "inc_qaly": base_result["inc_qaly"], "kind": "base"})
    st.plotly_chart(make_ce_plane(plane_points, wtp), width="stretch", config={"displayModeBar": False})

with tab_price:
    st.subheader("Price-to-value analysis")
    st.markdown(f"Explore incremental net monetary benefit across £0–£{reference_price:,.0f} per session, holding the selected clinical and delivery assumptions constant. The model does not include a confidential patient access scheme.")
    sweep_prices = [REFERENCE_PRICE * i / 40 for i in range(41)]
    price_rows = []
    for price in sweep_prices:
        p_input = replace(current_input, price_per_session=price)
        p_result = _cached_incremental(_input_cache_key(p_input))
        p_summary = economic_summary(p_result, wtp)
        price_rows.append({"price": price, "nmb": p_summary["nmb"], "inc_cost": p_summary["inc_cost"], "inc_qaly": p_summary["inc_qaly"]})
    def nmb_at_price(price: float) -> float:
        price_input = replace(current_input, price_per_session=float(price))
        price_result = _cached_incremental(_input_cache_key(price_input))
        return economic_summary(price_result, wtp)["nmb"]

    be = find_nmb_zero_by_search(nmb_at_price, 0.0, REFERENCE_PRICE, scan_intervals=40)
    pcol1, pcol2, pcol3 = st.columns(3)
    pcol1.metric("Effective price per session", f"£{effective_price:,.2f}")
    if be["price"] is not None and be["in_range"]:
        be_discount = (1.0 - be["price"] / REFERENCE_PRICE) * 100.0
        pcol2.metric("Break-even acquisition price", f"£{be['price']:,.2f}", help="Price at which incremental NMB is zero, calculated from live model runs")
        pcol3.metric("Implied discount", f"{be_discount:.1f}%", help="Against the model’s £489 historical list-price assumption")
    else:
        pcol2.metric("Break-even acquisition price", "Not in £0–£489 range")
        pcol3.metric("Implied discount", "Not found in range")
    pdf = pd.DataFrame(price_rows)
    pfig = go.Figure()
    pfig.add_trace(go.Scatter(x=pdf["price"], y=pdf["nmb"], mode="lines", name="Incremental NMB", line={"color": "#244C66", "width": 3}, hovertemplate="Price £%{x:.0f}/session<br>NMB £%{y:,.0f}<extra></extra>"))
    pfig.add_trace(go.Scatter(x=[effective_price], y=[summary["nmb"]], mode="markers", name="Selected price", marker={"color": "#C47A31", "size": 11}, hovertemplate="Selected price £%{x:.2f}<br>NMB £%{y:,.0f}<extra></extra>"))
    if be["price"] is not None and be["in_range"]:
        pfig.add_vline(x=be["price"], line_dash="dot", line_color="#B42332", annotation_text="Break-even price")
    pfig.add_hline(y=0, line_dash="dash", line_color="#7F8B99", annotation_text="NMB = 0")
    pfig.update_layout(height=420, margin={"l": 20, "r": 20, "t": 20, "b": 35}, xaxis_title="Esketamine acquisition price (£ per session)", yaxis_title=f"Incremental NMB at £{wtp:,.0f}/QALY (£)", legend={"orientation": "h", "y": -0.2})
    st.plotly_chart(pfig, width="stretch", config={"displayModeBar": False})
    st.caption("Each point is a deterministic live model run with the current clinical, horizon, treatment-duration and administration assumptions. Price changes affect acquisition costs; the model’s clinical transitions and QALYs are unchanged.")

with tab_sensitivity:
    st.subheader("Deterministic scenario comparison")
    st.markdown("Scenarios show how alternative persistence and relapse assumptions affect incremental costs, QALYs, ICER, and NMB. They are deterministic model runs, not empirical confidence bounds.")
    scenario_inputs = [
        ("Maintained benefit", replace(current_input, waning_per_cycle=0.0)),
        ("Current controls", current_input),
        ("Faster waning", replace(current_input, waning_per_cycle=0.50)),
        ("Company recurrence rates", replace(current_input, p_relapse_esk=0.0243, p_relapse_comp=0.0356)),
        ("Higher esketamine relapse illustration", replace(current_input, p_relapse_esk=0.100)),
    ]
    scenario_results = [(name, inp, _cached_incremental(_input_cache_key(inp))) for name, inp in scenario_inputs]
    scen_df = pd.DataFrame([_run_summary_row(name, res, wtp) for name, _, res in scenario_results])
    st.dataframe(scen_df.style.format({"Incremental cost (£)": "£{:,.0f}", "Incremental QALYs": "{:.3f}", "NMB (£)": "£{:,.0f}"}), width="stretch", hide_index=True)
    st.caption("Maintained benefit sets waning to 0%; faster waning sets it to 50% per off-treatment cycle. Company recurrence uses the TA854-cited 2.43%/3.56% four-week rates. The 10% esketamine relapse value is an explicitly illustrative selection-bias scenario, not an observed estimate.")

    current_point_runs = [{"label": name, "inc_cost": res["inc_cost"], "inc_qaly": res["inc_qaly"], "kind": "selected" if name == "Current controls" else "scenario"} for name, _, res in scenario_results]
    st.markdown("**Cost-effectiveness plane by deterministic scenario**")
    st.plotly_chart(make_ce_plane(current_point_runs, wtp), width="stretch", config={"displayModeBar": False})

    st.subheader("Time-horizon sensitivity")
    st.caption("Each horizon is a separate live model run with the selected clinical, price, treatment-duration, and administration assumptions. The 1-, 5-, and 10-year runs contain 13, 65, and 130 four-week cycles, respectively.")
    horizon_results = []
    for years in (1, 5, 10):
        horizon_input = replace(current_input, horizon_years=float(years))
        horizon_result = _cached_incremental(_input_cache_key(horizon_input))
        horizon_summary = economic_summary(horizon_result, wtp)
        horizon_results.append({
            "Horizon (years)": years,
            "Cycles": len(horizon_result["esketamine"]["trace"]),
            "Incremental cost (£)": horizon_summary["inc_cost"],
            "Incremental QALYs": horizon_summary["inc_qaly"],
            "ICER (£/QALY)": horizon_summary["icer_display"],
            "NMB (£)": horizon_summary["nmb"],
        })
    horizon_df = pd.DataFrame(horizon_results)
    st.dataframe(horizon_df.style.format({"Incremental cost (£)": "£{:,.0f}", "Incremental QALYs": "{:.3f}", "NMB (£)": "£{:,.0f}"}), width="stretch", hide_index=True)
    horizon_fig = go.Figure(go.Scatter(
        x=horizon_df["Horizon (years)"],
        y=horizon_df["NMB (£)"],
        mode="lines+markers+text",
        text=[f"£{value:,.0f}" for value in horizon_df["NMB (£)" ]],
        textposition="top center",
        line={"color": "#244C66", "width": 2.5},
        marker={"color": "#C47A31", "size": 9},
        hovertemplate="Horizon %{x} years<br>Incremental NMB £%{y:,.0f}<extra></extra>",
        name="Incremental NMB",
    ))
    horizon_fig.add_hline(y=0, line_dash="dash", line_color="#7F8B99", annotation_text="NMB = 0")
    horizon_fig.update_layout(height=320, margin={"l": 20, "r": 20, "t": 25, "b": 35}, xaxis_title="Time horizon (years)", yaxis_title=f"Incremental NMB at £{wtp:,.0f}/QALY (£)", showlegend=False)
    horizon_fig.update_xaxes(tickmode="array", tickvals=[1, 5, 10])
    st.plotly_chart(horizon_fig, width="stretch", config={"displayModeBar": False})

    st.subheader("Acquisition-price and discount threshold")
    st.caption("The break-even price is found by scanning the supported £0–£489/session range and refining an observed NMB sign change with bisection. Each evaluation calls the model at that price; no price-cost linearity assumption is needed. NMB = 0 is the model-based threshold at the selected WTP and clinical assumptions.")
    price_col1, price_col2 = st.columns(2)
    price_col1.metric("Selected effective price", f"£{effective_price:,.2f} / session")
    if be["price"] is not None and be["in_range"]:
        break_even_discount = (1.0 - be["price"] / REFERENCE_PRICE) * 100.0
        price_col2.metric("Break-even price / discount", f"£{be['price']:,.2f} / session · {break_even_discount:.1f}% discount")
    else:
        price_col2.metric("Break-even price / discount", "No NMB zero within £0–£489/session")

    price_points = [("Free acquisition", 0.0), ("Selected price", effective_price), ("Historical list price", REFERENCE_PRICE)]
    if be["price"] is not None and be["in_range"]:
        price_points.append(("Break-even", float(be["price"])))
    threshold_rows = []
    seen_prices = set()
    for label, price in price_points:
        rounded_price = round(float(price), 2)
        if rounded_price in seen_prices:
            continue
        seen_prices.add(rounded_price)
        price_result = _cached_incremental(_input_cache_key(replace(current_input, price_per_session=rounded_price)))
        price_summary = economic_summary(price_result, wtp)
        threshold_rows.append({
            "Price point": label,
            "Acquisition price (£/session)": rounded_price,
            "Discount vs list price": (1.0 - rounded_price / REFERENCE_PRICE) * 100.0,
            "Incremental cost (£)": price_summary["inc_cost"],
            "Incremental QALYs": price_summary["inc_qaly"],
            "Incremental NMB (£)": price_summary["nmb"],
        })
    st.dataframe(pd.DataFrame(threshold_rows).style.format({"Acquisition price (£/session)": "£{:,.2f}", "Discount vs list price": "{:.1f}%", "Incremental cost (£)": "£{:,.0f}", "Incremental QALYs": "{:.3f}", "Incremental NMB (£)": "£{:,.0f}"}), width="stretch", hide_index=True)

    st.markdown("**One-way sensitivity analysis**")
    st.caption("Deterministic endpoint runs. Ranges cover 56–84 mg list-price equivalents (£326–£489/session), observed TRANSFORM-2 response probabilities (0.520–0.693), company-cited versus illustrative esketamine relapse rates (0.0243–0.100 per four weeks), 4–18 treatment cycles, 0–50% waning per off-treatment cycle, and £25–£150 administration per session. These are scenario bounds, not probability distributions; comparator relapse is held at the selected value when varying esketamine relapse.")
    spec = [
        ("Acquisition price / session (£)", "price_per_session", 326.0, 489.0),
        ("Esketamine induction response probability", "p_response_esk", 0.520, 0.693),
        ("Esketamine relapse probability / 4 weeks", "p_relapse_esk", 0.0243, 0.100),
        ("Treatment duration (cycles)", "treatment_cycles", 4, 18),
        ("Waning per cycle", "waning_per_cycle", 0.0, 0.50),
        ("Administration / session (£)", "admin_per_session", 25.0, 150.0),
    ]
    owsa = _cached_one_way(_input_cache_key(current_input), tuple(spec))
    spans = model.tornado_spans(owsa)
    # Plot model-generated endpoints directly; threshold reference uses the selected WTP.
    tornado = go.Figure()
    for row in spans.itertuples(index=False):
        lo, hi = sorted([row.icer_low, row.icer_high])
        tornado.add_trace(go.Scatter(x=[lo, hi], y=[row.parameter, row.parameter], mode="lines+markers", line={"width": 9, "color": "#54758A"}, marker={"size": 8}, name=row.parameter, showlegend=False, hovertemplate=f"{row.parameter}<br>ICER £%{{x:,.0f}}/QALY<extra></extra>"))
    base_icer = float(current_result["icer"])
    if math.isfinite(base_icer):
        tornado.add_vline(x=base_icer, line_color="#C47A31", line_dash="dash", annotation_text="Current controls")
    tornado.add_vline(x=wtp, line_color="#2A7F78", line_dash="dot", annotation_text=f"WTP £{wtp:,.0f}")
    tornado.update_layout(height=360, margin={"l": 20, "r": 20, "t": 35, "b": 35}, xaxis_title="Deterministic ICER (£ per QALY)", yaxis_title="Input varied", showlegend=False)
    st.plotly_chart(tornado, width="stretch", config={"displayModeBar": False})
    st.caption("The endpoint intervals can reverse direction when a higher input improves QALYs or reduces the ICER; the plotted endpoints come from live `one_way()` model runs.")
    finite_spans = spans[spans["span"].map(math.isfinite)]
    if not finite_spans.empty:
        leading_driver = finite_spans.iloc[0]
        st.info(f"Within these selected deterministic ranges, **{leading_driver['parameter']}** produces the widest ICER spread (about £{leading_driver['span']:,.0f}/QALY). This identifies a value-driver to scrutinise in evidence and price discussions; it does not assign a probability to any outcome.")

with tab_evidence:
    st.subheader("Model & evidence")
    st.markdown("**Illustrative model, not an official appraisal model.** This is a transparent decision-support demonstration informed by public sources documented in this repository. It is not the NICE TA854 model, the company submission model, the ERG model, or an endorsed NICE estimate. It does not estimate whether a payer will approve, fund, or cover esketamine.")

    st.markdown("### Decision problem and structure")
    st.markdown("The model compares **esketamine nasal spray plus a newly started oral antidepressant** with **placebo nasal spray plus a newly started oral antidepressant** for adults with treatment-resistant depression (TRD). The starting age is 46 years, close to the SUSTAIN-1 mean age documented in the README. Costs and health outcomes are assessed from an **NHS perspective**. The starting cohort is normalised to 1, so reported totals are per starting patient (equivalently, per one-person cohort).")
    st.markdown("The reported health states are `depressed`, `responding` (response and remission combined), and `dead` (absorbing). Five internal groups retain whether a responding patient was eligible for later esketamine treatment: induction failures are not made eligible by later response. This eligibility bookkeeping is part of the model engine; the dashboard uses its reported trace and does not recreate transitions.")
    st.markdown("Time is represented in **four-week cycles** (13 cycles per year). The Python base case is 5 years (65 cycles); the horizon control runs 1, 5, or 10 years (13, 65, or 130 cycles). Each cycle’s costs and outcomes are discounted at the midpoint of that cycle using 3.5% annually. The cohort trace plots cycle-start occupancy and therefore has no extra endpoint observation at the end of the horizon.")

    st.markdown("### Treatment, clinical evidence, and persistence")
    st.markdown("The model follows the documented dosing schedule of 8 sessions in cycle 1, 4 in cycle 2, then 2 sessions per four-week cycle. Drug acquisition and administration costs stop after the selected treatment duration (9 cycles by default, about 36 weeks); under the model’s stopping rule, induction non-responders stop after the first cycle. Relapse benefit wanes only after treatment stops: each off-treatment cycle loses the selected fraction of the remaining difference between esketamine and comparator relapse probabilities. The default 15% and the 0% maintained-benefit / 50% faster-waning options are scenario assumptions, not trial estimates.")
    st.markdown("**TRANSFORM-2** (Popova et al., *American Journal of Psychiatry*, 2019), as cited in the TA854 committee papers, informs the acute day-28 induction response probabilities (0.693 for esketamine and 0.520 for comparator). Its remission figures (0.525 vs 0.310) are used only to weight the model’s collapsed responding-state utility; remission is not a separate health state. **SUSTAIN-1** (Daly et al., *JAMA Psychiatry*, 2019) informs maintenance relapse: the repository documents 26.7% vs 45.3% cumulative relapse among stable remitters over median 17.7-week exposure. The base-case four-week probabilities are mathematically derived from those cumulative figures using a constant-hazard conversion. These trials do not directly supply every transition, cost, utility, or structural assumption in this model.")
    st.markdown("Later re-response is fixed at an illustrative 0.08 in both arms, informed by the range of later-step STAR*D response noted in the README, rather than copied from the company transition matrix. The company recurrence probabilities (2.43% and 3.56% per four weeks) are available as a separate scenario. No adverse-event states, safety-related cost consequences, or treatment mortality benefit are modelled; non-response stopping is a structural rule documented in the model, not a safety outcome.")

    st.markdown("### Costs, utilities, and outcome calculations")
    st.markdown("Health-state costs are £980 per four-week cycle while depressed and £164 while responding, documented in the TA854 committee papers from a UK chart review (n=295). Utilities are 0.417 for depressed and 0.841 for the combined responding state. The latter is a weighted simplification of the committee-paper response/remission utilities (0.764/0.866), using TRANSFORM-2 remission proportions. Esketamine acquisition uses the historical BNF April 2022 84 mg list-price assumption of £489 per session, excluding VAT and without the confidential patient access scheme. Administration is a separate £25 per supervised session illustration (the README contrasts this with £150 for a 1:1 nurse-ratio scenario). A £4 per-cycle generic oral antidepressant cost is an illustrative BNF order-of-magnitude assumption applied to both arms. Background mortality uses an approximate, constant annual qx of 0.0016 at age 46, informed by ONS England 2018–2020 life tables and weighted toward the SUSTAIN-1 sex mix; it is not copied from a single ONS cell.")
    st.markdown("For the Python default, utility and life-year accrual use half-cycle average occupancy. Health-state and oral antidepressant costs also use half-cycle average occupancy; acquisition and administration costs use the treatment-eligible cohort at cycle start and the session schedule. Discounted arm totals are accumulated across all model cycles. Incremental outcomes consistently mean **esketamine minus comparator**:")
    st.latex(r"\Delta C = C_{\mathrm{esketamine}} - C_{\mathrm{comparator}},\qquad \Delta Q = Q_{\mathrm{esketamine}} - Q_{\mathrm{comparator}}")
    st.latex(r"\mathrm{ICER} = \frac{\Delta C}{\Delta Q}\; (\pounds/\mathrm{QALY}),\qquad \mathrm{NMB}(\lambda) = \lambda\,\Delta Q - \Delta C\; (\pounds)")
    st.markdown("The ICER is interpreted with the signs of both incremental outcomes: an intervention with lower cost and higher QALYs dominates; higher cost and lower QALYs is dominated; and a ratio is not a sufficient decision summary when incremental QALYs are zero/negative or the comparison lies in a trade-off quadrant. The app suppresses misleading ratios for non-positive incremental QALYs and uses incremental NMB at the selected threshold (λ) for the threshold interpretation. Positive NMB favours esketamine under the selected assumptions and threshold; it is not a probability of cost-effectiveness or a reimbursement decision.")

    st.markdown("### Input status in this application")
    input_status = {
        "horizon_years": ("User-adjustable scenario", "1 / 5 / 10-year horizon; 5 years is the Python default."),
        "discount_rate": ("Fixed; source/reference-case based", "3.5% annual discount rate for costs and QALYs."),
        "start_age": ("Fixed; source-informed", "46 years; close to the SUSTAIN-1 mean age."),
        "annual_qx": ("Fixed; derived approximation", "ONS England 2018–2020 life tables; age-46 sex-weighted approximation."),
        "p_response_esk": ("User-adjustable; source-derived default", "TRANSFORM-2 day-28 response; adjustable in Price & Access."),
        "p_response_comp": ("Fixed; source-derived", "TRANSFORM-2 day-28 comparator response."),
        "p_reresponse": ("Fixed; illustrative", "0.08 in both arms; later-step response context from STAR*D, not a copied transition."),
        "p_relapse_esk": ("User-adjustable scenario; derived default", "SUSTAIN-1 cumulative relapse converted to a four-week probability; scenario presets also available."),
        "p_relapse_comp": ("User-adjustable scenario; derived default", "SUSTAIN-1 cumulative relapse converted to a four-week probability; scenario presets also available."),
        "treatment_cycles": ("User-adjustable scenario", "Drug and administration cost duration; default 9 four-week cycles."),
        "waning_per_cycle": ("User-adjustable scenario", "Illustrative default 15%; maintained and faster-waning presets are also available."),
        "u_depressed": ("Fixed; source-derived", "TA854 committee-paper mapped utility, 0.417."),
        "u_responding": ("Fixed; source-derived and simplified", "0.841 weighted utility collapsing response and remission."),
        "c_depressed": ("Fixed; source-derived", "TA854 committee-paper health-state cost per four-week cycle."),
        "c_responding": ("Fixed; source-derived", "TA854 committee-paper health-state cost per four-week cycle."),
        "price_per_session": ("User-adjustable", "Historical £489 84 mg list-price reference; user sets a discount."),
        "admin_per_session": ("User-adjustable scenario", "Illustrative £25 default; separate from drug acquisition cost."),
        "oral_ad_per_cycle": ("Fixed; illustrative", "Generic sertraline BNF order-of-magnitude cost, applied to both arms."),
        "stop_if_no_response": ("Fixed structural rule", "Induction non-responders stop after cycle 1, following the documented benefit assessment rule."),
    }
    default_values = asdict(base_input)
    current_values = asdict(current_input)
    input_rows = [
        {
            "Model input": name,
            "Python default": default_values[name],
            "Current run": current_values[name],
            "Status": input_status[name][0],
            "Source / role": input_status[name][1],
        }
        for name in default_values
    ]
    st.dataframe(pd.DataFrame(input_rows), width="stretch", hide_index=True)
    st.caption("Source notes below reproduce the repository’s input log. ‘Source-derived’ includes mathematical conversions and mapped or weighted values; it does not mean that the exact model parameter was directly reported by a trial.")
    st.dataframe(pd.DataFrame(model.INPUT_LOG), width="stretch", hide_index=True)

    st.subheader("Model checks")
    diagnostics = model.checks(current_result)
    failures, information = split_diagnostics(diagnostics)
    if failures:
        st.error("Model diagnostics reported failures. Review before interpreting results.")
    else:
        st.success("No `FAIL:` diagnostics in this run.")
    for line in information:
        st.write("• " + line)
    for line in failures:
        st.write("❌ " + line)

    st.subheader("Workbook reconciliation is a separate configuration")
    st.markdown("The workbook uses cycle-start health-state/oral costs, excludes the post-transition state in final-cycle QALYs, starts waning one transition earlier, and stores relapse probabilities rounded to six decimals. These settings are not the dashboard default.")
    st.code("model.run_model(\n    model.Inputs(p_relapse_esk=0.067787, p_relapse_comp=0.127454),\n    drop_last_half_cycle=True,\n    waning_start_shift=-1,\n    half_cycle_costs=False,\n)")
    st.caption("This reconciliation setting reproduces the workbook’s documented cached summary; it does not make the workbook the live result source.")

    st.subheader("Decision threshold context")
    st.markdown("The historical TA854 appraisal used the £20,000–£30,000/QALY range. For relevant technology appraisals from April 2026, NICE’s current standard range is £25,000–£35,000/QALY. See the [NICE economic evaluation guidance](https://www.nice.org.uk/process/pmg36/chapter/economic-evaluation-2/) and [NICE 2025–26 annual report](https://indepth.nice.org.uk/nice-annual-report-and-accounts-2025-26/index.html). This current range is separate policy context and is not retroactively applied to TA854.")

    st.subheader("What the model does not represent")
    st.markdown("This is a simplified cohort model, not a reproduction of the appraisal submissions. It combines response and remission; does not model subsequent treatment lines, carer quality of life, adverse-event states or costs, or a mortality effect of treatment; uses a single historical 84 mg list price with no confidential PAS; and relies on several illustrative or converted assumptions. In particular, maintenance relapse evidence is translated to a constant four-week hazard and the treatment duration and post-treatment waning settings are scenarios. The Excel workbook has different documented cost, terminal-cycle, waning-timing, and probability-rounding conventions; its reconciliation configuration shown above is separate from the live Python base case.")
