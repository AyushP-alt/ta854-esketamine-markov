"""Streamlit AppTest coverage for live dashboard calculations and charts."""

from dataclasses import replace
import unittest

import plotly.io as pio
from streamlit.testing.v1 import AppTest

import model
from dashboard_logic import economic_summary


APP_FILE = "app.py"


def _widget(app, kind, label):
    return next(element for element in app.get(kind) if element.label == label)


def _metrics(app):
    return {metric.label: metric.value for metric in app.metric}


def _summary_info(app):
    return next(element.value for element in app.info if "Incremental NMB is" in element.value)


def _chart_figure(app, index):
    return pio.from_json(app.get("plotly_chart")[index].proto.spec)


class StreamlitIntegrationTests(unittest.TestCase):
    def start_app(self):
        app = AppTest.from_file(APP_FILE).run(timeout=60)
        self.assertEqual(len(app.exception), 0, [str(e.value) for e in app.exception])
        return app

    def test_base_case_metrics_are_current_model_outputs(self):
        app = self.start_app()
        metrics = _metrics(app)
        result = model.incremental(model.Inputs())
        summary = economic_summary(result, 25_000)

        self.assertEqual(metrics["Incremental cost"], f"£{summary['inc_cost']:,.0f}")
        self.assertEqual(metrics["Incremental QALYs"], f"{summary['inc_qaly']:.3f}")
        self.assertEqual(metrics["ICER"], summary["icer_display"])
        self.assertEqual(metrics["Incremental NMB"], f"£{summary['nmb']:,.0f}")
        self.assertEqual(
            [tab.label for tab in app.tabs],
            ["Value overview", "Sensitivity & price", "Assumptions & evidence"],
        )
        self.assertAlmostEqual(result["icer"], 82_131, delta=1)

    def test_price_control_updates_costs_icer_and_nmb_without_qaly_change(self):
        app = self.start_app()
        base_metrics = _metrics(app)
        _widget(app, "slider", "Discount from historical 84 mg list price").set_value(20).run(timeout=60)
        self.assertEqual(len(app.exception), 0, [str(e.value) for e in app.exception])

        changed_input = replace(model.Inputs(), price_per_session=489.0 * 0.8)
        result = model.incremental(changed_input)
        summary = economic_summary(result, 25_000)
        metrics = _metrics(app)
        self.assertEqual(metrics["Incremental cost"], f"£{summary['inc_cost']:,.0f}")
        self.assertEqual(metrics["ICER"], summary["icer_display"])
        self.assertEqual(metrics["Incremental NMB"], f"£{summary['nmb']:,.0f}")
        self.assertEqual(metrics["Incremental QALYs"], base_metrics["Incremental QALYs"])

    def test_wtp_control_updates_nmb_and_decision_without_changing_model_outcomes(self):
        app = self.start_app()
        _widget(app, "slider", "Discount from historical 84 mg list price").set_value(50).run(timeout=60)
        _widget(app, "selectbox", "WTP policy context").set_value(
            "Current standard NICE context · £25,000–£35,000/QALY"
        ).run(timeout=60)

        at_30k = _metrics(app)
        interpretation_30k = _summary_info(app)
        stable_values = tuple(at_30k[key] for key in ("Incremental cost", "Incremental QALYs", "ICER"))

        _widget(app, "slider", "WTP threshold (£/QALY)").set_value(35_000).run(timeout=60)
        at_35k = _metrics(app)
        interpretation_35k = _summary_info(app)
        self.assertNotEqual(at_30k["Incremental NMB"], at_35k["Incremental NMB"])
        self.assertIn("Incremental NMB is negative", interpretation_30k)
        self.assertIn("Incremental NMB is positive", interpretation_35k)
        self.assertNotEqual(interpretation_30k, interpretation_35k)
        self.assertEqual(stable_values, tuple(at_35k[key] for key in ("Incremental cost", "Incremental QALYs", "ICER")))

    def test_clinical_and_waning_controls_change_model_and_trace(self):
        app = self.start_app()
        base_metrics = _metrics(app)
        base_trace = _chart_figure(app, 0)
        base_responding_series = base_trace.data[1].y["bdata"]

        _widget(app, "slider", "Esketamine induction response probability").set_value(0.80).run(timeout=60)
        response_metrics = _metrics(app)
        response_trace = _chart_figure(app, 0)
        self.assertNotEqual(response_metrics["Incremental QALYs"], base_metrics["Incremental QALYs"])
        self.assertNotEqual(response_trace.data[1].y["bdata"], base_responding_series)

        _widget(app, "selectbox", "Post-treatment relapse-benefit waning").set_value(
            "Maintained benefit · 0% lost"
        ).run(timeout=60)
        waning_metrics = _metrics(app)
        waning_trace = _chart_figure(app, 0)
        self.assertNotEqual(waning_metrics["Incremental QALYs"], response_metrics["Incremental QALYs"])
        self.assertNotEqual(waning_trace.data[1].y["bdata"], response_trace.data[1].y["bdata"])

    def test_horizon_recalculates_and_ce_plane_uses_selected_wtp_and_effects(self):
        app = self.start_app()
        _widget(app, "selectbox", "Time horizon").set_value(1).run(timeout=60)
        result = model.incremental(model.Inputs(horizon_years=1))
        metrics = _metrics(app)
        self.assertEqual(metrics["Incremental QALYs"], f"{result['inc_qaly']:.3f}")
        self.assertEqual(len(result["esketamine"]["trace"]), 13)
        self.assertTrue(any("Current run · 1 year / 13 four-week cycles" in str(c.value) for c in app.caption))
        trace_chart = _chart_figure(app, 0)
        self.assertEqual(tuple(trace_chart.layout.xaxis3.range), (0, 1))

        _widget(app, "slider", "WTP threshold (£/QALY)").set_value(30_000).run(timeout=60)
        plane = _chart_figure(app, 1)
        wtp_line = plane.data[0]
        self.assertIn("£30,000/QALY", wtp_line.name)
        for x, y in zip(wtp_line.x, wtp_line.y):
            self.assertAlmostEqual(y, 30_000 * x, delta=1e-8)
        current_point = plane.data[1]
        self.assertAlmostEqual(current_point.x[-1], result["inc_qaly"], places=12)
        self.assertAlmostEqual(current_point.y[-1], result["inc_cost"], places=8)

    def test_reset_action_restores_base_case_controls_and_results(self):
        app = self.start_app()
        _widget(app, "selectbox", "Time horizon").set_value(1).run(timeout=60)
        _widget(app, "slider", "Discount from historical 84 mg list price").set_value(50).run(timeout=60)
        _widget(app, "slider", "Esketamine induction response probability").set_value(0.8).run(timeout=60)

        _widget(app, "button", "Reset to base case").click().run(timeout=60)

        self.assertEqual(_widget(app, "selectbox", "Time horizon").value, 5)
        self.assertEqual(_widget(app, "slider", "Discount from historical 84 mg list price").value, 0)
        self.assertEqual(_widget(app, "slider", "Esketamine induction response probability").value, model.Inputs().p_response_esk)
        self.assertEqual(_metrics(app)["ICER"], economic_summary(model.incremental(model.Inputs()), 25_000)["icer_display"])


if __name__ == "__main__":
    unittest.main()
