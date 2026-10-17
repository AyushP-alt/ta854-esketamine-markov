"""Regression coverage for pure dashboard interpretation helpers."""

import unittest
from dataclasses import replace
import math

import model
from dashboard_logic import (
    break_even_price,
    economic_summary,
    find_nmb_zero_by_search,
    format_icer,
    split_diagnostics,
    validate_probability,
)


class DashboardLogicTests(unittest.TestCase):
    def test_default_python_outputs_match_captured_baseline(self):
        result = model.incremental(model.Inputs())
        expected = {
            "esketamine": {"cost": 46684.04017486803, "qaly": 2.7589117096419336},
            "comparator": {"cost": 39450.7983984164, "qaly": 2.670842603705382},
            "inc_cost": 7233.2417764516285,
            "inc_qaly": 0.08806910593655148,
            "icer": 82131.43189693269,
        }
        for arm in ("esketamine", "comparator"):
            self.assertAlmostEqual(result[arm]["cost"], expected[arm]["cost"], delta=1e-9)
            self.assertAlmostEqual(result[arm]["qaly"], expected[arm]["qaly"], delta=1e-12)
            self.assertEqual(len(result[arm]["trace"]), 65)
        for key in ("inc_cost", "inc_qaly", "icer"):
            self.assertAlmostEqual(result[key], expected[key], delta=1e-9)

    def test_default_summary_uses_current_model_outputs(self):
        result = model.incremental(model.Inputs())
        summary = economic_summary(result, 25_000)
        self.assertAlmostEqual(summary["inc_cost"], result["inc_cost"], places=12)
        self.assertAlmostEqual(summary["inc_qaly"], result["inc_qaly"], places=12)
        self.assertAlmostEqual(summary["icer"], result["icer"], places=12)
        self.assertAlmostEqual(
            summary["nmb"], 25_000 * result["inc_qaly"] - result["inc_cost"], places=10
        )

    def test_price_changes_cost_but_not_qalys(self):
        base = model.incremental(model.Inputs())
        higher_price = model.incremental(replace(model.Inputs(), price_per_session=600.0))
        self.assertGreater(higher_price["inc_cost"], base["inc_cost"])
        self.assertAlmostEqual(higher_price["inc_qaly"], base["inc_qaly"], places=12)
        self.assertGreater(higher_price["esketamine"]["drug_cost"], base["esketamine"]["drug_cost"])
        self.assertEqual(higher_price["comparator"]["drug_cost"], 0.0)
        self.assertAlmostEqual(higher_price["esketamine"]["admin_cost"], base["esketamine"]["admin_cost"], places=12)
        self.assertNotAlmostEqual(higher_price["icer"], base["icer"])
        self.assertNotAlmostEqual(
            economic_summary(higher_price, 25_000)["nmb"],
            economic_summary(base, 25_000)["nmb"],
        )

    def test_wtp_changes_nmb_without_mutating_model_results(self):
        result = model.incremental(model.Inputs())
        result_before = (result["inc_cost"], result["inc_qaly"], result["icer"])
        low = economic_summary(result, 20_000)
        high = economic_summary(result, 35_000)
        self.assertNotEqual(low["nmb"], high["nmb"])
        self.assertEqual(result_before, (result["inc_cost"], result["inc_qaly"], result["icer"]))

    def test_horizon_maps_to_expected_cycle_count(self):
        for years, cycles in ((1, 13), (5, 65), (10, 130)):
            result = model.incremental(model.Inputs(horizon_years=years))
            self.assertEqual(len(result["esketamine"]["trace"]), cycles)
            self.assertEqual(len(result["comparator"]["trace"]), cycles)

    def test_waning_scenarios_use_supported_input(self):
        base = model.Inputs()
        maintained = model.incremental(replace(base, waning_per_cycle=0.0))
        faster = model.incremental(replace(base, waning_per_cycle=0.50))
        self.assertEqual(maintained["inputs"].waning_per_cycle, 0.0)
        self.assertEqual(faster["inputs"].waning_per_cycle, 0.50)
        self.assertNotAlmostEqual(maintained["inc_qaly"], faster["inc_qaly"])

    def test_induction_response_scenario_changes_outcomes_and_cohort_trace(self):
        base = model.incremental(model.Inputs())
        changed = model.incremental(replace(model.Inputs(), p_response_esk=0.80))
        self.assertNotAlmostEqual(changed["esketamine"]["qaly"], base["esketamine"]["qaly"])
        self.assertNotAlmostEqual(
            changed["esketamine"]["trace"].loc[1, "responding"],
            base["esketamine"]["trace"].loc[1, "responding"],
        )
        self.assertNotAlmostEqual(
            changed["esketamine"]["trace"].loc[5, "depressed"],
            base["esketamine"]["trace"].loc[5, "depressed"],
        )

    def test_waning_scenario_changes_cohort_trace_after_treatment_stops(self):
        base = model.incremental(model.Inputs())
        maintained = model.incremental(replace(model.Inputs(), waning_per_cycle=0.0))
        self.assertNotAlmostEqual(
            maintained["esketamine"]["trace"].loc[20, "responding"],
            base["esketamine"]["trace"].loc[20, "responding"],
        )

    def test_diagnostic_prefix_separates_failures_from_notes(self):
        failures, notes = split_diagnostics(["check passed", "FAIL: state sum", "warning note"])
        self.assertEqual(failures, ["FAIL: state sum"])
        self.assertEqual(notes, ["check passed", "warning note"])

    def test_icer_interpretation_handles_zero_and_dominance(self):
        self.assertIn("Not meaningful", format_icer(10.0, 0.0))
        self.assertIn("Dominant", format_icer(-10.0, 0.1))
        self.assertIn("Dominated", format_icer(10.0, -0.1))
        self.assertIn("use NMB", format_icer(-10.0, -0.1, 100.0))
        self.assertIn("use NMB", format_icer(0.0, -0.1, 0.0))

    def test_equal_cost_and_zero_effect_interpretation_is_not_mislabeled(self):
        zero_qaly = economic_summary({"inc_cost": -5.0, "inc_qaly": 0.0, "icer": float("inf")}, 25_000)
        self.assertIn("approximately zero", zero_qaly["icer_display"])
        self.assertIn("Costs are lower", zero_qaly["interpretation"])
        negative_qaly = economic_summary({"inc_cost": 0.0, "inc_qaly": -0.1, "icer": 0.0}, 25_000)
        self.assertIn("use NMB", negative_qaly["icer_display"])
        self.assertIn("same cost", negative_qaly["interpretation"])

        near_zero_qaly = economic_summary(
            {"inc_cost": 10.0, "inc_qaly": 1e-13, "icer": float("inf")}, 25_000
        )
        self.assertIn("Not meaningful", near_zero_qaly["icer_display"])
        self.assertIn("approximately no incremental QALY", near_zero_qaly["interpretation"])

    def test_break_even_price_uses_engine_results(self):
        zero = model.incremental(replace(model.Inputs(), price_per_session=0.0))
        one = model.incremental(replace(model.Inputs(), price_per_session=1.0))
        out = break_even_price(zero, one, 25_000, 0.0, 489.0)
        self.assertIsNotNone(out["price"])
        self.assertTrue(out["in_range"])

    def test_nmb_threshold_search_refines_nonlinear_root(self):
        root = math.sqrt(37.0)
        out = find_nmb_zero_by_search(lambda price: 37.0 - price**2, 0.0, 20.0)
        self.assertAlmostEqual(out["price"], root, places=2)
        self.assertLessEqual(abs(out["nmb"]), 0.1)

    def test_nmb_threshold_search_reports_no_crossing(self):
        out = find_nmb_zero_by_search(lambda price: 100.0 + price, 0.0, 20.0)
        self.assertIsNone(out["price"])
        self.assertFalse(out["in_range"])

    def test_nmb_threshold_search_accepts_endpoint_root(self):
        out = find_nmb_zero_by_search(lambda price: price - 20.0, 0.0, 20.0)
        self.assertEqual(out["price"], 20.0)
        self.assertEqual(out["nmb"], 0.0)

    def test_nmb_threshold_search_finds_first_of_multiple_crossings(self):
        out = find_nmb_zero_by_search(lambda price: (price - 2.0) * (price - 8.0), 0.0, 10.0, scan_intervals=10)
        self.assertEqual(out["price"], 2.0)

    def test_nmb_threshold_search_rejects_invalid_or_nonfinite_evaluations(self):
        with self.assertRaises(ValueError):
            find_nmb_zero_by_search(lambda price: price, 10.0, 10.0)
        with self.assertRaises(ValueError):
            find_nmb_zero_by_search(lambda price: price, 0.0, 10.0, scan_intervals=0)
        with self.assertRaises(ValueError):
            find_nmb_zero_by_search(lambda price: float("nan"), 0.0, 10.0)

    def test_invalid_probability_is_rejected(self):
        self.assertEqual(validate_probability(0.0, "response"), 0.0)
        self.assertEqual(validate_probability(0.693, "response"), 0.693)
        self.assertEqual(validate_probability(1.0, "response"), 1.0)
        with self.assertRaises(ValueError):
            validate_probability(1.01, "response")
        for invalid in (-0.01, float("nan"), float("inf"), float("-inf")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_probability(invalid, "response")


if __name__ == "__main__":
    unittest.main()
