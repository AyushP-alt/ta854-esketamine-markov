"""Numerical reconciliation of the Python engine to the companion workbook.

This is a separate configuration from the default Python base case. Reference
values below are the cached Results-sheet cells in the checked-in workbook.
"""

import unittest

import model


# The workbook stores relapse probabilities to six decimals, unlike the full
# precision values derived by the default Python implementation.
WORKBOOK_INPUTS = model.Inputs(
    p_relapse_esk=0.067787,
    p_relapse_comp=0.127454,
)

# Explicit tolerances allow for spreadsheet serialization/display precision.
COST_TOLERANCE = 0.01  # pounds
QALY_TOLERANCE = 1e-9  # QALYs
ICER_TOLERANCE = 0.01  # pounds per QALY


class ExcelReconciliationTests(unittest.TestCase):
    def test_workbook_style_run_matches_cached_results_sheet(self):
        run = model.run_model(
            WORKBOOK_INPUTS,
            drop_last_half_cycle=True,
            waning_start_shift=-1,
            half_cycle_costs=False,
        )
        result = run.result
        esketamine = result["esketamine"]
        comparator = result["comparator"]

        # Workbook Results!B4 and B5: discounted strategy totals.
        self.assertAlmostEqual(esketamine["cost"], 46944.209483434, delta=COST_TOLERANCE)
        self.assertAlmostEqual(comparator["cost"], 39611.0285841719, delta=COST_TOLERANCE)
        self.assertAlmostEqual(esketamine["qaly"], 2.73635664874335, delta=QALY_TOLERANCE)
        self.assertAlmostEqual(comparator["qaly"], 2.65217055671909, delta=QALY_TOLERANCE)

        # Workbook Results!B7:B9: esketamine minus comparator, then cost/QALY.
        self.assertAlmostEqual(run.inc_cost, 7333.18089926214, delta=COST_TOLERANCE)
        self.assertAlmostEqual(run.inc_qaly, 0.0841860920242632, delta=QALY_TOLERANCE)
        self.assertAlmostEqual(run.icer, 87106.7978443357, delta=ICER_TOLERANCE)

        # Confirm this was the intended reconciliation configuration.
        self.assertTrue(result["drop_last_half_cycle"])
        self.assertEqual(result["waning_start_shift"], -1)
        self.assertFalse(result["half_cycle_costs"])

    def test_default_python_configuration_remains_separate(self):
        default = model.run_model(model.Inputs()).result
        self.assertFalse(default["drop_last_half_cycle"])
        self.assertEqual(default["waning_start_shift"], 0)
        self.assertTrue(default["half_cycle_costs"])
        self.assertNotEqual(default["inputs"].p_relapse_esk, WORKBOOK_INPUTS.p_relapse_esk)
        self.assertNotEqual(default["inputs"].p_relapse_comp, WORKBOOK_INPUTS.p_relapse_comp)


if __name__ == "__main__":
    unittest.main()
