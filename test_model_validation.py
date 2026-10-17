"""Focused validation tests for long-term model behaviour.

These tests capture the current deterministic engine, not a clinical validation
or a reproduction of the NICE/company model. Numerical grid values are
regression checks; explicitly named ordering and no-effect checks are
mathematical invariants of the current implementation.
"""

from dataclasses import replace
import unittest

import model


class WaningHorizonValidationTests(unittest.TestCase):
    # (incremental cost, incremental QALYs, ICER), captured from the live
    # Python engine for the listed inputs. Costs and QALYs use the model's
    # 3.5% mid-cycle discounting.
    GRID = {
        0.00: {
            5: (3501.77904054811, 0.237215052394231, 14762.0439984915),
            10: (-872.685806957263, 0.412061686570526, -2117.85233958629),
            20: (-7578.30138262695, 0.680084028584439, -11143.1838774406),
        },
        0.05: {
            5: (6355.42118192672, 0.123155449458286, 51604.8718094229),
            10: (6287.78310667790, 0.125858932103976, 49958.9739207652),
            20: (6285.74074408684, 0.125940564997864, 49910.3743435919),
        },
        0.15: {
            5: (7233.24177645163, 0.0880691059365515, 82131.4318969327),
            10: (7233.13240589407, 0.0880734774595835, 82126.1134966916),
            20: (7233.13240354718, 0.0880734775533671, 82126.1133825940),
        },
        0.50: {
            5: (7624.23329686547, 0.0724412397359981, 105247.139953028),
            10: (7624.23200818038, 0.0724412912445240, 105247.047328918),
            20: (7624.23200818003, 0.0724412912445302, 105247.047328904),
        },
        1.00: {
            5: (7722.78138699373, 0.0685022889239333, 112737.566996766),
            10: (7722.78037523350, 0.0685023293638150, 112737.485673194),
            20: (7722.78037523321, 0.0685023293638167, 112737.485673187),
        },
    }

    def test_waning_horizon_regression_grid_and_cohort_traces(self):
        for waning, by_horizon in self.GRID.items():
            for years, expected in by_horizon.items():
                with self.subTest(waning=waning, years=years):
                    result = model.incremental(
                        replace(
                            model.Inputs(),
                            horizon_years=years,
                            waning_per_cycle=waning,
                        )
                    )
                    cycles = years * model.CYCLES_PER_YEAR
                    for arm_name in ("esketamine", "comparator"):
                        trace = result[arm_name]["trace"]
                        self.assertEqual(len(trace), cycles)
                        self.assertEqual(int(trace.iloc[-1]["cycle"]), cycles - 1)
                        self.assertAlmostEqual(
                            trace.loc[:, ["depressed", "responding", "dead"]]
                            .sum(axis=1)
                            .min(),
                            1.0,
                            places=10,
                            msg=f"{arm_name} cycle-start states must sum to one",
                        )
                    for key, value in zip(("inc_cost", "inc_qaly", "icer"), expected):
                        self.assertAlmostEqual(
                            result[key],
                            value,
                            delta=1e-7 if key != "inc_qaly" else 1e-11,
                            msg=f"Regression changed for waning={waning}, {years}y, {key}",
                        )

    def test_five_year_waning_ordering_follows_effect_definition(self):
        # waning_per_cycle is the fraction of the remaining incremental
        # relapse benefit lost per off-treatment cycle: larger means faster
        # loss, not stronger treatment effect.
        results = {
            w: model.incremental(
                replace(model.Inputs(), horizon_years=5, waning_per_cycle=w)
            )
            for w in (0.00, 0.05, 0.15, 0.50, 1.00)
        }
        waning_values = tuple(results)
        for lower, higher in zip(waning_values, waning_values[1:]):
            self.assertGreater(
                results[lower]["inc_qaly"],
                results[higher]["inc_qaly"],
                msg=(
                    "At five years, faster loss of the esketamine relapse advantage "
                    "should reduce incremental QALYs under the current transition rule"
                ),
            )
            self.assertLess(
                results[lower]["inc_cost"],
                results[higher]["inc_cost"],
                msg=(
                    "At five years, faster loss of the relapse advantage should "
                    "remove health-state cost offsets; drug schedule is unchanged"
                ),
            )
        # The sign change at maintained effect makes a blanket monotonic-ICER
        # rule across horizons inappropriate; no such invariant is asserted.

    def test_long_run_prefix_matches_shorter_horizon_runs(self):
        longest = model.incremental(model.Inputs(horizon_years=20))
        for years in (1, 3, 5, 10, 15):
            cycles = years * model.CYCLES_PER_YEAR
            shorter = model.incremental(model.Inputs(horizon_years=years))
            with self.subTest(years=years):
                for arm_name in ("esketamine", "comparator"):
                    short_arm = shorter[arm_name]
                    long_arm = longest[arm_name]
                    for outcome in ("cost", "qaly"):
                        self.assertAlmostEqual(
                            short_arm[outcome],
                            long_arm["trace"].iloc[:cycles][f"disc_{outcome}"].sum(),
                            delta=1e-9,
                            msg=f"{arm_name} {outcome} must equal the {years}-year prefix",
                        )
                    prefix_ly = sum(
                        (
                            long_arm["trace"].iloc[cycle]["occ_depressed"]
                            + long_arm["trace"].iloc[cycle]["occ_responding"]
                        )
                        * model.CYCLE_YEARS
                        * model.discount_factor(cycle, model.Inputs().discount_rate)
                        for cycle in range(cycles)
                    )
                    self.assertAlmostEqual(
                        short_arm["ly"], prefix_ly, delta=1e-9,
                        msg=f"{arm_name} life-years must equal the {years}-year prefix",
                    )
                    for state in ("depressed", "responding", "dead"):
                        self.assertTrue(
                            short_arm["trace"][state].equals(
                                long_arm["trace"].iloc[:cycles][state].reset_index(drop=True)
                            ),
                            msg=f"{arm_name} {state} trace changed before year {years}",
                        )
                self.assertEqual(len(shorter["esketamine"]["trace"]), cycles)
                self.assertEqual(len(longest["esketamine"]["trace"]), 260)

    def test_relapse_reresponse_and_waning_inputs_reach_long_horizon_results(self):
        base_inputs = replace(model.Inputs(), horizon_years=20)
        base = model.incremental(base_inputs)
        scenarios = {
            "eligible relapse probability": replace(
                base_inputs, p_relapse_esk=base_inputs.p_relapse_esk + 0.01
            ),
            "shared re-response probability": replace(
                base_inputs, p_reresponse=base_inputs.p_reresponse + 0.02
            ),
            "post-treatment waning": replace(base_inputs, waning_per_cycle=0.50),
        }
        for label, inputs in scenarios.items():
            with self.subTest(parameter=label):
                changed = model.incremental(inputs)
                self.assertNotAlmostEqual(
                    changed["inc_qaly"], base["inc_qaly"], delta=1e-10,
                    msg=f"Changing {label} did not affect long-run incremental QALYs",
                )
                self.assertNotAlmostEqual(
                    changed["esketamine"]["trace"].loc[20, "responding"],
                    base["esketamine"]["trace"].loc[20, "responding"],
                    delta=1e-10,
                    msg=(
                        f"Changing {label} did not affect the post-treatment cohort trace; "
                        "cycle 20 is used because positive waning can make late endpoint "
                        "occupancy converge even when cumulative outcomes differ"
                    ),
                )


class NoEffectValidationTests(unittest.TestCase):
    def test_cost_only_increase_raises_icer_when_incremental_qalys_are_positive(self):
        base = model.incremental(model.Inputs())
        higher_price = model.incremental(
            replace(model.Inputs(), price_per_session=model.Inputs().price_per_session + 100)
        )
        self.assertGreater(base["inc_qaly"], 0.0)
        self.assertAlmostEqual(
            higher_price["inc_qaly"], base["inc_qaly"], delta=1e-12,
            msg="Changing acquisition price alone must not change incremental QALYs",
        )
        self.assertGreater(
            higher_price["inc_cost"], base["inc_cost"],
            msg="Higher acquisition price must increase incremental costs",
        )
        self.assertGreater(
            higher_price["icer"], base["icer"],
            msg="With fixed positive incremental QALYs, higher incremental cost must raise the ICER",
        )

    def test_equal_clinical_assumptions_leave_only_intervention_cost_difference(self):
        base = model.Inputs()
        no_effect_inputs = replace(
            base,
            p_response_esk=base.p_response_comp,
            p_relapse_esk=base.p_relapse_comp,
        )
        result = model.incremental(no_effect_inputs)
        esk = result["esketamine"]
        comp = result["comparator"]

        self.assertAlmostEqual(
            result["inc_qaly"], 0.0, delta=1e-12,
            msg="No-effect invariant violated: equal arm outcomes must yield zero incremental QALYs",
        )
        self.assertAlmostEqual(
            esk["ly"] - comp["ly"], 0.0, delta=1e-12,
            msg="No-effect invariant violated: equal arm outcomes must yield zero incremental life-years",
        )
        for outcome in ("qaly", "ly", "state_cost", "oral_cost"):
            self.assertAlmostEqual(
                esk[outcome], comp[outcome], delta=1e-12,
                msg=f"No-effect invariant violated: arm {outcome} should be equal",
            )

        # The current engine has esketamine acquisition and administration
        # costs only in the intervention arm. The comparator has no drug or
        # administration line. This identity also accounts for any future
        # arm-specific costs if they are added to the corresponding totals.
        expected_incremental_cost = (
            esk["drug_cost"] + esk["admin_cost"]
            - comp["drug_cost"] - comp["admin_cost"]
        )
        self.assertAlmostEqual(
            result["inc_cost"], expected_incremental_cost, delta=1e-9,
            msg=(
                "No-effect cost invariant violated: incremental cost must equal "
                "the difference in intervention-specific acquisition and administration costs"
            ),
        )
        self.assertGreater(result["inc_cost"], 0.0)

        for state in ("depressed", "responding", "dead"):
            self.assertTrue(
                esk["trace"][state].equals(comp["trace"][state]),
                msg=f"No-effect invariant violated: cycle-start {state} traces differ",
            )


if __name__ == "__main__":
    unittest.main()
