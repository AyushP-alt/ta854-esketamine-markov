"""RELIG uses the arm relapse rate. RINEL uses the comparator rate."""

import unittest
from unittest.mock import patch

import numpy as np

import model
from model import (
    DEAD_X,
    FDEP,
    RELIG,
    RDEP,
    RINEL,
    Inputs,
    arm_rates,
    cycle_death_prob,
    run_arm,
    transition_matrix,
)


class RelapseSplitTests(unittest.TestCase):
    def test_relig_and_rinel_use_separate_relapse_probabilities(self):
        p_respond = 0.08
        p_eligible = 0.05
        p_ineligible = 0.20
        p_death = 0.01
        stay = 1.0 - p_death
        matrix = transition_matrix(
            p_respond,
            p_relapse_eligible=p_eligible,
            p_relapse_ineligible=p_ineligible,
            p_death=p_death,
            induction=False,
        )

        self.assertTrue(np.all(np.isfinite(matrix)))
        self.assertTrue(np.all((matrix >= 0.0) & (matrix <= 1.0)))
        self.assertTrue(np.allclose(matrix.sum(axis=1), 1.0))

        self.assertAlmostEqual(matrix[RELIG, RDEP], p_eligible * stay)
        self.assertAlmostEqual(matrix[RELIG, RELIG], (1.0 - p_eligible) * stay)
        self.assertAlmostEqual(matrix[RELIG, DEAD_X], p_death)
        self.assertAlmostEqual(matrix[RELIG, FDEP], 0.0)

        self.assertAlmostEqual(matrix[RINEL, FDEP], p_ineligible * stay)
        self.assertAlmostEqual(matrix[RINEL, RINEL], (1.0 - p_ineligible) * stay)
        self.assertAlmostEqual(matrix[RINEL, DEAD_X], p_death)
        self.assertAlmostEqual(matrix[RINEL, RDEP], 0.0)
        self.assertNotAlmostEqual(matrix[RINEL, FDEP], p_eligible * stay)

        # Response routing is unchanged: after induction, FDEP goes to RINEL.
        self.assertAlmostEqual(matrix[FDEP, RINEL], p_respond * stay)
        self.assertAlmostEqual(matrix[FDEP, FDEP], (1.0 - p_respond) * stay)
        self.assertAlmostEqual(matrix[FDEP, RELIG], 0.0)
        self.assertAlmostEqual(matrix[RDEP, RELIG], p_respond * stay)
        self.assertAlmostEqual(matrix[RDEP, RDEP], (1.0 - p_respond) * stay)

    def test_every_arm_cycle_matrix_is_a_valid_probability_matrix(self):
        inputs = Inputs()
        p_death = cycle_death_prob(inputs.annual_qx)
        stay = 1.0 - p_death
        for cycle in (0, 1, 8, 9, 10, 64):
            for esk in (True, False):
                for shift in (0, -1):
                    p_respond, p_eligible = arm_rates(
                        cycle, esk, inputs, waning_start_shift=shift
                    )
                    matrix = transition_matrix(
                        p_respond,
                        p_relapse_eligible=p_eligible,
                        p_relapse_ineligible=inputs.p_relapse_comp,
                        p_death=p_death,
                        induction=(cycle == 0),
                    )
                    self.assertTrue(np.all(np.isfinite(matrix)))
                    self.assertTrue(np.all((matrix >= 0.0) & (matrix <= 1.0)))
                    self.assertTrue(np.allclose(matrix.sum(axis=1), 1.0))
                    self.assertAlmostEqual(matrix[RELIG, RDEP], p_eligible * stay)
                    self.assertAlmostEqual(matrix[RINEL, FDEP], inputs.p_relapse_comp * stay)
                    if not esk:
                        self.assertAlmostEqual(p_eligible, inputs.p_relapse_comp)

    def test_run_arm_passes_waned_eligible_rate_and_comparator_rate(self):
        inputs = Inputs()
        captured = []
        real = model.transition_matrix

        def wrapper(p_respond, p_relapse_eligible, p_relapse_ineligible, p_death, induction):
            captured.append((p_relapse_eligible, p_relapse_ineligible))
            return real(
                p_respond,
                p_relapse_eligible,
                p_relapse_ineligible,
                p_death,
                induction,
            )

        with patch.object(model, "transition_matrix", wrapper):
            run_arm(True, inputs)
        _, on_treatment = arm_rates(8, True, inputs)
        _, waned = arm_rates(9, True, inputs)
        self.assertAlmostEqual(captured[8][0], on_treatment)
        self.assertAlmostEqual(captured[9][0], waned)
        self.assertLess(on_treatment, waned)
        self.assertLess(waned, inputs.p_relapse_comp)
        self.assertAlmostEqual(captured[9][1], inputs.p_relapse_comp)
        self.assertNotAlmostEqual(captured[9][0], captured[9][1])

        captured.clear()
        with patch.object(model, "transition_matrix", wrapper):
            run_arm(True, inputs, waning_start_shift=-1)
        _, shifted = arm_rates(8, True, inputs, waning_start_shift=-1)
        self.assertAlmostEqual(captured[8][0], shifted)
        self.assertAlmostEqual(captured[8][0], waned)
        self.assertAlmostEqual(captured[8][1], inputs.p_relapse_comp)

        captured.clear()
        with patch.object(model, "transition_matrix", wrapper):
            run_arm(False, inputs)
        for eligible, ineligible in captured:
            self.assertAlmostEqual(eligible, inputs.p_relapse_comp)
            self.assertAlmostEqual(ineligible, inputs.p_relapse_comp)


if __name__ == "__main__":
    unittest.main()
