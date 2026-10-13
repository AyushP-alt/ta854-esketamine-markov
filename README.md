# Esketamine cohort Markov model (illustrative)

Illustrative model using published TA854 inputs; not a replication.

This is a teaching model of the comparison in the company submission for NICE TA854: esketamine nasal spray plus a newly started oral antidepressant, versus placebo nasal spray plus a newly started oral antidepressant, in adults with treatment-resistant depression.

It is not the company model, not the ERG model, and not a NICE estimate. List price is used. The confidential patient access scheme is not applied. Do not cite the ICERs below as TA854 results.

## Decision problem

- Population: adults with treatment-resistant depression, starting age 46 (SUSTAIN-1 mean 46.3).
- Intervention: esketamine plus a newly started oral antidepressant.
- Comparator: placebo spray plus a newly started oral antidepressant (the TA854 company comparison).
- Perspective: NHS.
- Discount rate: 3.5% per year on costs and QALYs (NICE reference case).
- Cycle: 4 weeks. Horizon: 5 years (65 cycles), the company's original horizon. The committee preferred longer; that is a scenario, not the base case.
- States reported: depressed, responding/remitted, dead. The engine keeps five stocks so an induction failure is not put back on esketamine. Eligible responders relapse at the esketamine rate. Responders who never responded at induction relapse at the comparator rate.

## The two assumptions the committee contested

Treatment duration. Drug and administration costs stop after `treatment_cycles` (base case 9, about 36 weeks). Non-responders stop after the induction cycle, which matches the instruction to reassess at the end of induction.

Waning. After stopping, the esketamine arm's relapse rate moves toward the comparator rate. Each cycle off treatment, a fraction `waning_per_cycle` of the remaining incremental effect is lost. Zero is a company-style maintained effect. The base case uses 0.15. The committee-style scenario uses 0.50.

## What the base case says

The current base case is this command, run from this folder. The notebook still shows the earlier stored output.

```bash
python -c "import model; print(model.incremental(model.Inputs())['icer'])"
```

Base case, list price, 5 years, full-precision relapse probabilities: incremental cost £7,233, incremental QALYs 0.088, ICER £82,131 per QALY. Esketamine has about 2.76 discounted QALYs and the comparator about 2.67. Discounted life-years are about 4.58 in both arms.

That is above the committee's published list-price range of £64,554 to £73,158 per QALY. The company reported about £7,389 to £10,790 per QALY, with incremental QALYs of about 0.32 to 0.37 and incremental costs of about £2,700 to £3,400. Reasons this model differs:

- Three states, not the company's five (no separate remission or recovery, no subsequent treatment lines).
- Relapse is converted from the SUSTAIN-1 cumulative endpoint, not taken from the company's 2.43% and 3.56% recurrence probabilities, and not from STAR*D for the comparator. The committee criticised that mixed-source relapse.
- Re-response after cycle 1 is an illustrative 0.08 in both arms, not a copied transition matrix.
- Session cost uses the 84 mg list price (£489) for every session. The submission used a dose mix and a confidential discount.
- Administration cost is a constructed 6:1 figure (£25 per session), not a published TA unit cost.
- No carer disutility, no adverse-event costs, no excess mortality.

Checks that do pass: every cycle's state probabilities sum to 1, and discounted QALYs are below discounted life-years in both arms. Drug costs after induction are limited to people who responded in that first cycle. A later response from the induction-failure pool is not eligible, and it relapses at the comparator probability.

## Workbook

`TA854_Esketamine_Illustrative_Markov.xlsx` uses the same clinical transitions as `model.py`. `EligResp` relapses at the treatment-adjusted probability. `IneligResp` relapses at `pRelComp`. `DepRelapsed` keeps both new relapses from `EligResp` and people who stayed depressed because they did not re-respond.

The sheet also keeps its own timing. QALYs are half-cycle corrected and the last cycle drops the end state. State and oral costs use the row stock. Waning is applied one transition early. Relapse is stored at six decimals, 0.067787 and 0.127454.

Those conventions are the Python call `run_model(Inputs(p_relapse_esk=0.067787, p_relapse_comp=0.127454), drop_last_half_cycle=True, waning_start_shift=-1, half_cycle_costs=False)`. It reproduces the sheet: esketamine £46,944.21 and 2.736 QALYs, comparator £39,611.03 and 2.652 QALYs, ICER £87,106.80 per QALY.

## Which input moves the ICER

Waning. With the trial-derived relapse rates, waning of 0 gives £14,762 per QALY and waning of 0.50 gives £105,247. The swing is about £90,000. Treatment duration from 4 to 18 cycles moves the ICER by about £9,300. That is much smaller than waning. It is a result of this structure, not evidence that duration did not matter to the committee.

## Does any scenario cross £20,000–£30,000?

Yes, one of them, and only on the company side of the dispute.

| Scenario | What is switched on | ICER |
|---|---|---|
| Base | 9 cycles, waning 0.15, admin £25, trial-derived relapse, 84 mg list price | £82,131 |
| Company-style | waning 0, admin £25, company recurrence rates 2.43% and 3.56% | £47,809 |
| Company-style, trial relapse | waning 0, admin £25, SUSTAIN-1 derived relapse | £14,762 |
| Committee-style | waning 0.50, admin £150 (1:1), esketamine relapse raised to 0.10 for selection bias | £219,884 |

The maintained-effect scenario crosses below £20,000. The committee-style scenario does not enter the £20,000–£30,000 range. It stays well above it. The base case also stays above it. The workbook ICER of £87,106.80 is the same engine with the sheet's timing and six-decimal relapse rates. It is not a fifth clinical scenario.

## Sources

The input log is the first table in the notebook and `outputs/input_log.csv`. Primary public sources:

- NICE TA854 guidance and committee papers (December 2022), including the list price, health-state costs (£980 and £164 per 4-week cycle), mapped utilities (0.417, 0.764, 0.866), and the company/ERG ICERs.
- Popova et al., American Journal of Psychiatry, 2019 (TRANSFORM-2 response 69.3%, remission 52.5%, versus 52.0% and 31.0%).
- Daly et al., JAMA Psychiatry, 2019 (SUSTAIN-1 relapse 26.7% versus 45.3% in stable remitters; median exposure 17.7 weeks).
- BNF list prices quoted in TA854 section 2.3: £163, £326 and £489 for 28 mg, 56 mg and 84 mg.
- ONS national life tables, England, 2018–2020. The age-46 qx of 0.0016 is a sex-weighted approximation, not a copied spreadsheet cell.

## Files

- `model.py` — cohort engine.
- `test_relapse_split.py` — checks that eligible and ineligible responders use different relapse probabilities.
- `TA854_Esketamine_Illustrative_Markov.xlsx` — the same transitions in Excel, with the sheet's half-cycle, waning, and rounding conventions.
- `esketamine_ta854_illustrative.ipynb` — narrative, input table, scenarios, tornado. Its stored results predate the ineligible-relapse correction.
- `outputs/` — tornado, state trace, scenario and OWSA tables from that same earlier run. Re-run `model.py` for the current base case.

## Licence of the numbers

TA854 text, the BNF price list and the trial papers are someone else's copyright. This folder stores short figures and citations, not copied text. The code is original.
