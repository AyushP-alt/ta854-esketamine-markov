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
- States: depressed, responding/remitted, dead.

## The two assumptions the committee contested

Treatment duration. Drug and administration costs stop after `treatment_cycles` (base case 9, about 36 weeks). Non-responders stop after the induction cycle, which matches the instruction to reassess at the end of induction.

Waning. After stopping, the esketamine arm's relapse rate moves toward the comparator rate. Each cycle off treatment, a fraction `waning_per_cycle` of the remaining incremental effect is lost. Zero is a company-style maintained effect. The base case uses 0.15. The committee-style scenario uses 0.50.

## What the base case says

Run `esketamine_ta854_illustrative.ipynb` or `python model.py` is not the entry point; the notebook is. From this folder:

```bash
python -c "import model; print(model.incremental(model.Inputs())['icer'])"
```

Base case, list price, 5 years: incremental cost about £6,954, incremental QALYs about 0.099, ICER about £70,000 per QALY.

That sits next to the committee's published list-price range of £64,554 to £73,158 per QALY. It does not match it. The company reported about £7,389 to £10,790 per QALY, with incremental QALYs of about 0.32 to 0.37 and incremental costs of about £2,700 to £3,400. Reasons this model differs:

- Three states, not the company's five (no separate remission or recovery, no subsequent treatment lines).
- Relapse is converted from the SUSTAIN-1 cumulative endpoint, not taken from the company's 2.43% and 3.56% recurrence probabilities, and not from STAR*D for the comparator. The committee criticised that mixed-source relapse.
- Re-response after cycle 1 is an illustrative 0.08 in both arms, not a copied transition matrix.
- Session cost uses the 84 mg list price (£489) for every session. The submission used a dose mix and a confidential discount.
- Administration cost is a constructed 6:1 figure (£25 per session), not a published TA unit cost.
- No carer disutility, no adverse-event costs, no excess mortality.

Checks that do pass: every cycle's state probabilities sum to 1, and discounted QALYs (about 2.77 on esketamine and 2.67 on the comparator) are below discounted life-years (about 4.58) in both arms. Drug costs after induction are limited to people who responded in that first cycle.

## Which input moves the ICER

Waning. Moving it from 0 to 0.50 swings the ICER by about £90,000. No waning, with the trial-derived relapse rates, gives about £5,300 per QALY. Fast waning gives about £96,000. Depressed-state utility is next. Treatment duration barely moves the ICER in this structure, because extra cycles add drug cost and QALYs together. That is a result of the simplification, not evidence that duration did not matter to the committee.

## Does any scenario cross £20,000–£30,000?

Yes, one of them, and only on the company side of the dispute.

| Scenario | What is switched on | ICER |
|---|---|---|
| Base | 9 cycles, waning 0.15, admin £25, trial-derived relapse, 84 mg list price | about £70,000 |
| Company-style | waning 0, admin £25, company recurrence rates 2.43% and 3.56% | about £34,000 |
| Company-style, trial relapse | waning 0, admin £25, SUSTAIN-1 derived relapse | about £5,300 |
| Committee-style | waning 0.50, admin £150 (1:1), esketamine relapse raised to 0.10 for selection bias | about £207,000 |

The maintained-effect scenario crosses below £20,000. The committee-style scenario does not enter the £20,000–£30,000 range. It stays well above it. The base case also stays above it.

## Sources

The input log is the first table in the notebook and `outputs/input_log.csv`. Primary public sources:

- NICE TA854 guidance and committee papers (December 2022), including the list price, health-state costs (£980 and £164 per 4-week cycle), mapped utilities (0.417, 0.764, 0.866), and the company/ERG ICERs.
- Popova et al., American Journal of Psychiatry, 2019 (TRANSFORM-2 response 69.3%, remission 52.5%, versus 52.0% and 31.0%).
- Daly et al., JAMA Psychiatry, 2019 (SUSTAIN-1 relapse 26.7% versus 45.3% in stable remitters; median exposure 17.7 weeks).
- BNF list prices quoted in TA854 section 2.3: £163, £326 and £489 for 28 mg, 56 mg and 84 mg.
- ONS national life tables, England, 2018–2020. The age-46 qx of 0.0016 is a sex-weighted approximation, not a copied spreadsheet cell.

## Files

- `esketamine_ta854_illustrative.ipynb` — narrative, input table, base case, scenarios, tornado.
- `model.py` — cohort engine, about the length asked for.
- `outputs/` — tornado, state trace, scenario and OWSA tables.

## Licence of the numbers

TA854 text, the BNF price list and the trial papers are someone else's copyright. This folder stores short figures and citations, not copied text. The code is original.
