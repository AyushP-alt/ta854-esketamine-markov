# Esketamine cohort Markov model (illustrative)

This is a small teaching model of one comparison from the company submission for NICE TA854. Adults with treatment-resistant depression start esketamine nasal spray plus a newly started oral antidepressant, or placebo nasal spray plus a newly started oral antidepressant.

The inputs are published figures. The model is a simplified illustration. It is not the company model, the Evidence Review Group model, or a NICE estimate. It uses the list price. The confidential patient access scheme is not applied. Please do not quote the ICERs below as TA854 results.

## Who is being compared

The cohort starts at age 46, close to the SUSTAIN-1 mean of 46.3. Costs and health benefits are counted from the NHS perspective. Both are discounted at 3.5% a year, which is the NICE reference case.

Time moves in 4-week cycles. The base case runs for 5 years, which is 65 cycles. That was the company's original time horizon. The committee preferred a longer one. A longer horizon can be tried as a scenario. It is not the base case.

People are reported in three health states: depressed, responding or remitted, and dead. Behind those totals the model remembers five groups. The extra detail is there so that someone who did not respond to the first course of esketamine is not put back on the drug later. If they do respond later, they relapse at the comparator rate. People who responded to the first course, and are still eligible for esketamine, relapse at the lower esketamine rate.

## Two arguments from the appraisal

The committee and the company disagreed about how long treatment lasts, and about whether the benefit fades after treatment stops.

Drug and administration costs stop after `treatment_cycles`. In the base case that is 9 cycles, about 36 weeks. People who do not respond in the first cycle stop straight away. That follows the instruction to look for benefit at the end of induction.

After esketamine stops, its advantage on relapse fades. Each cycle off treatment, a share `waning_per_cycle` of the remaining advantage is lost. The base case uses 0.15, so 15% of what is left disappears each cycle. A value of 0 means the advantage stays, which is the company-style assumption. The committee-style scenario uses 0.50, so half of the remaining advantage is lost each cycle.

## The base-case result

Run this from the project folder. It prints the current ICER. The notebook still shows an older saved result, from before the relapse rule above was in place.

```bash
python -c "import model; print(model.incremental(model.Inputs())['icer'])"
```

At list price, over 5 years, and with the relapse probabilities kept at full precision, esketamine costs £7,233 more and gives 0.088 more QALYs. The ICER is £82,131 per QALY. A QALY here is a year of life adjusted for quality of life. The esketamine group accumulates about 2.76 discounted QALYs and the comparator about 2.67. Both groups have about 4.58 discounted life-years. Life-years match because both arms face the same risk of death. QALYs are lower because a year lived with depression, or even in response, is worth less than a year in full health.

£82,131 sits above the list-price range of £64,554 to £73,158 per QALY quoted for the committee. The company reported about £7,389 to £10,790 per QALY. In that analysis the extra QALYs were about 0.32 to 0.37 and the extra cost was about £2,700 to £3,400. This model lands elsewhere for a few concrete reasons:

- Health is collapsed into three reported states. The company model had separate remission and recovery states, and further lines of treatment after the first one.
- Relapse is worked out from the SUSTAIN-1 totals: 26.7% and 45.3% of stable remitters relapsed over a median of 17.7 weeks. The company used 2.43% and 3.56% per 4 weeks, and used STAR*D for the comparator. The committee criticised that mix of sources.
- After the first cycle, the chance of responding again is an illustrative 8% in both arms. It is not a probability copied from the company transition matrix.
- Every dosing session is charged at the 84 mg list price, £489. The submission used a mixture of doses and a confidential discount.
- The £25 administration cost per session is a rough 6-patients-to-1-nurse figure. It is not a unit cost published in the appraisal.
- Carer quality of life, side-effect costs, and any effect of treatment on mortality are left out.

A few checks are built in. In every cycle the state shares add up to 1, so the cohort is not gaining or losing people by mistake. Discounted QALYs stay below discounted life-years in both arms, which they should, because quality of life is below full health. After the first cycle, esketamine is paid for only while someone who responded in that first cycle is still responding. Someone who missed that first response is not treated again, and if they respond later they relapse at the comparator rate.

## The Excel workbook

`TA854_Esketamine_Illustrative_Markov.xlsx` follows the same health transitions as `model.py`.

`EligResp` holds people who responded to the first course and are still eligible for esketamine. They relapse at the treatment-adjusted probability, which gets worse after treatment stops. `IneligResp` holds people who are responding but never responded at induction. They relapse at `pRelComp`, the comparator probability. `DepRelapsed` holds two groups: people who have just relapsed out of `EligResp`, and people who were already in that relapsed group and did not respond again this cycle.

The workbook also has a few timing habits of its own. QALYs use the average of the state at the start and end of the cycle, except in the last cycle, where the end state is left out. Health-state costs and the oral antidepressant cost use the state at the start of the cycle. Waning begins one transition earlier than in the default Python run. The relapse probabilities are stored to six decimal places: 0.067787 for esketamine and 0.127454 for the comparator.

This Python call uses those same settings:

```bash
python -c "import model; r=model.run_model(model.Inputs(p_relapse_esk=0.067787, p_relapse_comp=0.127454), drop_last_half_cycle=True, waning_start_shift=-1, half_cycle_costs=False); print(r.icer)"
```

It matches the workbook. Esketamine costs £46,944.21 and produces 2.736 QALYs. The comparator costs £39,611.03 and produces 2.652 QALYs. The ICER is £87,106.80 per QALY.

## What moves the result

Waning moves it most. With the trial-based relapse rates, no waning gives £14,762 per QALY. Losing half of the remaining effect each cycle gives £105,247. The gap is about £90,000.

How long treatment is paid for matters less. Shortening it to 4 cycles or extending it to 18 moves the ICER by about £9,300. That is a feature of this simple structure. It is not evidence that treatment length was unimportant to the committee.

## Is any scenario inside £20,000 to £30,000?

One scenario falls below £20,000. It is the company-style case in which the benefit does not fade. The base case and the committee-style case both stay above £30,000.

| Scenario | What is changed | ICER |
|---|---|---|
| Base | 9 cycles, 15% of the remaining effect lost each cycle, administration £25, trial-based relapse, 84 mg list price | £82,131 |
| Company-style | Benefit kept in full, administration £25, company recurrence rates 2.43% and 3.56% | £47,809 |
| Company-style, trial relapse | Benefit kept in full, administration £25, relapse taken from SUSTAIN-1 | £14,762 |
| Committee-style | Half of the remaining effect lost each cycle, administration £150, esketamine relapse raised to 0.10 as a stand-in for selection bias | £219,884 |

The workbook figure of £87,106.80 is the base-case engine run with the spreadsheet's timing and rounded relapse rates. It is not a separate clinical scenario.

## Sources

The input log is the first table in the notebook and `outputs/input_log.csv`. The public sources are:

- NICE TA854 guidance and committee papers (December 2022), for the list price, health-state costs (£980 depressed and £164 responding per 4-week cycle), mapped utilities (0.417, 0.764, 0.866), and the company and Evidence Review Group ICERs.
- Popova et al., American Journal of Psychiatry, 2019 (TRANSFORM-2). Response was 69.3% versus 52.0%. Remission was 52.5% versus 31.0%.
- Daly et al., JAMA Psychiatry, 2019 (SUSTAIN-1). Among stable remitters, 26.7% relapsed on esketamine and 45.3% on placebo spray. Median exposure was 17.7 weeks.
- BNF list prices quoted in TA854 section 2.3: £163, £326 and £489 for 28 mg, 56 mg and 84 mg.
- ONS national life tables for England, 2018–2020. The age-46 risk of death, 0.0016 per year, is a sex-weighted approximation. It is not a cell copied from the ONS spreadsheet.

## Files

- `model.py` holds the cohort calculations.
- `test_relapse_split.py` checks that eligible and ineligible responders use different relapse probabilities.
- `TA854_Esketamine_Illustrative_Markov.xlsx` is the Excel version, including the sheet's own timing and rounding.
- `esketamine_ta854_illustrative.ipynb` walks through the inputs, scenarios, and tornado chart. The numbers saved in the notebook are from before the ineligible-relapse correction.
- `outputs/` holds the tornado chart, state traces, and scenario tables from that same earlier run. Run `model.py` again for the current base case.

## Licence of the numbers

The TA854 text, the BNF price list, and the trial papers belong to their authors. This folder keeps short figures and citations. It does not copy the source documents. The code is original.
