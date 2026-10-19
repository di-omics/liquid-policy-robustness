# Lab notebook

Entries are in order. Every prediction is written down before the run it predicts.

## 2026-10-18: setup

**Commands.** `uv run collect.py`, then `uv run train.py --arch all --seeds 5`.

Both run without any TODO filled in. `collect.py` also prints how well the scripted
teacher does on the evaluation seeds with clean sensors.

**Q1.** Read `expert_action` in `common.py`. Which of the 10 observation numbers does
the teacher use, and which does it ignore? A student network gets all 10. Why could
that matter once the sensors get noisy?

A:

**Q2.** `evaluate.py` scores each episode with `info["reward_dist"]` from the simulator,
not with `obs[8:10]`, the fingertip-minus-target reading the policy sees. What would
go wrong if it used the observation?

A:

**Q3.** The README says a tie with the LSTM gives the credit to memory. What result
would give the credit to the liquid neurons themselves? What would you conclude if
`cfc` beat `lstm` but `ncp` lost to both? (Read the `cfc` and `ncp` rows of the README
table first: what else differs between them besides wiring?)

A:

**Q4.** The noise for each number is scaled by that number's spread in the
demonstrations, instead of one size in meters or radians for all ten. Why? Name one
real sensor problem this noise model does not capture.

A:

**Q5.** On a given seed, every policy receives exactly the same noise draws. What does
that buy you when comparing two policies on only 100 episodes?

A:

**After `train.py`.** It prints each network's best validation error and its largest
torque. Which network fits the teacher worst on clean data? If that network also does
worst under noise, what else could explain it besides noise sensitivity?

A:

## Step 3: noise sweep (`evaluate.py`, `plot.py`)

First fill in the TODO in `common.py` (`SensorNoise.observation`, one line).

Write these before running `uv run evaluate.py`:

- The deciding number (for example: the difference in mean final distance between
  `cfc` and `lstm` at sigma 1.0, with its 95% interval across the 5 training runs):
- How small a difference counts as a tie (for example: the interval lies inside
  +/- 1 cm), and why that margin:
- The teacher's clean score, from the `collect.py` output:
- Your guess for the teacher at sigma 1.0:
- Your guess for the best student at sigma 0: closer to the teacher, or twice as far?
- Rank the five (teacher, mlp, lstm, cfc, ncp) from best to worst at sigma 1.0:
- Does `cfc` degrade more slowly than `lstm` as noise grows? yes or no, and how
  confident (percent):
- One result that would surprise you:

Then:

- Result:
- Interpretation:
- What remains uncertain:
