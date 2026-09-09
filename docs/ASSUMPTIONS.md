# Baseline Scientific Assumptions

PolicyRecLab is intentionally a **one-step contextual-bandit** environment.

The baseline experiments assume:

1. One action is selected per decision point.
2. The finite set of users acts as the context population.
3. Rewards are Bernoulli with known conditional mean `mu(x, a)`.
4. No interference occurs between decision units.
5. No slate or position effects are present.
6. No sequential carry-over effects are present.
7. The context population is stationary within an experiment.
8. Evaluated target policies are specified independently of sampled evaluation rewards.
9. All actions are eligible for all contexts in the baseline environment.
10. Reward is observed for the chosen action.
11. Behavior-policy probabilities are known in the primary synthetic branch.
12. Positivity/support is required for nonparametric off-policy identification,
    except in milestones that intentionally violate it.

## Milestone-specific status

- **v0.0:** establishes the simulator and directly computed finite-population
  policy value.
- **v0.1:** changes the behavior policy while holding the reward environment
  fixed to demonstrate policy-generated exposure.
- **v0.2:** explicitly checks assumption 12 and intentionally includes one
  deterministic-logging case that violates positivity.

Later milestones should relax or intentionally violate assumptions explicitly
rather than silently changing the data-generating process.
