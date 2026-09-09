# Baseline Scientific Assumptions

PolicyRecLab v0.0 is intentionally a **one-step contextual-bandit** environment.

The baseline experiments assume:

1. One action is selected per decision point.
2. The finite set of users acts as the context population.
3. Rewards are Bernoulli with known conditional mean `mu(x, a)`.
4. No interference occurs between decision units.
5. No slate or position effects are present.
6. No sequential carry-over effects are present.
7. The context population is stationary within an experiment.
8. In v0.0, the evaluated policy is specified independently of observed rewards.
9. All actions are available for all contexts in the baseline environment.
10. Reward is observed for the chosen action.

Later milestones should relax or intentionally violate assumptions explicitly rather
than silently changing the data-generating process.
