# v0.1 — Policy-Generated Exposure Bias

## Research question

Can two logging policies operating in the same fixed reward environment produce
materially different observed action exposure and reward distributions?

## Hypothesis

Yes. Logged data are generated jointly by the environment and the behavior
policy. Therefore, the empirical distribution of observed actions and rewards
need not represent the full context-action reward landscape.

## Frozen environment

v0.1 keeps the v0.0 one-step Bernoulli reward environment fixed. It changes only
the behavior policy.

Two behavior policies are compared:

1. **Uniform:** every action has probability `1 / |A|`.
2. **Oracle greedy:** chooses the action with the highest known simulator
   conditional mean reward for each context.

The oracle greedy policy is deliberately synthetic. It is used as a controlled
data-generating intervention, not as a claim about how a production recommender
would know rewards.

## Measurements

For each policy, report:

- exact finite-population on-policy value,
- empirical logged mean reward,
- share of logged actions equal to the context-wise greedy action.

The same environment is used for both policies.

## Interpretation boundary

The logged sample mean estimates the value of the behavior policy that generated
the log. It is **not** an estimator of some different target policy.

This milestone demonstrates policy-generated exposure. It does not yet claim
that IPS or any other off-policy estimator can recover another policy's value.

## Expected result

Uniform logs expose the context-wise greedy action roughly `1 / |A|` of the
time, while deterministic greedy logs expose it every time. Their observed
reward means should consequently differ even though the underlying reward
matrix is unchanged.

## Next milestone

v0.2 will turn this observation into an identification question by studying
positivity/support: which target policies can and cannot be evaluated from a
given logging policy?
