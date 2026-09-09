# v1.0 — Repeated Policy Feedback Loops

## Research question

What happens when a recommendation policy does not merely evaluate logged data,
but actively determines the data that the next policy will learn from?

The repeated loop is

\[
\pi_t
\rightarrow
\text{exposure under }\pi_t
\rightarrow
\text{logged rewards}
\rightarrow
\text{learner}
\rightarrow
\pi_{t+1}.
\]

This is the first PolicyRecLab milestone in which policy-generated exposure
changes the data-generating process for future policies.

## Controlled mechanism

The simulator environment remains fixed.

At deployment round \(t\):

1. the current policy generates a logged contextual-bandit sample,
2. rewards are observed only for exposed actions,
3. a smoothed context-action reward table is fitted from that deployment's
   observations,
4. the next policy is epsilon-greedy with respect to those fitted scores.

This learner is intentionally simple. The purpose of v1.0 is to isolate the
feedback mechanism rather than to optimize predictive performance.

## Two exploration regimes

The main comparison uses:

- a more exploratory epsilon-greedy loop,
- a more exploitative epsilon-greedy loop.

Both start from the same uniform initial policy.

The experiment tracks whether lower exploration makes future logs more
concentrated and less useful for evaluating broad alternative policies.

## Diagnostics

Each deployment records:

- exact finite-population value of the deployed policy,
- normalized catalog exposure entropy,
- exposure Gini coefficient,
- minimum observed action share,
- maximum observed action share,
- minimum behavior probability relevant to a uniform target.

The last quantity is a direct future-evaluability diagnostic.

For epsilon-greedy policies over \(K\) actions, non-greedy actions receive

\[
\pi_b(a\mid x)=\frac{\epsilon}{K}.
\]

As epsilon shrinks, the uniform target remains formally supported while overlap
can become extremely weak.

## Feedback amplification

The scientific mechanism is:

\[
\text{policy choice}
\rightarrow
\text{which actions are observed}
\rightarrow
\text{which rewards are learned well}
\rightarrow
\text{next policy}.
\]

Early random estimation errors can therefore affect later exposure.

This is a policy-generated data feedback loop.

v1.0 does not claim that every loop must collapse to one action, that lower
exploration must always hurt long-run reward, or that exposure concentration
must increase monotonically at every finite deployment.

Those are empirical properties to measure, not universal laws.

## Exposure concentration is not identical to contextual support

Catalog-level exposure entropy and Gini summarize marginal exposure.

They do not prove contextual positivity.

A system could expose every action somewhere while still assign zero or tiny
probability to an action for specific contexts.

Therefore v1.0 reports both marginal concentration diagnostics and the minimum
behavior probability across the full context-action policy table.

## Inference scope

The logging policy changes over deployments.

This makes the overall sequence adaptive.

v1.0 therefore treats the repeated loop as a controlled simulation study and
does not apply ordinary IID confidence intervals across the entire adaptive
sequence.

Later adaptive-inference work would require sequential or martingale-valid
methods rather than blindly reusing static-policy IID intervals.

## Connection to earlier milestones

v0.6 showed that exploration affects future evaluability in a one-shot logging
experiment.

v1.0 extends that idea dynamically:

\[
\text{exploration today}
\rightarrow
\text{training data tomorrow}
\rightarrow
\text{future policy behavior}.
\]

v0.9 also remains relevant: if policies are repeatedly selected using noisy
offline estimates, selection bias can interact with this feedback loop.

## Next milestone

v1.1 integrates the project into an OPE reliability system. Every reported
offline policy value should carry identification, overlap, estimator, policy
selection, and inference diagnostics rather than appearing as an isolated
number.
