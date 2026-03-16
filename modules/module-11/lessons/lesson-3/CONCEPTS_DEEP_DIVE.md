# Lesson 3: Key Concepts Deep Dive

## 1. Nash Equilibrium Instability

In multi-agent games, many Nash equilibria often exist. Learning algorithms may converge to suboptimal or even pathological equilibria. Example: both players learning to defect in prisoner's dilemma is a Nash equilibrium but worse than mutual cooperation. Selecting equilibria remains an open problem.

## 2. Non-Stationary Environments in MARL

Standard RL assumes stationary environments. In MARL, other agents' learning creates moving targets—the environment changes as agents adapt. This violates stationarity assumptions, causing algorithms to diverge or fail to converge.

## 3. Credit Assignment Ambiguity

When a team scores, who deserves credit? The agent who scored? The one who passed? Defenders who prevented opponents from scoring? This ambiguity makes learning slow and unstable in naive approaches.

## 4. Communication Emergence

Can agents learn to communicate without explicit design? Through mutual adaptation and pressure to coordinate, implicit communication protocols often emerge—agents learn to convey information through their behavior.

## 5. Tragedy of the Commons

Individual agents optimizing independently can lead to collective disaster. Each agent ignores negative externalities, depleting shared resources. Requires mechanisms (penalties, communication, hierarchy) to align individual and collective interests.

## 6. Emergent Complexity

Sophisticated strategies emerge from simple rules and competitive pressure. Deception, cooperation, territory control—complex behaviors arise naturally without explicit programming.

## 7. Population Diversity Benefits

Diverse strategies in a population prevent convergence to suboptimal equilibria. Maintaining population diversity through mutation and selection enables discovery of better solutions.

