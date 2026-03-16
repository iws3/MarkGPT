# Common Pitfalls in Multi-Agent Reinforcement Learning

## 1. Ignoring Non-Stationarity
**The Pitfall**: Treating multi-agent environments as stationary when other agents are learning.

**Why It Matters**:
- Other agents' policies change during training
- Breaks convergence guarantees for single-agent algorithms
- Policy instability and oscillation

**How to Avoid**:
- Implement independent learners with explicit non-stationary handling
- Use experience from different agent perspectives
- Consider population-based training

## 2. Improper Credit Assignment Across Agents
**The Pitfall**: Using global rewards without proper decomposition, blaming/crediting wrong agents.

**Why It Matters**:
- Agents can't learn their individual contributions
- Free-rider problem: some agents do nothing
- Scales poorly with agent count

**How to Avoid**:
- Use value decomposition (QMIX): combine individual values monotonically
- Implement intrinsic motivation based on individual agent actions
- Clearly separate individual and team rewards

## 3. Symmetry and Permutation Invariance Ignored
**The Pitfall**: Treating agent ordering as meaningful when agents are symmetric or interchangeable.

**Why It Matters**:
- Unnecessary state bloat from permutations
- Poor generalization to different team sizes
- Sample inefficiency

**How to Avoid**:
- Use permutation-invariant architectures (mean/max pooling over agents)
- Graph neural networks to handle variable team sizes
- Test: performance should be unchanged by agent reordering

## 4. Wrong Communication Protocol Design
**The Pitfall**: Implementing unlimited/unstructured communication channel without constraints.

**Why It Matters**:
- Agents can trivialize the task through perfect communication
- Doesn't reflect real-world constraints
- Hides underlying coordination failures

**How to Avoid**:
- Bottleneck communication: limited bandwidth
- Communication protocol separate from learning
- Compare performance with/without communication

## 5. Neglecting Opponent Modeling
**The Pitfall**: In competitive settings, not considering what opponent might do.

**Why It Matters**:
- Agents can overfit to specific opponent policies
- Exploitable strategies fail against adaptive opponents
- Suboptimal equilibrium play

**How to Avoid**:
- In self-play: use diverse population as opponents
- Self-awareness: "What would I do if I were my opponent?"
- Robust optimization: train against multiple opponent types

## 6. Scalability Assumption Without Testing
**The Pitfall**: Assuming multi-agent algorithm scales to 100+ agents without checking runtime.

**Why It Matters**:
- Communication overhead grows quadratically
- Coordination becomes combinatorially hard
- Tractability limits often hit at 10-20 agents

**How to Avoid**:
- Always run scalability tests: measure wall-clock time vs agent count
- Implement hierarchical/modular approaches for large teams
- Document algorithmic complexity

## 7. Mixing Cooperative and Competitive Assumptions
**The Pitfall**: Using cooperative algorithms (e.g., QMIX) in competitive tasks.

**Why It Matters**:
- Theoretical guarantees break down
- Agents learn to collude when they should compete
- Convergence to wrong solution concept

**How to Avoid**:
- Explicitly declare: "This is cooperative/competitive/mixed"
- Use appropriate solution concepts: Nash equilibrium for competitive
- Run diagnostic: check if agents collude when shouldn't

## 8. Ignoring Exploration Challenges in Multi-Agent
**The Pitfall**: Using single-agent epsilon-greedy in cooperative exploration.

**Why It Matters**:
- Exploration needs coordination (different agents explore different parts)
- Random exploration by one agent can destroy team performance
- Sample inefficiency

**How to Avoid**:
- Implement coordinated epsilon-greedy: same exploration signal to all agents
- Or use agent-specific noise: different agents have different noise
- Use entropy regularization to encourage diversity

## 9. Terminal State Handling with Mixed Outcomes
**The Pitfall**: Ambiguous terminal state definition when agents have conflicting goals.

**Why It Matters**:
- Does game end when one agent wins? Both lose?
- Value of terminal state is team-dependent
- Learning feedback inconsistent

**How to Avoid**:
- Explicitly define terminal conditions
- Specify reward for each terminal outcome per agent
- Test invariant: sum of agent rewards should be consistent

## 10. Environment Curriculum Not Synchronized
**The Pitfall**: Advancing difficulty for some agents while others struggle.

**Why It Matters**:
- Agents trained at different skill levels can't coordinate
- Curriculum learning fails to improve team performance
- Catastrophic forgetting when difficulty jumps

**How to Avoid**:
- Synchronize curriculum across all agents
- Monitor team performance, not individual performance
- Gradual difficulty increase based on team metric

## Summary
Multi-agent RL pitfalls center on handling non-stationarity, credit assignment, and coordination. Focus on making each agent's learning signal clear and aligned with team objectives.
