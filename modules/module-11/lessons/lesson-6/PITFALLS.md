# Common Pitfalls in Hierarchical RL

## 1. Inappropriate Temporal Abstraction Level
**The Pitfall**: Setting skill duration/scope at wrong granularity.

**Why It Matters**:
- Too fine: no real abstraction, still massive search space
- Too coarse: sub-goals become impossible to reach
- Skills don't compose well

**How to Avoid**:
- Analyze problem: natural subtasks identification
- Measure coverage: can skills reach all important states?
- Test composability: can combos of skills solve problems?

## 2. Skill Learning Reward Misalignment
**The Pitfall**: Skill rewards don't align with high-level task reward.

**Why It Matters**:
- Learned skills don't help solve main task
- Wasted computational effort on irrelevant skills
- Poor hierarchical policy

**How to Avoid**:
- Derive skill rewards from task: related to main objective
- Use mutual information: skills should distinguish relevant states
- Validate: each skill improves task performance

## 3. Hierarchical Abstraction Without Bottom-Up Learning
**The Pitfall**: Imposing hierarchy from top-down without grounding in primitive skills.

**Why It Matters**:
- Abstract skills can't be realized by primitives
- Policy doesn't actually work in practice
- Theory-practice gap

**How to Avoid**:
- Learn primitives first, then compose
- Or joint learning: primitives and skills co-adapt
- Verify: high-level policy achievable by primitives

## 4. Goal Space Design Fails to Induce Useful Skills
**The Pitfall**: Random goal sampling induces trivial or useless skills.

**Why It Matters**:
- Some goals are unreachable from current state
- Some goals trivial to reach
- Skill diversity is low

**How to Avoid**:
- Curriculum: start with reachable goals, increase difficulty
- Measure: skill diversity (mutual information between skills)
- Human in loop: specify desired skill set

## 5. Option Value Function Initialization
**The Pitfall**: Option values not properly initialized for Bellman backup.

**Why It Matters**:
- Wrong initialization biases option selection
- Convergence to suboptimal option policy
- Instability in learning

**How to Avoid**:
- Initialize option values optimistically: high values to encourage exploration
- Or initialize pessimistically: low values for safe initial behavior
- Track initialization impact on final performance

## 6. Termination Function Design Neglect
**The Pitfall**: Using fixed termination probability instead of learned termination.

**Why It Matters**:
- Fixed probability doesn't adapt to task
- Option terminates at wrong time
- Wastes steps or cuts short

**How to Avoid**:
- Learn termination: β(s) = probability of terminating in state s
- Ensure: termination incentivized when option "done"
- Measure: average option duration matches task needs

## 7. State Abstraction Breaks Credit Assignment
**The Pitfall**: Abstraction to fewer state representation loses critical information.

**Why It Matters**:
- Can't distinguish important state differences
- Credit assignment becomes impossible
- Learning signal disappears

**How to Avoid**:
- Validate abstraction: sufficient to distinguish optimal actions
- Measure: loss of information from abstraction
- Progressive abstraction: gradually coarsen state representation

## 8. Exploration-Exploitation Tradeoff at All Levels
**The Pitfall**: Treating hierarch as independent MDPs; exploration not coordinated.

**Why It Matters**:
- High-level exploiting while low-level exploring causes instability
- Or vice versa: low exploring while high locked in
- Poor sample efficiency

**How to Avoid**:
- Synchronized exploration: same ε-greedy at all levels
- Or decoupled: high-level decide what to explore
- Monitor: exploration rate at each level

## 9. Scalability Not Verified with Hierarchy Depth
**The Pitfall**: Assuming multi-level hierarchy scales without testing.

**Why It Matters**:
- Deep hierarchies often don't reduce complexity
- Bottleneck: interfacing between levels
- Empirically may perform worse than flat

**How to Avoid**:
- Measure: total number of steps to solve vs hierarchy depth
- Compare: flat vs 2-level vs 3-level vs ...
- Document: for your domain, optimal depth is ___

## 10. Inter-Level Credit Assignment Ambiguity
**The Pitfall**: When option fails, unclear if skill is bad or high-level selection is bad.

**Why It Matters**:
- Learning signal ambiguous
- Both levels update in conflicting directions
- Convergence stalls

**How to Avoid**:
- Separate credit: use advantage functions at each level
- Log debug information: which level caused failure
- Use intrinsic motivation: measure skill improvement separately

## Summary
Hierarchical RL pitfalls stem from improper abstraction design and credit assignment across levels. The key question: "Is my abstraction natural and learnable from primitives?"
