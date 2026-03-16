# Lesson 6: Key Concepts Deep Dive

## 1. Option Discovery Difficulty

Automatically discovering useful options is hard—the space of possible options is infinite. Most methods require either domain knowledge or auxiliary signals (empowerment, subgoal discovery, mutual information maximization).

## 2. Skill Compositionality

Well-designed skills compose—combining sub-skills produces more complex behaviors. This compositionality enables exponential growth in capability from linear growth in learned skills. Achieving compositionality requires careful skill design.

## 3. Temporal Abstraction Trade-offs

Faster skill execution (shorter avg duration) means quicker rewards but less time for sophisticated behaviors. Slower skills enable complexity but reduce decision frequency. Optimal skill timescales vary by problem.

## 4. Goal Specification in Hierarchies

In goal-conditioned hierarchies, how are goals specified? Explicit goals (e.g., "reach position X") are interpretable but restrictive. Learned goal embeddings are flexible but less interpretable.

## 5. Intrinsic Motivation Design

Motivating workers to achieve manager-set goals requires careful reward design. Too weak and workers ignore goals; too strong and workers abandon environmental rewards. Balancing is problem-dependent.

