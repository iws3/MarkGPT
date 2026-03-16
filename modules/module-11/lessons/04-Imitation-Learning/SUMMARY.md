# Lesson 4: Summary

Imitation learning enables agents to learn from demonstrations without explicit reward design. Behavioral cloning is simple but suffers from distribution shift—small errors compound into trajectory divergence. DAgger, GAIL, and other methods address this through interactive refinement or adversarial training. IRL inverts the problem to infer rewards from behavior.

**Core Learning Outcomes:**
- Behavioral cloning is fast but limited by distribution shift
- Distribution shift causes exponential error growth in long horizons
- DAgger solves shift interactively with expert queries
- GAIL uses adversarial training for better generalization
- Quality and diversity of demonstrations fundamentally limit performance

For practitioners: Collect diverse, representative expert demonstrations; use DAgger if expert access is available.
