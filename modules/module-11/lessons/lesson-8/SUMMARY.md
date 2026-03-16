# Lesson 8: Summary

Inverse RL infers reward functions from observed behavior, complementing forward RL. Maximum entropy IRL resolves reward ambiguity by choosing highest-entropy consistent rewards. GAIL uses adversarial training for scalability. IRL is valuable for understanding behavior, extracting preferences, and alignment. Preference learning provides an efficient alternative to full demonstrations.

**Core Learning Outcomes:**
- Behavior underdetermines objectives—multiple rewards explain same policy
- Maximum entropy IRL resolves ambiguity through principled assumptions
- GAIL scalably learns via trajectory distribution matching
- IRL enables preference extraction and value alignment
- Preference-based learning is more efficient than demonstration-based approaches

For practitioners: Use preference learning from humans when possible—it's more alignment-friendly than IRL.
