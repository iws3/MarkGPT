# Lesson 5: Summary

Meta-learning enables rapid adaptation to new tasks by learning across diverse task distributions. MAML learns good initializations; metric learning learns similarity functions. Few-shot learning leverages this to master new tasks from minimal examples. Task distribution and meta-training procedure fundamentally determine generalization to new tasks.

**Core Learning Outcomes:**
- Meta-learning improves over single-task learning through task diversity
- MAML via bi-level optimization learns task-adaptable initializations
- Metric learning (Prototypicals, Matching Networks) learns distance functions
- Few-shot learning emerges naturally from meta-learning frameworks
- Task distribution design is crucial—too narrow or broad both hurt performance

For practitioners: Use meta-learning when you have diverse related tasks available for pre-training.
