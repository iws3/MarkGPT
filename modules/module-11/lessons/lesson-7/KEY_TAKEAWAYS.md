# Lesson 7: Key Takeaways and Summary

## Model-Based RL Essentials

1. **World models**: Predict environment dynamics, rewards, termination.
2. **Planning with models**: Use learned models to simulate and optimize.
3. **Dyna**: Combine real experience with imagined planning.
4. **MPC**: Continuously re-optimize action sequences online.
5. **Latent space models**: Efficient models in compressed representations.
6. **Model uncertainty**: Quantify when models are unreliable.
7. **Ensemble models**: Multiple models provide uncertainty estimates.
8. **Imagination-augmented learning**: Learn from imagined trajectories.

## Key Algorithms

- **PETS**: Probabilistic ensembles with trajectory sampling.
- **PlaNet**: Latent world model for planning and learning.
- **Dreamer**: World models with imagination-based RL.

## When Model-Based RL Works

- Sample efficiency is critical.
- Simulators available for pre-training models.
- Accurate models learnable (physics-based domains).
- Long-horizon planning needed.

## Challenges

- Compounding model errors over long horizons.
- Learning accurate models in high dimensions.
- Computational cost of planning.

