# Lesson 3: Key Takeaways and Summary

## Multi-Agent Dynamics

1. **Non-stationary environments**: Agents' learning changes others' environment.
2. **Game theory concepts**: Nash equilibrium, coordination games, zero-sum games.
3. **Credit assignment complexity**: Attributing global rewards to individual agents.
4. **Emergent behaviors**: Sophisticated strategies arise from agents' mutual adaptation.
5. **Cooperation challenges**: Agents must learn to coordinate without explicit communication.
6. **Population effects**: Individual agent performance depends on population composition.
7. **Training procedures matter**: Self-play, population-based training, exploitation of diversity.

## Key Algorithms

- **QMIX**: Value decomposition for cooperative settings, monotonicity constraint ensuring consistency.
- **Independent Q-learning**: Simplest approach but often unstable.
- **Centralized training, decentralized execution**: Use global information during training, local policies during deployment.
- **Communication learning**: Agents discover communication protocols implicitly.

## Practical Considerations

- **Scalability limits**: Complexity grows exponentially with agent count.
- **Exploration difficulty**: Discovering multi-agent coordination requires careful exploration.
- **Alignment assumptions**: Shared rewards in cooperative games; opposing in competitive.
- **Reproducibility**: Multi-agent systems are sensitive to initialization and stochasticity.

