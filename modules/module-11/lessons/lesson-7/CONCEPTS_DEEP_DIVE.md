# Lesson 7: Key Concepts Deep Dive

## 1. Forward Model Accuracy Requirements

How accurate must world models be? For short-horizon planning, rough models suffice. For long-horizon, small model errors compound catastrophically. Ensemble models help by quantifying uncertainty—when many models disagree, don't trust predictions.

## 2. Latent Representation Learning

Learning models in latent spaces requires learning good representations. Autoencoders provide unsupervised representations; variational autoencoders (VAEs) provide interpretable latent distributions. Representation quality fundamentally limits model and downstream planning quality.

## 3. Planning Horizon Selection

Longer horizon planning = more computation but better long-term decisions. Shorter horizon = faster decisions but myopic. Adaptive horizons (plan further when state uncertainty is low, shorter when high) balance computation and decision quality.

## 4. Model Reuse Across Tasks

Can models learned in one domain transfer to related domains? Transfer depends on domain similarity. Physics models transfer well (similar dynamics); visual models transfer poorly (domain gap in images). Determines feasibility of model reuse.

## 5. Imagination Versus Reality

Agents must distinguish imagined from real data. This prevents models from reinforcing their own errors (agents learning from mistakes in their imagined rollouts). Tracking data source throughout learning enables appropriate weighting.

