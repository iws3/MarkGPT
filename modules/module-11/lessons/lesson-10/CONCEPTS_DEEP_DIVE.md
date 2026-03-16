# Lesson 10: Key Concepts Deep Dive

## 1. Production Versus Research Environments

Research: unlimited interaction, reward is well-specified, failures are acceptable. Production: expensive interaction, complex objectives, failures unacceptable. The shift requires fundamentally different algorithm choices and risk management.

## 2. On-Deployment Distribution Shift

Deployed systems encounter states outside training distribution. Conservative policies that maintain performance in the face of shift are preferred over policies that optimize for training distribution but fail in deployment.

## 3. Exploration-Exploitation Timing

Early training: explore broadly to understand system. As learning progresses: exploitation becomes more important. Managing this transition—exploration budgets—enables safe learning in production.

## 4. Real-World Interaction Cost

Each real interaction is expensive: time, money, potential risk. Minimizing samples required for learning is paramount. This drives emphasis on offline learning and sample-efficient algorithms.

## 5. Monitoring and Alerting

What should be monitored? Primary objective (revenue, safety), constraints (SLA satisfaction), system health (crashes, errors), and staleness (model age, data drift). Automated alerting enables rapid response to problems.

## 6. Rollout Strategies

Gradual deployment (1% → 10% → 100%) detects problems before affecting all users. Multi-armed bandits maintain baseline and new policy in parallel, routing more traffic to better policy. A/B tests provide statistical evidence of improvement.

## 7. Retraining Synchronization

When should policies be retrained? Fixed schedules (daily, weekly) are simple but might miss opportunities or continue with stale data. Event-triggered retraining (when performance drops) is responsive but requires careful threshold tuning.

## 8. System Complexity Management

Well-engineered systems combine RL with rules, learned models, and classical methods. Pure RL rarely outperforms thoughtful combinations. Architecture determines what's possible—good systems separate concerns (perception, reasoning, planning, execution).

