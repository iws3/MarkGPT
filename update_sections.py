#!/usr/bin/env python3
import subprocess
from pathlib import Path

section_content = {
    "1_supervised_learning": [
        "## Overview of Supervised Learning\n\nSupervised learning encompasses algorithms that learn to predict targets by observing labeled training data examples. These fundamentals are essential because most practical machine learning problems involve some form of supervision—predicting house prices, classifying emails as spam, or diagnosing diseases. The key insight is that the quality of supervision data directly determines learning quality; clean, representative, well-labeled data is prerequisite for effective supervised models.",
        "## Regression vs Classification\n\nSupervised learning divides into regression (predicting continuous values) and classification (predicting discrete categories). Regression predicts quantities like prices, temperatures, or stock values. Classification predicts categories like spam/ham, disease/healthy, or species type. These two tasks require different loss functions and evaluation metrics: regression uses mean squared error and R-squared, while classification uses cross-entropy loss and accuracy/precision/recall.",
        "## Hyperparameter Tuning and Model Selection\n\nAfter training a model, practitioners must select hyperparameters: learning rates, regularization strengths, tree depths, and ensemble sizes. Grid search exhaustively evaluates combinations; random search samples randomly. Cross-validation on multiple data folds provides robust performance estimates. The validation curve shows performance vs. hyperparameter values; the learning curve shows performance vs. training set size, revealing whether more data or better models help.",
        "## Deployment and Monitoring\n\nOnce trained, models are deployed to make predictions on new data. Monitoring is critical: model performance degrades over time if new data drifts from training distribution. Detecting performance degradation and retraining regularly maintains prediction quality. A/B testing compares model versions. Explainability becomes crucial in high-stakes applications like healthcare or finance. Techniques like SHAP values and LIME explain individual predictions.",
        "## Common Pitfalls and Best Practices\n\nCommon mistakes include data leakage (inadvertently using test data during training), train/test mismatch (different distributions), ignoring class imbalance, and overfitting to small datasets. Best practices include proper data splitting, stratified sampling, careful feature engineering, and always validating on held-out data. Starting with simple baseline models before complex algorithms prevents unnecessary complexity."
    ],
    "2_unsupervised_learning": [
        "## The Unsupervised Learning Challenge\n\nUnlike supervised learning where targets guide learning, unsupervised learning must discover structure from data alone. This is both powerful—no expensive labeling required—and challenging: there's no ground truth to evaluate whether discovered patterns are meaningful or artifacts. Unsupervised learning is essential for exploratory data analysis, preprocessing, and applications where labels are unavailable or expensive.",
        "## Clustering Applications\n\nClustering has numerous applications: customer segmentation identifies purchasing patterns; gene expression clustering reveals disease subtypes; image clustering organizes photo libraries; and text clustering groups similar documents. In each case, clustering discovers natural groupings without predefined categories. The challenge is that optimal clustering depends on application context; there's rarely a single 'correct' clustering.",
        "## Dimensionality Reduction Benefits\n\nReducing dimensions improves computational efficiency, reduces overfitting due to fewer features, and enables visualization of high-dimensional data. PCA and manifold learning techniques reveal intrinsic dimensionality: datasets might be high-dimensional but lie near lower-dimensional structures. Understanding intrinsic structure guides feature engineering and model complexity selection.",
        "## Combining Unsupervised and Supervised Learning\n\nUnsupervised learning frequently serves as a preprocessing step for supervised learning. Clustering can create features (cluster membership); dimensionality reduction reduces computational cost; and outlier detection removes problematic training examples. Semi-supervised learning combines both: learning from both labeled and unlabeled data. Transfer learning leverages structure learned on unsupervised tasks to improve supervised performance."
    ],
    "3_reinforcement_learning": [
        "## From Passive Observation to Active Learning\n\nReinforcement learning fundamentally differs from supervised and unsupervised learning: agents learn through interaction and feedback. An agent takes actions in an environment, receives rewards or penalties, and learns to maximize cumulative reward. This framework models learning as sequential decision-making—the essence of intelligence. Applications range from game-playing (AlphaGo), to robotics (robot control), to autonomous driving, to recommendation systems.",
        "## Exploration vs Exploitation\n\nA central tension in reinforcement learning is exploration-exploitation: should the agent try uncertain actions to discover better strategies (explore) or exploit known good actions (exploit)? Effective agents balance both: initially exploring to discover good strategies, then exploiting them with increasing confidence. Sophisticated exploration strategies substantially outperform naive random exploration, reducing sample requirements for learning.",
        "## Sample Efficiency and Learning Speed\n\nReinforcement learning is sample-inefficient: learning good policies often requires millions of environment interactions. This creates a practical bottleneck for real-world applications. Techniques to improve sample efficiency include experience replay (reusing past experiences), prioritized replay (focusing on important experiences), and off-policy learning (learning from past policies). Transfer learning and multi-task learning can leverage prior knowledge to accelerate learning."
    ]
}

def add_section_content():
    """Add content to section READMEs."""
    total = 0
    for section, content_list in section_content.items():
        readme_path = Path(f"modules/module-1.1/{section}/README.md")
        
        if not readme_path.exists():
            print(f"[!] {section} README not found")
            continue
        
        # Add each paragraph
        for i, paragraph in enumerate(content_list, 1):
            with open(readme_path, 'a', encoding='utf-8') as f:
                f.write("\n\n" + paragraph)
            
            # Stage and commit
            base_dir = "."
            subprocess.run(['git', 'add', str(readme_path)], cwd=base_dir, capture_output=True)
            commit_msg = f"Add {section.replace('_', ' ').title()} content - Section {i}/{len(content_list)}"
            subprocess.run(['git', 'commit', '-m', commit_msg], cwd=base_dir, capture_output=True)
            print(f"[+] {section} paragraph {i}")
            total += 1
    
    return total

# Execute
print("Adding section-level content...\n")
commits_added = add_section_content()
print(f"\nAdded {commits_added} commits!")
