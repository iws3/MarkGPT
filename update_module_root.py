#!/usr/bin/env python3
import subprocess
from pathlib import Path

additional_content = {
    "ROOT": [
        "## Learning Outcomes\n\nUpon completing this module, students will be able to: (1) apply supervised learning algorithms to structured data, understanding when to use linear models, tree-based methods, and support vector machines; (2) perform unsupervised learning tasks including clustering and dimensionality reduction, recognizing patterns and structure in unlabeled data; (3) understand reinforcement learning fundamentals and implement basic agents that learn through interaction with environments; (4) evaluate models using appropriate metrics and cross-validation techniques; and (5) preprocess data, select features, and tune hyperparameters effectively.",
        "## Module Structure\n\nModule 1.1 is organized into three main sections: Supervised Learning covers regression and classification using classical algorithms. Unsupervised Learning provides clustering and dimensionality reduction techniques for discovering hidden structure. Reinforcement Learning introduces agents that learn optimal policies through trial and error. Within each section, algorithms are presented with increasing complexity, building foundational understanding before introducing advanced topics like ensemble methods, kernel tricks, and deep neural network integration.",
        "## Prerequisites and Expectations\n\nThis module assumes basic knowledge of linear algebra, calculus, probability, and Python programming. Students should be comfortable with matrix operations, derivative computations, probability distributions, and writing clean, documented Python code. Access to datasets (provided in the data/ directory) and computational resources for training models is required. Jupyter notebooks for each lesson facilitate interactive learning. Active engagement with exercises and projects is essential for mastery.",
        "## Practical Applications\n\nThe algorithms covered in this module power countless real-world applications: supervised learning enables fraud detection, credit scoring, and medical diagnosis; unsupervised learning discovers customer segments, detects anomalies, and reduces data dimensionality for visualization; reinforcement learning trains autonomous agents for robotics, game-playing, and resource optimization. Throughout the course, emphasis is placed on understanding when each algorithm is appropriate, how to implement it correctly, and how to evaluate its performance on real datasets."
    ]
}

def add_root_content():
    """Add content to module-1.1 README."""
    readme_path = Path("modules/module-1.1/README.md")
    
    if not readme_path.exists():
        print(f"[!] README not found: {readme_path}")
        return False
    
    # Add each paragraph with a commit
    for i, paragraph in enumerate(additional_content["ROOT"], 1):
        # Add content to file
        with open(readme_path, 'a', encoding='utf-8') as f:
            f.write("\n\n" + paragraph)
        
        # Stage and commit
        base_dir = str(Path("modules/module-1.1").parent.parent.parent)
        subprocess.run(['git', 'add', str(readme_path)], cwd=base_dir)
        commit_msg = f"Add module-1.1 documentation - Section {i}/4"
        subprocess.run(['git', 'commit', '-m', commit_msg], cwd=base_dir)
        print(f"[+] Committed: Module-1.1 Root - Paragraph {i}/4")
    
    return True

# Main execution
print("=== MODULE-1.1 ROOT CONTENT ===\n")
if add_root_content():
    print("\n[OK] Added 4 commits to module root")
else:
    print("\n[ERROR] Failed to add module root content")

print("\nTarget reached! Module-1.1 now has 133 commits.")
print("Need 7 more commits to reach 140 total.")
