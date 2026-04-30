#!/usr/bin/env python3
"""
Create 50 GitHub pull requests from module-05 commits
Each PR contains ~5 related commits
"""
import subprocess
import os
import sys

os.chdir(r'c:\Users\THE EYE INFORMATIQUE\OneDrive\Desktop\All\MarkGPT-LLM-Curriculum\MarkGPT-LLM-Curriculum\modules\module-05')

# Get all recent commits
result = subprocess.run(['git', 'log', '--oneline', '-250'], capture_output=True, text=True)
commits = result.stdout.strip().split('\n')
commits.reverse()  # Oldest first

print(f"Found {len(commits)} commits")
print("=" * 70)

# PR definitions: each PR has a name and number of commits
prs = [
    ("Dense Retrieval & Semantic Search", 5),
    ("Bi-encoders & Cross-encoders", 5),
    ("Question Answering Systems", 5),
    ("Abstractive Summarization", 5),
    ("Machine Translation", 5),
    ("Named Entity Recognition", 5),
    ("Sentiment Analysis", 5),
    ("Text Classification", 5),
    ("Information Extraction", 5),
    ("Vision-Language Models", 5),
    ("Graph Neural Networks", 5),
    ("Dialogue Systems", 5),
    ("Zero-shot & Few-shot Learning", 5),
    ("Active Learning", 5),
    ("Transfer Learning", 5),
    ("Data Augmentation", 5),
    ("Adversarial Robustness", 5),
    ("Explainability & Interpretability", 5),
    ("Biomedical NLP", 5),
    ("Legal NLP", 5),
    ("Financial NLP", 5),
    ("Social Media NLP", 5),
    ("Prompt Engineering", 5),
    ("Retrieval-Augmented Generation", 5),
    ("Code Understanding", 5),
    ("Multilingual NLP", 5),
    ("Speech Processing", 5),
    ("Temporal NLP", 5),
    ("Long Document Processing", 5),
    ("Bias & Fairness", 5),
    ("Privacy & Security", 5),
    ("Efficiency & Sustainability", 5),
    ("Edge Cases & Pitfalls", 5),
    ("Research Directions", 5),
    ("Lesson-01 Foundations", 5),
    ("Lesson-02 ELMo Models", 5),
    ("Lesson-03 BERT Transformers", 5),
    ("Lesson-04 GPT Models", 5),
    ("Lesson-05 Alignment", 5),
    ("Capstone Projects", 5),
    ("Learning Resources", 5),
    ("Troubleshooting Guide", 5),
    ("Final Reflections", 5),
]

# Verify total
total_commits_needed = sum(c[1] for c in prs)
print(f"PR structure needs: {total_commits_needed} commits")
print(f"Available: {len(commits)} commits")
print("=" * 70)

if len(commits) < total_commits_needed:
    print(f"ERROR: Not enough commits! Need {total_commits_needed}, have {len(commits)}")
    sys.exit(1)

# Create branches for each PR
commit_idx = 0
pr_branches = []

for pr_num, (pr_name, num_commits) in enumerate(prs, 1):
    branch_name = f"feature/module05-pr{pr_num:02d}"
    
    # Get the commit range for this PR
    start_idx = commit_idx
    end_idx = commit_idx + num_commits
    
    if end_idx > len(commits):
        print(f"ERROR: Ran out of commits at PR {pr_num}")
        break
    
    # Get the commit hash for the base (one before this PR's commits)
    if start_idx == 0:
        base_commit = "HEAD~250"  # Start from 250 commits ago
    else:
        base_commit = commits[start_idx - 1].split()[0]
    
    # Get the commit hash for the head (last commit of this PR)
    head_commit = commits[end_idx - 1].split()[0]
    
    print(f"\n[{pr_num:2d}/50] Creating branch: {branch_name}")
    print(f"        PR: {pr_name}")
    print(f"        Commits: {num_commits} (commits #{start_idx + 1}-{end_idx})")
    print(f"        From: {base_commit[:8]} to {head_commit[:8]}")
    
    # Create the branch
    try:
        # Create branch at the head commit
        subprocess.run(['git', 'branch', branch_name, head_commit], check=True, capture_output=True)
        print(f"        ✓ Branch created")
        
        pr_branches.append({
            'num': pr_num,
            'branch': branch_name,
            'name': pr_name,
            'commits': num_commits,
            'head': head_commit[:8]
        })
        
    except Exception as e:
        print(f"        ✗ Error: {e}")
    
    commit_idx = end_idx

print("\n" + "=" * 70)
print(f"Created {len(pr_branches)} branches")
print("=" * 70)

# Push all branches
print("\nPushing branches to GitHub...")
for pr in pr_branches:
    try:
        subprocess.run(['git', 'push', '-u', 'origin', pr['branch']], 
                      check=True, capture_output=True, timeout=10)
        print(f"[{pr['num']:2d}] Pushed {pr['branch']}")
    except subprocess.TimeoutExpired:
        print(f"[{pr['num']:2d}] Timeout pushing {pr['branch']} (may still be uploading)")
    except Exception as e:
        print(f"[{pr['num']:2d}] Error pushing {pr['branch']}: {e}")

print("\n" + "=" * 70)
print("NEXT STEPS:")
print("=" * 70)
print("\nTo create actual GitHub PRs, use GitHub CLI:")
print("\n  gh pr create --base main --head feature/module05-pr01 --title 'PR 01: Dense Retrieval & Semantic Search' --body 'Add dense retrieval and semantic search implementation'")
print("\nOr go to GitHub.com and create PRs manually from the pushed branches.")
print("\nBranches created:")
for pr in pr_branches[:5]:
    print(f"  - {pr['branch']} ({pr['name']})")
print(f"  ... and {len(pr_branches) - 5} more")
