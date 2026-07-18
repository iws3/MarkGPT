#!/usr/bin/env python3
"""
Module-05 phase 3 - 95 commits
Advanced topics: transformers, multimodal, graph neural networks, and more
"""
import subprocess
import os
import sys

os.chdir(r'c:\Users\THE EYE INFORMATIQUE\OneDrive\Desktop\All\MarkGPT-LLM-Curriculum\MarkGPT-LLM-Curriculum\modules\module-05')

sections = [
    # PR 15: Vision-Language Models (5 commits)
    ("## Vision-Language Models\n\n"
     "### Motivation\n\n"
     "Images and text often together\n"
     "Single model for both modalities\n"
     "Enables new capabilities\n"
     "Example: Image captioning\n"
     "Cross-modal retrieval\n\n",
     "Add VLM intro"),
    
    ("### CLIP: Contrastive Learning\n\n"
     "Image encoder and text encoder\n"
     "Contrastive loss: Match image-text pairs\n"
     "Zero-shot image classification\n"
     "Remarkably effective\n"
     "Foundation for many models\n\n",
     "Add CLIP"),
    
    ("### Vision Transformers\n\n"
     "Apply transformers to images\n"
     "Patch embedding: 16x16 patches\n"
     "Linear projection to embeddings\n"
     "Then standard transformer\n"
     "Competitive with CNNs\n\n",
     "Add vision transformers"),
    
    ("### Image Captioning\n\n"
     "Encoder-decoder architecture\n"
     "CNN or ViT encoder\n"
     "Transformer decoder\n"
     "Generates image description\n"
     "Datasets: COCO, Flickr30K\n\n",
     "Add image captioning"),
    
    ("### Visual Question Answering\n\n"
     "Answer questions about images\n"
     "Both image and text reasoning\n"
     "Fusion of modalities\n"
     "VQA dataset\n"
     "Challenging task\n\n",
     "Add visual QA"),
    
    # PR 16: Graph Neural Networks (5 commits)
    ("## Graph Neural Networks for NLP\n\n"
     "### Motivation\n\n"
     "Text has graph structure\n"
     "Dependency trees\n"
     "Knowledge graphs\n"
     "Entity relations\n"
     "GNNs capture structure\n\n",
     "Add GNN intro"),
    
    ("### Graph Convolutional Networks\n\n"
     "Aggregate neighbor information\n"
     "h_i = sigma(sum_j W * h_j)\n"
     "Multiple layers\n"
     "Captures local structure\n"
     "Used for semantic role labeling\n\n",
     "Add GCN"),
    
    ("### Knowledge Graph Embeddings\n\n"
     "Represent entities and relations\n"
     "Triple-based: (h, r, t)\n"
     "Score function: Measure correctness\n"
     "TransE: Simple baseline\n"
     "Link prediction\n\n",
     "Add KG embeddings"),
    
    ("### Text as Graphs\n\n"
     "Dependency parsing: Syntactic structure\n"
     "Semantic graphs: Meaning\n"
     "GNNs improve over linear models\n"
     "Example: Improved NER\n"
     "Structure helps\n\n",
     "Add text graphs"),
    
    ("### Message Passing\n\n"
     "Nodes exchange information\n"
     "Learnable message functions\n"
     "Aggregation of messages\n"
     "Update node representations\n"
     "Flexible framework\n\n",
     "Add message passing"),
    
    # PR 17: Dialogue Systems (5 commits)
    ("## Conversational AI and Dialogue\n\n"
     "### Task Definition\n\n"
     "Generate relevant responses\n"
     "Given conversation history\n"
     "Open-domain chitchat\n"
     "Task-oriented dialogue\n"
     "Different challenges\n\n",
     "Add dialogue intro"),
    
    ("### Response Generation\n\n"
     "Sequence-to-sequence models\n"
     "Encoder: Conversation context\n"
     "Decoder: Generate response\n"
     "Attention important\n"
     "Beam search for decoding\n\n",
     "Add response generation"),
    
    ("### Evaluation\n\n"
     "Automatic metrics: BLEU, ROUGE\n"
     "Limitations: Imperfect correlation\n"
     "Human evaluation: Gold standard\n"
     "Dimensions: Relevance, coherence\n"
     "Informativeness\n\n",
     "Add dialogue eval"),
    
    ("### Personalization\n\n"
     "User persona: Consistent personality\n"
     "Persona description\n"
     "Condition response generation\n"
     "PersonaChat dataset\n"
     "More engaging\n\n",
     "Add persona"),
    
    ("### Goal-oriented Dialogue\n\n"
     "Slot filling: Extract information\n"
     "Dialogue acts: Intent\n"
     "State tracking\n"
     "Dialogue policy: What to say\n"
     "Task completion\n\n",
     "Add task dialogue"),
    
    # PR 18: Zero-shot and Few-shot Learning (5 commits)
    ("## Zero-shot and Few-shot Learning\n\n"
     "### Zero-shot Learning\n\n"
     "No examples for target task\n"
     "Use task description\n"
     "Transfer from pre-training\n"
     "GPT-3 does this\n"
     "Remarkable capability\n\n",
     "Add zero-shot"),
    
    ("### Attribute-based Approach\n\n"
     "Define attributes of classes\n"
     "Example: Birds have feathers, wings\n"
     "Classify by attribute\n"
     "Semantic bridge\n"
     "Limited to known attributes\n\n",
     "Add attributes"),
    
    ("### Few-shot Learning\n\n"
     "K examples per class\n"
     "K=1: One-shot\n"
     "K=5: Few-shot\n"
     "Meta-learning: Learn to learn\n"
     "MAML popular method\n\n",
     "Add few-shot"),
    
    ("### Prototypical Networks\n\n"
     "Average embedding per class\n"
     "Prototype is class center\n"
     "Classify by distance to prototype\n"
     "Simple and effective\n"
     "Works surprisingly well\n\n",
     "Add prototypical"),
    
    ("### Domain Generalization\n\n"
     "Zero-shot to unseen domain\n"
     "Different text distribution\n"
     "Pre-training helps\n"
     "Multi-domain training\n"
     "Open challenge\n\n",
     "Add domain gen"),
    
    # PR 19: Active Learning (5 commits)
    ("## Active Learning\n\n"
     "### Motivation\n\n"
     "Labeling is expensive\n"
     "Which examples to label?\n"
     "Active learning answers this\n"
     "Iterative process\n"
     "Select informative examples\n\n",
     "Add active learning"),
    
    ("### Uncertainty Sampling\n\n"
     "Label most uncertain examples\n"
     "Model confidence low\n"
     "Expected to be informative\n"
     "Easy to implement\n"
     "Often effective\n\n",
     "Add uncertainty"),
    
    ("### Query by Committee\n\n"
     "Ensemble of models\n"
     "Label examples where disagreement\n"
     "High information\n"
     "More expensive\n"
     "Better performance\n\n",
     "Add committee"),
    
    ("### Expected Model Change\n\n"
     "Will example change model?\n"
     "Gradient-based\n"
     "Computationally expensive\n"
     "Principled approach\n"
     "Highest quality\n\n",
     "Add model change"),
    
    ("### Core-set Approach\n\n"
     "Select diverse examples\n"
     "Represent data distribution\n"
     "Minimize coverage\n"
     "Geometric approach\n"
     "Works well\n\n",
     "Add coreset"),
    
    # PR 20: Transfer Learning (5 commits)
    ("## Transfer Learning\n\n"
     "### Pre-training Strategy\n\n"
     "Large unlabeled data\n"
     "Learn general representations\n"
     "Fine-tune on target task\n"
     "Huge improvement\n"
     "Modern default approach\n\n",
     "Add transfer intro"),
    
    ("### Domain Adaptation\n\n"
     "Source and target different\n"
     "Continued pre-training helps\n"
     "Adversarial training\n"
     "Domain discriminator\n"
     "Make representations invariant\n\n",
     "Add domain adapt"),
    
    ("### Task Similarity\n\n"
     "More similar tasks help more\n"
     "Source: Classification\n"
     "Target: Classification\n"
     "vs target: Generation\n"
     "Related tasks better\n\n",
     "Add task similarity"),
    
    ("### Catastrophic Forgetting\n\n"
     "Fine-tune on target\n"
     "Forget source task\n"
     "Problem in continual learning\n"
     "Elastic weight consolidation\n"
     "Parameter isolation\n\n",
     "Add forgetting"),
    
    ("### Multi-task Learning\n\n"
     "Shared representations\n"
     "Multiple task losses\n"
     "Auxiliary tasks help\n"
     "Regularization effect\n"
     "Parameter sharing\n\n",
     "Add multi-task"),
    
    # PR 21: Data Augmentation (5 commits)
    ("## Data Augmentation\n\n"
     "### Back-translation\n\n"
     "Translate to intermediate language\n"
     "Translate back\n"
     "Paraphrase created\n"
     "Preserves meaning\n"
     "Very effective\n\n",
     "Add back-translation aug"),
    
    ("### EDA: Easy Data Augmentation\n\n"
     "Random insertion\n"
     "Random deletion\n"
     "Random swap\n"
     "Synonym replacement\n"
     "Simple but surprisingly effective\n\n",
     "Add EDA"),
    
    ("### Contextual Augmentation\n\n"
     "Language model based\n"
     "Replace words with LM predictions\n"
     "Preserves context\n"
     "Higher quality\n"
     "More computationally expensive\n\n",
     "Add contextual aug"),
    
    ("### Mixup and Cutoff\n\n"
     "Mix embeddings of two examples\n"
     "Interpolate in embedding space\n"
     "Cutoff: Drop random tokens\n"
     "Regularization\n"
     "Helps generalization\n\n",
     "Add mixup"),
    
    ("### Self-training\n\n"
     "Use model on unlabeled data\n"
     "High confidence predictions\n"
     "Add to training set\n"
     "Iterative\n"
     "Semi-supervised approach\n\n",
     "Add self-training"),
    
    # PR 22: Robustness (5 commits)
    ("## Robustness and Adversarial Examples\n\n"
     "### Adversarial Attacks\n\n"
     "Small perturbation\n"
     "Fool model\n"
     "Text: Word level\n"
     "Shows brittleness\n"
     "Security concern\n\n",
     "Add adversarial"),
    
    ("### FGSM Attack\n\n"
     "Fast gradient sign method\n"
     "Gradient of loss wrt input\n"
     "Step in gradient direction\n"
     "Simple and effective\n"
     "Baseline attack\n\n",
     "Add FGSM"),
    
    ("### Adversarial Training\n\n"
     "Generate adversarial examples\n"
     "Train on both original and adv\n"
     "More robust model\n"
     "Cost: Slower training\n"
     "Better generalization\n\n",
     "Add adversarial train"),
    
    ("### Certified Robustness\n\n"
     "Provable robustness\n"
     "Randomized smoothing\n"
     "Guarantees within epsilon\n"
     "Expensive\n"
     "Research active\n\n",
     "Add certified"),
    
    ("### Out-of-distribution Detection\n\n"
     "Detect unusual inputs\n"
     "Maximum softmax\n"
     "Energy-based score\n"
     "Mahalanobis distance\n"
     "Safety-critical systems\n\n",
     "Add OOD detect"),
    
    # PR 23: Explainability (5 commits)
    ("## Explainability and Interpretability\n\n"
     "### Attention Visualization\n\n"
     "Heatmap of weights\n"
     "What model attends to\n"
     "Interpretable but limited\n"
     "Not complete explanation\n"
     "Useful diagnostic\n\n",
     "Add attention interp"),
    
    ("### Feature Attribution\n\n"
     "Gradient-based\n"
     "Integrated gradients\n"
     "Path integration\n"
     "Baseline approach\n"
     "Accumulate gradients\n\n",
     "Add attribution"),
    
    ("### LIME\n\n"
     "Local interpretable model\n"
     "Approximate with simple model\n"
     "Around prediction\n"
     "Perturb inputs\n"
     "Fit linear model\n\n",
     "Add LIME"),
    
    ("### SHAP\n\n"
     "Shapley values\n"
     "Game theory\n"
     "Fair feature contribution\n"
     "Expensive\n"
     "Gold standard\n\n",
     "Add SHAP"),
    
    ("### Probing Tasks\n\n"
     "Train classifier on hidden states\n"
     "What information is encoded\n"
     "Layer-wise analysis\n"
     "Reveals learned structure\n"
     "Syntactic vs semantic\n\n",
     "Add probing"),
]

readme_path = 'README.md'

print(f"Starting module-05 phase 3 with {len(sections)} commits...")
print("=" * 70)

for i, (content, msg) in enumerate(sections, 1):
    with open(readme_path, 'a', encoding='utf-8') as f:
        f.write(content)
    
    try:
        subprocess.run(['git', 'add', 'README.md'], check=True, capture_output=True)
        subprocess.run(['git', 'commit', '-m', msg], check=True, capture_output=True)
        print(f"[OK] {i:3d}: {msg}")
        sys.stdout.flush()
    except Exception as e:
        print(f"[FAIL] {i:3d}: {msg}")

print("=" * 70)
print(f"[DONE] Phase 3 added {len(sections)} commits!")

result = subprocess.run(['git', 'log', '--oneline'], capture_output=True, text=True)
total = len(result.stdout.strip().split('\n'))
print(f"Total repository commits: {total}")
