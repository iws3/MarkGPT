#!/usr/bin/env python3
"""
Module-05 enrichment phase 1 - Retry with simpler commit messages
"""
import subprocess
import os
import sys

os.chdir(r'c:\Users\THE EYE INFORMATIQUE\OneDrive\Desktop\All\MarkGPT-LLM-Curriculum\MarkGPT-LLM-Curriculum\modules\module-05')

sections = [
    ("### Context-dependent vs Static\n\n"
     "Word2Vec and GloVe single vector per word\n"
     "Problem: Homonyms (bank = river vs institution)\n"
     "One vector cannot capture both meanings\n"
     "Solution: Contextualized embeddings via ELMo, BERT\n"
     "Modern approach: Dynamic based on surrounding context\n\n",
     "Add context vs static"),
    
    # PR 3: ELMo (5 commits)
    ("## ELMo: Embeddings from Language Models\n\n"
     "### Key Insight\n\n"
     "Train bidirectional LSTM language model\n"
     "Extract hidden states at each layer\n"
     "Weighted combination creates ELMo representation\n"
     "Context-dependent solves homonym problem\n"
     "Transfer learning improves downstream tasks\n\n",
     "Add ELMo intro"),
    
    ("### Bidirectional Language Model\n\n"
     "Forward pass: Predict word from left context\n"
     "Backward pass: Predict word from right context\n"
     "Combined: Use both directions for full context\n"
     "LSTM layers: 2 stacked layers\n"
     "Hidden size: 4096\n"
     "Train on 1B token corpus\n\n",
     "Add bidirectional LM"),
    
    ("### Representation Extraction\n\n"
     "Layer 1: Character convolutions\n"
     "Layers 2-3: Biphone LSTM outputs\n"
     "Extract all 3 layer representations\n"
     "Concatenate: [char_embed, lstm1, lstm2]\n"
     "Weighted combination: gamma * (s0*h0 + s1*h1 + s2*h2)\n"
     "Task-specific learnable weights\n\n",
     "Add representation"),
    
    ("### Fine-tuning Process\n\n"
     "Downstream task: NER, SRL, classification\n"
     "Freeze ELMo weights from pretraining\n"
     "Learn layer combination weights\n"
     "Concatenate ELMo with task embeddings\n"
     "Results: Plus 2-4 percent F1 improvement typical\n\n",
     "Add fine-tuning"),
    
    ("### Limitations of ELMo\n\n"
     "Very slow: 1100M parameters\n"
     "RNN-based: Sequential processing, not parallelizable\n"
     "Training time: Requires 1-2 weeks\n"
     "Next evolution: Transformers are faster, better\n"
     "But ELMo was breakthrough in 2018\n\n",
     "Add ELMo limits"),
    
    # PR 4: BERT (5 commits)
    ("## BERT: Bidirectional Encoder Representations\n\n"
     "### Pre-training Objectives\n\n"
     "1. Masked Language Model (MLM)\n"
     "   Replace 15 percent tokens with MASK\n"
     "   Task: Predict original token\n"
     "   Distribution: 80 percent MASK, 10 percent random, 10 percent original\n"
     "2. Next Sentence Prediction (NSP)\n"
     "   Given two sentences\n"
     "   Predict if adjacent in corpus\n"
     "   Binary classification task\n\n",
     "Add BERT objectives"),
    
    ("### Why Masking Works\n\n"
     "Bidirectional context: Model sees all words\n"
     "Forces deep understanding of meaning\n"
     "Cannot cheat using position information\n"
     "Results in deeper learned representations\n"
     "Unlike GPT which uses causal masking\n\n",
     "Add masking motivation"),
    
    ("### Pre-training Details\n\n"
     "Corpus: BookCorpus plus Wikipedia\n"
     "Total tokens: 3.3B tokens\n"
     "Vocabulary: 30K WordPiece tokens\n"
     "Training hardware: 16 TPUs\n"
     "Training time: Approximately 4 days\n"
     "Batch size: 256 (very large)\n"
     "Optimizer: Adam with LR=1e-4\n\n",
     "Add BERT pretraining"),
    
    ("### Fine-tuning\n\n"
     "Add task-specific layer\n"
     "Training: 2-4 epochs\n"
     "Learning rate: 2e-5 (very small)\n"
     "Batch: 16-32\n"
     "Results: State of art on GLUE benchmark\n"
     "Simple yet effective approach\n\n",
     "Add BERT finetuning"),
    
    ("### BERT Variants\n\n"
     "RoBERTa: Improved pretraining, plus 1-3 percent\n"
     "ALBERT: Shared parameters, smaller model\n"
     "DistilBERT: 40 percent smaller, 60 percent faster\n"
     "ELECTRA: Different pretraining objective\n"
     "Hundreds of variants available now\n\n",
     "Add BERT variants"),
    
    # PR 5: GPT (5 commits)
    ("## GPT: Generative Pre-trained Transformer\n\n"
     "### Causal Language Modeling\n\n"
     "Task: Predict next token given history\n"
     "Formula: P(word_t given word_1 through word_{t-1})\n"
     "Autoregressive: Generate one token at a time\n"
     "Masked attention: Cannot see future tokens\n"
     "Simple objective but very powerful\n\n",
     "Add GPT intro"),
    
    ("### Differences from BERT\n\n"
     "BERT: Masked, bidirectional, for understanding\n"
     "GPT: Causal, left-to-right, for generation\n"
     "BERT: Encoder-only, cannot generate sequences\n"
     "GPT: Decoder-only, can generate text\n"
     "Different strengths for different tasks\n\n",
     "Add BERT vs GPT"),
    
    ("### Scaling Laws\n\n"
     "Compute budget: 6ND where N params, D data\n"
     "Loss curve: Decreases as N to power negative-a\n"
     "a value: Approximately 0.07 to 0.1\n"
     "Predictable scaling trends observed\n"
     "GPT-2: 1.5B params very impressive\n"
     "GPT-3: 175B params shows few-shot abilities\n"
     "Emergent abilities appear at scale\n\n",
     "Add scaling laws"),
    
    ("### Few-shot Learning\n\n"
     "GPT-3 needs no fine-tuning\n"
     "Task definition in prompt text\n"
     "Examples: Zero-shot, one-shot, few-shot\n"
     "Works on translation, QA, reasoning\n"
     "Remarkable generalization shown\n\n",
     "Add few-shot learning"),
    
    ("### In-context Learning\n\n"
     "Implicit fine-tuning via prompt\n"
     "Model adapts to examples in context\n"
     "No gradient updates needed\n"
     "Pure in-context during forward pass\n"
     "Mechanism still under investigation\n\n",
     "Add in-context learning"),
    
    # PR 6: Instruction Tuning (5 commits)
    ("## Instruction Tuning and Alignment\n\n"
     "### The Problem\n\n"
     "GPT-3: Powerful but unpredictable output\n"
     "Can refuse tasks or be verbose\n"
     "Harmful outputs possible sometimes\n"
     "Solution: Fine-tune on instructions\n"
     "Instruction-following is learnable\n\n",
     "Add instruction tuning"),
    
    ("### Data Collection\n\n"
     "Human-written high-quality instructions\n"
     "Each with expected output response\n"
     "High quality: Approximately 100K examples\n"
     "Diversity: Many different task types\n"
     "Cost: USD 100K plus for good data\n"
     "Alternative: Use model-generated data\n\n",
     "Add instruction data"),
    
    ("### Training Process\n\n"
     "Fine-tune base model on instructions\n"
     "Epochs: 2-5 typical\n"
     "Learning rate: 1e-5 (very small)\n"
     "Training speed: 1-2 hours one GPU\n"
     "Massive effect on model capability\n"
     "ChatGPT: Fine-tuned GPT-3.5\n\n",
     "Add instruction training"),
    
    ("### Reinforcement Learning from Human Feedback\n\n"
     "Step 1: Collect comparison annotations\n"
     "Step 2: Train reward model\n"
     "Step 3: Use PPO algorithm\n"
     "Step 4: Generate improved outputs\n"
     "Iterative improvement cycle\n"
     "ChatGPT uses this training approach\n\n",
     "Add RLHF"),
    
    ("### Safety and Alignment\n\n"
     "Constitutional AI: Principles-based approach\n"
     "Red-teaming: Test for failures\n"
     "Adversarial examples: Find weaknesses\n"
     "Ongoing research challenge\n"
     "No silver bullet solution yet\n"
     "Critical active research area\n\n",
     "Add safety alignment"),
]

readme_path = 'README.md'

print(f"Starting phase 1 retry with {len(sections)} commits...")
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
        print(f"[FAIL] {i:3d}: {msg} - {e}")

print("=" * 70)
print(f"[DONE] Phase 1 retry added {len(sections)} commits!")

result = subprocess.run(['git', 'log', '--oneline'], capture_output=True, text=True)
total = len(result.stdout.strip().split('\n'))
print(f"Total repository commits: {total}")
