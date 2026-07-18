#!/bin/bash

branches=(
  "feature/module05-pr01:Dense Retrieval & Semantic Search"
  "feature/module05-pr02:Cross-Encoders & Ranking"
  "feature/module05-pr03:Contrastive Learning for Search"
  "feature/module05-pr04:Retrieval-Augmented QA"
  "feature/module05-pr05:Multi-hop & Conversational QA"
  "feature/module05-pr06:Extractive & Abstractive Summarization"
  "feature/module05-pr07:Sequence-to-Sequence & NMT"
  "feature/module05-pr08:Back-translation & MT Evaluation"
  "feature/module05-pr09:Named Entity Recognition"
  "feature/module05-pr10:Sentiment Analysis & Emotions"
  "feature/module05-pr11:Text Classification & Multi-label"
  "feature/module05-pr12:Relation & Event Extraction"
  "feature/module05-pr13:Vision-Language Models"
  "feature/module05-pr14:Graph Neural Networks for NLP"
  "feature/module05-pr15:Conversational AI & Dialogue"
  "feature/module05-pr16:Zero-shot & Few-shot Learning"
  "feature/module05-pr17:Active Learning Strategies"
  "feature/module05-pr18:Transfer & Domain Adaptation"
  "feature/module05-pr19:Data Augmentation Techniques"
  "feature/module05-pr20:Adversarial Examples & Training"
  "feature/module05-pr21:Interpretability & Explanation"
  "feature/module05-pr22:Biomedical NLP Applications"
  "feature/module05-pr23:Legal NLP & Document Understanding"
  "feature/module05-pr24:Financial NLP & Sentiment"
  "feature/module05-pr25:Social Media & Noisy Text"
  "feature/module05-pr26:Prompt Engineering Essentials"
  "feature/module05-pr27:Retrieval-Augmented Generation"
  "feature/module05-pr28:Code Understanding & Generation"
  "feature/module05-pr29:Multilingual & Cross-lingual Transfer"
  "feature/module05-pr30:Speech & Audio NLP"
  "feature/module05-pr31:Temporal & Dynamic Language"
  "feature/module05-pr32:Long Document Processing"
  "feature/module05-pr33:Bias, Fairness & Ethics"
  "feature/module05-pr34:Privacy & Security in NLP"
  "feature/module05-pr35:Efficient & Sustainable NLP"
  "feature/module05-pr36:Edge Cases & Common Pitfalls"
  "feature/module05-pr37:Research Frontiers & Future"
  "feature/module05-pr38:Lesson-specific Content"
  "feature/module05-pr39:Capstone Project & Resources"
  "feature/module05-pr40:Troubleshooting & Best Practices"
  "feature/module05-pr41:Module Summary & Next Steps"
  "feature/module05-pr42:Lesson 1 - Word Embeddings"
  "feature/module05-pr43:Lesson 2 - ELMo & Contextualization"
)

echo "Creating pull requests from existing branches..."
echo "============================================================"

count=0
for entry in "${branches[@]}"; do
  IFS=':' read -r branch title <<< "$entry"
  count=$((count + 1))
  
  gh pr create \
    --base master \
    --head "$branch" \
    --title "Module-05 PR $count: $title" \
    --body "Comprehensive enrichment of module-05 NLP curriculum" \
    --no-maintainer-edit 2>/dev/null && \
    echo "[OK] $count: $title" || \
    echo "[SKIP] $count: $title (might already exist)"
done

echo "============================================================"
echo "PR creation complete! Check GitHub for all 43+ pull requests."
