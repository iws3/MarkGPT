# Project Ideas & Student Capstone Guide

## Beginner Projects (Week 1-2)

### Project 1: Custom Dataset Fine-tuning
**Difficulty**: ⭐ (Beginner)
**Time**: 4-6 hours

**Objective**: Fine-tune MarkGPT-Nano on a custom text corpus and measure improvements

**Steps**:
1. Collect 1000+ documents in your domain (e.g., news articles, poetry, technical docs)
2. Prepare dataset using `src/utils/datasets.py`
3. Run LoRA fine-tuning from FINETUNING_BEST_PRACTICES.md
4. Evaluate before/after using perplexity and human judgment
5. Document results in a markdown report

**Expected Output**:
- Fine-tuned model checkpoint
- Before/after perplexity comparison
- 5 example generations (before/after)
- 500-word reflection on results

**Starter Code**:
```python
from src.training.train import train_lora
from src.utils.datasets import load_json_dataset

# Load custom dataset
dataset = load_json_dataset('my_data.jsonl')

# Fine-tune
checkpoint = train_lora(
    model_name='markgpt-nano',
    dataset=dataset,
    num_epochs=3,
    output_dir='./custom_checkpoint'
)

# Evaluate
from src.utils.evaluation import evaluate_perplexity
ppl = evaluate_perplexity(checkpoint, test_set)
print(f"Perplexity: {ppl:.2f}")
```

---

### Project 2: Multilingual Generation
**Difficulty**: ⭐ (Beginner)  
**Time**: 6-8 hours

**Objective**: Generate text in both English and Banso, compare quality

**Multilingual Tasks**:
1. Generate 10 prompts in English
2. Translate to Banso using reference material
3. Generate with MarkGPT for both languages
4. Compare outputs with human evaluation rubric

**Rubric** (1-5 scale):
- Coherence: Does it make sense?
- Relevance: Does it match the prompt?
- Fluency: Is the language natural?
- Correctness: Is it factually accurate?

**Deliverable**:
- Spreadsheet with all outputs and ratings
- Analysis: Which language performed better? Why?

---

## Intermediate Projects (Week 3-4)

### Project 3: Model Compression & Deployment
**Difficulty**: ⭐⭐ (Intermediate)
**Time**: 12-16 hours

**Objective**: Compress MarkGPT-50M and deploy to different devices

**Tasks**:
1. **Quantization**: INT8 quantization using torch.quantization
2. **Distillation**: Train smaller student model from teacher
3. **Deployment**: 
   - [ ] Flask API on laptop
   - [ ] Docker containerization
   - [ ] Optional: HuggingFace Hub deployment
4. **Benchmarking**: Compare speed/size/quality tradeoffs

**Metrics to Compare**:
| Method | Model Size | Inference Time | Accuracy @ 100 tokens |
|---|---|---|---|
| Original | 100MB | 450ms | 98% |
| INT8 Quantized | 25MB | 180ms | ? |
| Distilled 50% | 50MB | 200ms | ? |
| Distilled + Quantized | 12MB | 75ms | ? |

**Deliverable**:
- Deployed model (link or Docker image)
- Benchmark comparison table
- Technical report (1000 words)
- GitHub repo with code

---

### Project 4: Evaluation Framework Implementation
**Difficulty**: ⭐⭐ (Intermediate)
**Time**: 10-12 hours

**Objective**: Build automated evaluation pipeline for model outputs

**Components**:
1. **Metric Implementation**: BLEU, ROUGE, BERTScore
2. **Human Evaluation Interface**: HTML form for raters
3. **Analysis Dashboard**: Visualization of results
4. **Inter-rater Agreement**: Cohen's kappa calculation

**Code Framework**:
```python
class EvaluationPipeline:
    def __init__(self, model, test_set, num_raters=3):
        self.model = model
        self.test_set = test_set
        self.num_raters = num_raters
    
    def generate_outputs(self):
        # Generate predictions for all test examples
        pass
    
    def create_eval_interface(self):
        # Generate HTML for raters
        pass
    
    def compute_metrics(self):
        # Calculate BLEU, ROUGE, BERTScore
        pass
    
    def analyze_results(self):
        # Cohen's kappa, agreement statistics
        pass
    
    def create_report(self):
        # Generate evaluation report
        pass
```

**Deliverable**:
- Evaluation pipeline code
- HTML interface screenshot
- Final evaluation report with all metrics
- Recommendations for model improvement

---

## Advanced Projects (Week 5-6)

### Project 5: Custom Model Architecture Experiment
**Difficulty**: ⭐⭐⭐ (Advanced)
**Time**: 20-24 hours

**Objective**: Implement a novel architectural component and measure impact

**Possible Experiments**:
1. **Sparse Attention**: Implement strided or local attention patterns
2. **Adaptive Computation**: Route through different numbers of layers per example
3. **Product-Key Networks**: Learn input-dependent context representations
4. **Sliding Window Attention**: Reduce complexity for long sequences

**Requirements**:
1. Modify model architecture in `src/model/markgpt.py`
2. Maintain backward compatibility
3. Train on small dataset (1B tokens)
4. Compare against baseline on multiple metrics
5. Analyze computational tradeoffs

**Experimental Report Template**:
```markdown
# Experiment Report: [Your Innovation Name]

## Motivation
Why is this change valuable?

## Method
How does it work? Include math and code.

## Experimental Setup
- Training data: 1B tokens
- Baseline: MarkGPT-70M
- Modified: MarkGPT-70M + [Your Change]

## Results
- Perplexity comparison
- Training time comparison
- Memory usage comparison
- Qualitative observations

## Conclusion
Did it help? When might this be useful?

## Future Work
What could be improved?
```

**Deliverable**:
- Modified model code
- Training logs
- 5-10 page technical report
- Open source contribution (optional PR)

---

### Project 6: Banso Language Research
**Difficulty**: ⭐⭐⭐ (Advanced)
**Time**: Semester-long

**Objective**: Advance Banso language NLP through MarkGPT

**Research Directions**:
1. **Dataset Enhancement**: Collect and annotate natural Banso text
2. **Linguistic Analysis**: Study transfer from English to Banso
3. **Named Entity Recognition**: Build Banso NER dataset
4. **Machine Translation**: Train EN-Banso translation system
5. **Language Preservation**: Create educational resources

**Proposed Outcome**:
- 100K+ cleaned Banso text corpus
- Bilingual LLM (EN-Banso) checkpoint
- Academic paper on findings
- Open-source dataset (with community consent)

**Key Milestones**:
- Month 1: Data collection and annotation
- Month 2: Corpus analysis and preprocessing
- Month 3: Model training and evaluation
- Month 4: Paper writing and code release

---

## Evaluation & Grading

### Beginner Project Rubric (20 points)
- Code Quality (5 pts): Is it well-structured and documented?
- Results (7 pts): Are metrics computed correctly?
- Analysis (5 pts): Is the interpretation thoughtful?
- Presentation (3 pts): Is the report clear?

### Intermediate Project Rubric (30 points)
- Technical Depth (8 pts): Complexity and correctness
- Implementation (8 pts): Code quality and reproducibility
- Results (8 pts): Metrics and comparisons
- Report (6 pts): Clarity and insight

### Advanced Project Rubric (50 points)
- Novel Contribution (15 pts): Originality and significance
- Rigorous Evaluation (15 pts): Experimental design, statistics
- Code Quality (10 pts): Reproducibility and maintenance
- Academic Writing (10 pts): Clarity and contribution to field

---

## Resources & Support

### Per-Project Guidance
- **Discord**: Real-time Q&A with instructors
- **Office Hours**: Scheduled 1-on-1 mentoring
- **Code Reviews**: Peer review via GitHub PRs
- **Workshops**: Demo sessions on key topics

### Capstone Timeline
| Week | Beginner | Intermediate | Advanced |
|---|---|---|---|
| 1-2 | Project 1 or 2 | Project 3 or 4 | Planning & setup |
| 3-4 | Project 2 or 3 | Project 4 or 5 | Literature review |
| 5-6 | Exploration | Experimentation | Core development |
| 7-8 | Polish & document | Analysis & writing | Evaluation & paper |
| 9 | Presentation | Presentation | Presentation |

---

**Capstone Guide Version**: 1.0
**Last Updated**: 2024
