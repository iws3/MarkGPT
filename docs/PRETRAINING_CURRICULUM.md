# Pre-Training Curriculum & Data Strategy

## Curriculum Learning Overview

### Motivation

Training LLMs on raw data doesn't work optimally. Better to structure training:

1. **Easy → Hard progression**
2. **General → Specific**
3. **Clean → Noisy data**

### Why Curriculum?

```
Without curriculum:
  Step 1: Random initialization + raw data
  → Model learns common patterns quickly
  → Then struggles on rare/hard patterns
  → Final model has unbalanced knowledge

With curriculum:
  Step 1: Clean, common data
  → Model learns strong foundation
  Step 2: Gradually introduce harder examples
  → Model refines understanding
  → Final model robust and balanced
```

---

## MarkGPT Pre-Training Curriculum

### Phase 1: English Foundation (0-20% of training)

**Goal**: Learn English strongly before multilingual.

```python
phase_1_config = {
    "name": "English Foundation",
    "data_sources": [
        "wikipedia_en",      # 50% - diverse, clean
        "cc100_en",          # 30% - web data (noisy but large)
        "books_en",          # 20% - coherent long-form
    ],
    "duration_steps": 40000,  # Out of total 200K
    "target_perplexity": 20,  # Initial target
}

# Data ratio
data_distribution = {
    "wikipedia_en": 0.5,
    "cc100_en": 0.3,
    "books_en": 0.2,
}
```

### Phase 2: Banso Introduction (20-50%)

**Goal**: Learn Banso representations while retaining English.

```python
phase_2_config = {
    "name": "Banso Introduction",
    "data_sources": [
        "wikipedia_en": 0.3,   # Reduce English (already learned)
        "cc100_en": 0.1,
        "banso_corpus": 0.4,   # Primary focus: Banso
        "parallel_en_banso": 0.2,  # Bilingual data crucial
    ],
    "duration_steps": 60000,
    "curriculum_schedule": "linear",  # Gradually increase Banso fraction
}

def phase_2_schedule(step):
    """Gradually increase Banso fraction."""
    # Step 40K (start of phase 2) → English 50%
    # Step 100K (end of phase 2) → English 30%
    progress = (step - 40000) / 60000  # 0 → 1
    english_fraction = 0.5 - 0.2 * progress
    banso_fraction = 1 - english_fraction
    return english_fraction, banso_fraction
```

### Phase 3: Diverse & Challenging (50-100%)

**Goal**: Handle diverse language phenomena.

```python
phase_3_config = {
    "name": "Diverse & Challenging",
    "data_sources": {
        "wikipedia_mixed": 0.2,    # Both languages
        "news_mixed": 0.2,         # Current events
        "technical_docs": 0.1,     # Code, formulas
        "social_media": 0.2,       # Informal
        "parallel_mine": 0.15,     # Mined parallel data
        "synthetic": 0.05,         # Augmented
    },
    "duration_steps": 100000,
    "curriculum_strategy": "epoch_based",  # Shuffle data each epoch
}
```

---

## Data Strategy: English-Banso Balance

### The Challenge

```
English: ~200B tokens available
Banso: ~500M tokens available (if lucky)
Ratio: 400:1

Naive mixing:
  50% English + 50% Banso
  → 200B * 0.5 + 500M * 0.5
  → 100B + 250M ≈ 100.25B tokens
  → Mostly English, wasted potential

Better curriculum:
  Phase 1: 100% English (40K steps) → Learn English
  Phase 2: Gradually shift to bilingual
  Phase 3: 50-50 but sample Banso more  → Balance representations
```

### Oversampling Strategy

```python
def create_curriculum_dataloader(phase, batch_size=32):
    """Load data according to curriculum."""
    
    if phase == 1:
        # English only
        return DataLoader(english_dataset, batch_size=batch_size)
    
    elif phase == 2:
        # Mix English and Banso
        english_loader = DataLoader(
            english_dataset,
            batch_size=int(batch_size * 0.3),
            shuffle=True
        )
        banso_loader = DataLoader(
            banso_dataset,
            batch_size=int(batch_size * 0.7),
            shuffle=True,
            sampler=WeightedRandomSampler(
                weights=banso_frequency_weights,
                num_samples=len(banso_dataset),
                replacement=True  # Oversample rare Banso examples
            )
        )
        # Combine loaders
        return CombinedLoader(english_loader, banso_loader)
    
    elif phase == 3:
        # Balanced bilingual
        mixed_loader = DataLoader(
            ConcatDataset([english_dataset, banso_dataset]),
            batch_size=batch_size,
            sampler=WeightedRandomSampler(
                weights=combined_weights,  # Equal effective weight
                num_samples=total_samples
            ),
            shuffle=True
        )
        return mixed_loader
```

---

## Stopping Points & Evaluation

### Checkpoint Strategy

```python
class CurriculumCheckpoint:
    def __init__(self, config):
        self.phases = config.phases
        self.checkpoint_interval = config.checkpoint_interval
    
    def should_checkpoint(self, step):
        """Determine if this step needs a checkpoint."""
        
        # Always checkpoint at phase boundaries
        for phase in self.phases:
            if step == phase['start_step'] or step == phase['end_step']:
                return True, f"phase_{phase['name']}"
        
        # Regular checkpoints
        if step % self.checkpoint_interval == 0:
            return True, f"step_{step}"
        
        return False, None

# Phase boundaries
phase_boundaries = [
    {"name": "english_only", "start": 0, "end": 40000},
    {"name": "banso_intro", "start": 40000, "end": 100000},
    {"name": "diverse", "start": 100000, "end": 200000},
]
```

### Evaluation Metrics by Phase

```python
def evaluate_curriculum_phase(model, phase_name):
    """Evaluate model on phase-specific metrics."""
    
    if phase_name == "english_only":
        # Focus on English performance
        metrics = {
            "english_perplexity": eval_english(model),
            "english_downstream": eval_downstream_english(model),
        }
    
    elif phase_name == "banso_intro":
        # Track language entanglement
        metrics = {
            "english_ppl": eval_english(model),
            "banso_ppl": eval_banso(model),
            "english_retention": compare_to_phase1(model),  # Catastrophic forgetting?
            "interference": measure_language_interference(model),
        }
    
    elif phase_name == "diverse":
        # Comprehensive evaluation
        metrics = {
            "english_ppl": eval_english(model),
            "banso_ppl": eval_banso(model),
            "translation_bleu": eval_translation(model),
            "code_generation": eval_code(model),
            "reasoning": eval_downstream(model),
        }
    
    return metrics
```

### Catastrophic Forgetting

Monitor if English performance drops during Banso introduction:

```python
def detect_catastrophic_forgetting(metrics_history):
    """Flag if model forgets English."""
    
    english_ppl = [m['english_ppl'] for m in metrics_history]
    
    # Check for sudden increase
    recent_avg = np.mean(english_ppl[-10:])
    early_avg = np.mean(english_ppl[:10])
    
    if recent_avg > 1.5 * early_avg:
        print("⚠️ WARNING: Catastrophic forgetting detected!")
        print(f"   English PPL: {early_avg:.1f} → {recent_avg:.1f}")
        return True
    
    return False

# During training
for checkpoint in checkpoints:
    metrics = evaluate_model(model, checkpoint)
    metrics_history.append(metrics)
    
    if detect_catastrophic_forgetting(metrics_history):
        print("   → Reduce Banso data fraction")
        print("   → Increase English mixture")
```

---

## Co-Training Strategy (Alternative)

Instead of strict phases, train on both languages simultaneously:

```python
class MultilingualCo-Trainer:
    def __init__(self, english_dataset, banso_dataset):
        self.english_loader = DataLoader(english_dataset, batch_size=32)
        self.banso_loader = DataLoader(banso_dataset, batch_size=32)
    
    def get_batch(self, step):
        """Alternate languages or blend."""
        
        if step % 4 == 0:  # Every 4th step: English only
            return next(iter(self.english_loader))
        elif step % 4 == 1:  # Every 4th+1: Banso only
            return next(iter(self.banso_loader))
        else:  # Odd steps: blend
            en_batch = next(iter(self.english_loader))
            bn_batch = next(iter(self.banso_loader))
            return combine_batches(en_batch, bn_batch)
```

---

## Data Augmentation in Curriculum

### Phase-Specific Augmentations

```python
class CurriculumAugmentations:
    def __init__(self, phase):
        self.phase = phase
    
    def augment(self, text):
        if self.phase == 1:
            # Phase 1: Minimal augmentation (trust English data)
            return text
        
        elif self.phase == 2:
            # Phase 2: Back-translation for Banso
            if random.random() < 0.1:
                # Banso → English → Banso
                en_trans = translate(text, 'banso', 'en')
                return translate(en_trans, 'en', 'banso')
            return text
        
        elif self.phase == 3:
            # Phase 3: Aggressive augmentation
            augmentations = [
                lambda t: add_typos(t, prob=0.05),
                lambda t: paraphrase(t),
                lambda t: translate_back(t),  # Round-trip
            ]
            aug = random.choice(augmentations)
            return aug(text) if random.random() < 0.2 else text
    
    return text
```

---

## Pre-Training Timeline (MarkGPT)

```
Step 0         Step 40K      Step 100K     Step 200K
|              |              |              |
Phase 1        Phase 2        Phase 3        Complete
English →      Bilingual      Diverse →      Release
Only           Intro          Hardened       (v1.0)
      ↓             ↓              ↓
     40K           60K            100K
    steps        steps            steps
```

### Key Milestones

| Step | Target | Action |
|------|--------|--------|
| 0 | Start | Initialize, log baseline |
| 10K | 30 PPL | Verify training |
| 40K* | 15 PPL | Phase 1 complete, checkpoint |
| 60K | 20 PPL | Banso intro half-way |
| 100K* | 18 PPL | Phase 2 complete, comprehensive eval |
| 150K | 17 PPL | Mid-phase-3 review |
| 200K* | 16 PPL | **Training complete**, tag for release |

*Starred: Major checkpoints for multi-GPU distribution

---

**Pre-Training Curriculum v1.0**
**Last Updated**: 2024
