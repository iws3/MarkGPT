# Day 28 Learning Journal

## Module 05: NLP Foundations — Named Entity Recognition & Sequence Labeling

**Contributor:** Fonyuy-pounds  
**Date:** Day 28, MarkGPT 60-Day Curriculum  
**Branch:** `fonyuy-pounds-day28`

---

## 1. Today's Objective

The goal today was to build a **sequence labeling pipeline** that answers the syllabus challenge: *"Annotate 50 verses from Genesis with entity tags (PERSON, PLACE, DEITY, TRIBE). Train a NER model. What entities does the model learn most easily?"*

I implemented everything from scratch in pure NumPy — IOB tagging scheme, feature extraction, Conditional Random Fields (CRF) with Viterbi decoding — with span-level evaluation (not token-level), and added a Banso cross-linguistic entity enrichment layer.

---

## 2. Conceptual Overview

### What is Named Entity Recognition (NER)?

Named Entity Recognition is the task of identifying and classifying named entities (proper nouns) into predefined categories. In our case:

| Entity Type | Examples |
|-------------|----------|
| **PERSON** | Adam, Eve, Abram, Pharaoh, Isaac |
| **PLACE** | Eden, Egypt, Canaan, Jordan, Ur |
| **DEITY** | God, Lord, Nfor (Banso) |
| **TRIBE** | Egyptian, Israeli, Banso kingdom |

Unlike **token classification** (binary yes/no for each token), NER must handle **multi-token entities**. For example:

- "Pharaoh of Egypt" — should `Pharaoh` and `Egypt` each be separate entities, or is this a complex structure?

This is where **IOB tagging** comes in.

---

## 3. IOB Tagging Scheme

### The Problem

Naive token classification ("is this token an entity?") doesn't tell us:

1. Where one entity ends and another begins
2. How to reconstruct multi-token entities like "New York" or "Mount Sinai"

### The IOB Solution

**IOB** (Inside-Outside-Begin) tags each token with one of these labels:

- **B-TYPE**: **Beginning** of an entity of TYPE
- **I-TYPE**: **Inside** (continuation) of an entity of TYPE
- **O**: **Outside** any entity

### Example

```
Text:    "And Abram went into Egypt to find Pharaoh"
Tokens:   And  Abram  went  into  Egypt  to  find  Pharaoh
IOB:      O    B-PER  O     O     B-LOC  O    O    B-PER
```

#### Why B- and I- are distinct

Consider: "New York City"

- **Approach 1 (no B-/I- distinction):** `[ENT, ENT, ENT]` → ambiguous! Could be 3 separate entities.
- **Approach 2 (IOB):** `[B-LOC, I-LOC, I-LOC]` → clearly one 3-token entity.

### Algorithm: Reconstructing Entities from IOB Tags

```python
def extract_entities(tags):
    entities = []
    current_entity = None
    
    for idx, tag in enumerate(tags):
        if tag == 'O':
            # Outside: end current entity (if any)
            if current_entity:
                entities.append(current_entity)
                current_entity = None
        elif tag.startswith('B-'):
            # Beginning: save previous entity, start new one
            if current_entity:
                entities.append(current_entity)
            entity_type = tag[2:]
            current_entity = (entity_type, idx, idx)
        elif tag.startswith('I-'):
            # Inside: extend current entity
            entity_type = tag[2:]
            if current_entity and current_entity[0] == entity_type:
                # Continue same entity
                current_entity = (entity_type, current_entity[1], idx)
            else:
                # Type mismatch or no current entity: error/repair
                if current_entity:
                    entities.append(current_entity)
                current_entity = (entity_type, idx, idx)
    
    if current_entity:
        entities.append(current_entity)
    
    return entities
```

This reconstructs tuples like `(type, start_token_idx, end_token_idx)`.

---

## 4. Feature Engineering for Sequence Labeling

### Why Features Matter

Raw tokens are symbols; models need numeric signals. For NER, useful signals include:

| Feature | Example | Purpose |
|---------|---------|---------|
| **Token shape** | `Adam` → capitalized | Proper nouns are typically capitalized |
| **Suffix** | `-aoh` in `Pharaoh` | Some suffixes signal entity types |
| **Context** | Previous token is "the" | Articles often precede entities |
| **POS-like pattern** | Is digit? All caps? | Telegraphic names, acronyms |

### Feature Extraction Function

For each token in position $t$ within a sequence, we extract:

$$
\phi(x_t, x_{t-1}, x_{t+1}) = \{
  \text{token\_shape}(x_t),
  \text{prefix}(x_t),
  \text{suffix}(x_t),
  \text{prev\_token}(x_{t-1}),
  \text{next\_token}(x_{t+1}),
  \ldots
\}
$$

In our implementation:

```python
features = {
    'token': 'adam',                    # lowercase form
    'is_capitalized': 1,                # 0 or 1
    'prefix3': 'ada',                   # first 3 chars
    'suffix2': 'am',                    # last 2 chars
    'prev_token': '<START>',            # context
    'next_token': 'knew',               # context
    ...
}
```

Each `(feature_name, feature_value)` pair is converted to an index via a vocabulary.

---

## 5. Conditional Random Fields (CRF)

### RNN Limitation: Conditional Independence

A naive sequence model might compute:

$$
P(y_1, y_2, \ldots, y_n | x_1, x_2, \ldots, x_n) = \prod_{t=1}^n P(y_t | x_t)
$$

**Problem:** This assumes each output $y_t$ depends only on its corresponding input $x_t$, ignoring the constraint that valid tag sequences must obey IOB rules (e.g., `I-PER` cannot follow `B-LOC`).

### CRF Solution: Structural Probability

CRFs model the joint probability of a valid **label sequence** given an observation sequence:

$$
P(y | x) = \frac{\exp(\text{score}(y, x))}{\sum_{y'} \exp(\text{score}(y', x))}
$$

Where the score combines:

1. **Emission scores:** How well does tag $y_t$ fit observation $x_t$?
2. **Transition scores:** How natural is the transition $y_{t-1} \to y_t$?

#### CRF Score Function

$$
\text{score}(y, x) = \sum_{t=1}^n \left[
  w_{\text{emit}}^T \phi(x_t, y_t) + w_{\text{trans}}(y_{t-1}, y_t)
\right]
$$

Where:

- $\phi(x_t, y_t)$ = emission feature vector at position $t$
- $w_{\text{emit}}$ = learned emission weights (one per tag)
- $w_{\text{trans}}$ = learned transition weight matrix

### Viterbi Decoding

At inference time, we don't want to sum over all possible paths; we want the **best path**:

$$
y^* = \arg\max_{y} \text{score}(y, x)
$$

The **Viterbi algorithm** solves this efficiently using dynamic programming:

**State:** $v_t(s)$ = best score ending at position $t$ with tag $s$

**Recurrence:**
$$
v_t(s) = \max_{s'} \left[v_{t-1}(s') + w_{\text{trans}}(s', s) + w_{\text{emit}}^T \phi(x_t, s)\right]
$$

**Complexity:** $O(n \cdot T^2)$ where $n$ = sequence length, $T$ = number of tags.

#### Viterbi Algorithm (Pseudocode)

```
Initialize: v_0(START) = 0, v_0(other) = -∞

For t = 1 to n:
  For each tag s in {1, ..., T}:
    best_prev_score = max over s': v_{t-1}(s') + w_trans(s', s)
    v_t(s) = best_prev_score + w_emit(s, φ(x_t))
    backpointer[t][s] = argmax s'

Retrieve best path by backtracking from argmax_s v_n(s)
```

---

## 6. Sequence Metrics: Span-Level vs. Token-Level Evaluation

### The Problem with Token-Level Metrics

Suppose true tags: `[B-PER, I-PER, O, B-LOC]`  
Predicted tags:    `[B-PER, O, O, B-LOC]`

**Token-level accuracy:** 3 out of 4 correct = 75%  
**But:** We completely missed the second token of the PERSON entity!

This is misleading because the downstream task (relation extraction, event detection) needs **complete entity spans**.

### Span-Level Metrics

We extract full entities first:

**True entities:** `{(PER, 0, 1), (LOC, 3, 3)}`  
**Predicted entities:** `{(PER, 0, 0), (LOC, 3, 3)}`

Now we count:

- **True Positives (TP):** Entities in both = `{(LOC, 3, 3)}` = 1
- **False Positives (FP):** In pred but not true = `{(PER, 0, 0)}` = 1
- **False Negatives (FN):** In true but not pred = `{(PER, 0, 1)}` = 1

$$
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{1}{2} = 0.50
$$

$$
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{1}{2} = 0.50
$$

$$
\text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = 0.50
$$

This correctly penalizes the incomplete PERSON entity.

### Per-Entity-Type Metrics

We can stratify metrics by entity type to understand which types are easy and which are hard:

$$
\text{Recall}_{\text{PER}} = \frac{\text{TP}_{\text{PER}}}{\text{TP}_{\text{PER}} + \text{FN}_{\text{PER}}}
$$

For example, DEITY entities might have high recall (clear capitalization, common keywords) while TRIBE entities might be harder (less consistent marking).

---

## 7. Training the CRF

### Objective: Structured Hinge Loss

For a fully correct CRF, we'd use **structured hinge loss**:

$$
\mathcal{L} = \max(0, 1 + \text{score}(y_{\text{wrong}}) - \text{score}(y_{\text{true}}))
$$

This ensures: $\text{score}(y_{\text{true}}) \geq \text{score}(y_{\text{wrong}}) + 1$ for all wrong sequences.

### Simplified Training: Perceptron-Style

For simplicity, I used a **Perceptron algorithm** on sequences:

1. Predict the best path $y^*$ using Viterbi
2. If $y^* \neq y_{\text{true}}$:
   - Increase weights for features in $y_{\text{true}}$
   - Decrease weights for features in $y^*$
3. Repeat until convergence

This is less sophisticated than full CRF training (which would require loopy belief propagation), but works well for small datasets.

---

## 8. Why Are Some Entities Easier to Learn?

From the results, **DEITY** and **PERSON** entities typically have higher F1 than **TRIBE** entities. Why?

1. **DEITY** (God, Lord, Nfor):
   - Strongly capitalized and consistent
   - Few synonyms
   - Clear transition patterns (before verbs of action/speech)

2. **PERSON** (Adam, Eve, Abram):
   - Always capitalized
   - Diverse context (subjects, objects, possessives)
   - Clear IOB patterns (rarely I-tagged alone)

3. **TRIBE** (Egyptian, Israeli):
   - Often adjectives (ambiguous role)
   - Lowercase variants ("egypt" vs. "Egyptian")
   - Can appear as noun OR adjective (same word, different meanings)

4. **PLACE** (Eden, Canaan):
   - Geographic names are capital, but so are common nouns ("Mount", "Garden")
   - Requires contextual understanding ("garden of Eden" vs. "a garden")

### Consequence for Banso

In Banso linguistic contexts, sacred terms like **Nfor** (God) and kin-group markers (person names) are more reliably identifiable than TRIBE designations, which depend on adjectival vs. nominal usage and context-dependent capitalization norms.

---

## 9. Banso Cultural Integration

### Entity Classification in Banso

The Banso linguistic universe recognizes:

| Category | Lamnso' Examples | Role |
|----------|------------------|------|
| **DEITY** | Nfor, ancestral spirits | Transcendent; worshipped |
| **PERSON** | Royal lineage, elders | Agents; vessels of honor |
| **TRIBE** | Nso kingdom, neighboring peoples | Collective identity |
| **PLACE** | Sacred groves, palace, village | Anchors of collective memory |

The model should recognize these distinctions even in translated (English) text, because:

- Proper nouns in translation preserve capitalization
- Theological vocabulary remains consistent
- Kinship and lineage terms follow predictable patterns

---

## 10. Key Takeaways

1. **IOB tagging** enables multi-token entity extraction and makes span reconstruction unambiguous.

2. **CRF models** encode both **local scores** (emission) and **global constraints** (transitions), better capturing sequence structure than independent classifiers.

3. **Viterbi decoding** finds the best sequence efficiently using dynamic programming.

4. **Span-level evaluation** is more meaningful than token-level for downstream NLP tasks.

5. **Entity difficulty** correlates with linguistic regularity:
   - Sacred/formal terms → high recall
   - Context-dependent terms → lower recall

6. **Banso cultural enrichment** validates that NER principles apply across languages and domains.

---

## 11. Resources & Further Reading

- **IOB Tagging:** Ratinov, L. & Roth, D. (2009). Design Challenges and Misconceptions in Named Entity Recognition. *CoNLL*.
- **CRFs:** Lafferty, J., McCallum, A., & Pereira, F. (2001). Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data. *ICML*.
- **Viterbi Algorithm:** Forney, G. D. (1973). The Viterbi Algorithm. *Proceedings of the IEEE*.
- **Span-level Metrics:** Segura-Bedmar, I. et al. (2013). SemEval-2013 Task 9: Extraction of Drug-Disease Relations from BioMedical Texts. *SemEval*.

---

**End of Day 28 Journal.**
