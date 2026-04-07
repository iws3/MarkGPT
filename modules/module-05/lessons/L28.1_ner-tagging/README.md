# Named Entity Recognition and Tagging
## Comprehensive Learning Guide

## NER Task Definition

NER identifies named entities in text.

Entity types include person, organization, location.

Entity boundaries must be precise.

Multi-word entities require coordinated predictions.

Nested entities complicate label structure.

Entity context provides classification signal.

Domain variation affects entity definitions.

## Sequence Labeling

BIO tagging represents entity boundaries.

BIOES tagging distinguishes entity endings.

IOBES tagging further refines boundaries.

CRF decoding ensures valid label sequences.

HMM captures label dependencies.

LSTM-CRF combines neural and structured learning.

Beam search finds high-probability sequences.

## NER Applications

Information extraction builds knowledge graphs.

Question answering locates relevant entities.

Coreference resolution links entity mentions.

Entity linking maps entities to knowledge base.

Semantic role labeling identifies relationships.

Relation extraction connects entity pairs.

Event extraction identifies complex structures.

## Advanced NER Methods

Distant supervision generates training data automatically.

Transfer learning improves low-resource NER.

## NER Tagging Schemes

### IOB1

I-tag for continuation
Simpler than BIO
B only when adjacent to same
Less common
Historic

### IOB2 (BIO)

B: Begin tag
I: Inside tag
O: Outside
Most common
Standard scheme

### IOBES

B, I, O, E (end), S (single)
Most explicit
Distinguishes single vs multi
Better for some tasks
Slightly higher accuracy

### Nested NER

Overlapping entities
"United States" inside "United"
Rare, harder
Different schemes
Research area

### Multi-token Entities

Names span multiple tokens
"John Smith" = 2 tokens
Sequential dependency
CRF handles well
Important for real data

## NER System Architecture

### Feature Engineering

Lexical: word, case, digits
Syntactic: POS tags
Semantic: word embeddings
External: gazetteers, wikipeda
Combined into feature vector

### Gazetteer Resources

Lists of known entities
Person names, locations, orgs
Wikipedia dumps, YAGO, DBpedia
Signals but incomplete
Noisy annotations

