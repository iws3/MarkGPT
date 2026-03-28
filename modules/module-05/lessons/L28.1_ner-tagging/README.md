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

