# TF-IDF and Term Weighting
## Comprehensive Learning Guide

## TF-IDF Calculation

TF-IDF weights term importance in documents.

Term frequency counts word occurrences.

Document frequency counts documents containing term.

Inverse document frequency penalizes common words.

TF-IDF product balances local and global frequency.

Logarithmic scaling compresses large frequencies.

Normalization makes vectors comparable across documents.

## Weighting Schemes

Raw counts preserve frequency information.

Logarithmic frequencies reduce range.

Augmented frequency controls TF influence.

Double normalization compares documents fairly.

Pivoted normalization corrects length bias.

BM25 extends TF-IDF with saturation.

Domain-specific schemes address task characteristics.

## Information Retrieval

TF-IDF improves document ranking.

Cosine similarity measures document relevance.

Query vectors use same weighting as documents.

Sparse representations efficiently store vectors.

Dimensionality reduction preserves important information.

Semantic extensions improve retrieval.

Multi-field documents separate component weights.

## Advanced IR Techniques

Semantic search uses embeddings for retrieval.

## TF-IDF Variants

### Log-TF

TF(t,d) = 1 + log(count)
Sublinear scaling
Dampens frequency effect
Often better than raw
Standard choice

### Probabilistic IDF

IDF(t) = log((N - count) / count)
Probabilistic interpretation
Different from standard
Sometimes useful
Less common

