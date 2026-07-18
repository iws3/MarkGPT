# Legal & Compliance Documentation

## Terms of Service & Model License

### Model License: CC-BY-4.0

**You are free to**:
- ✓ Share: Copy and redistribute the model
- ✓ Adapt: Remix, transform, and build upon the model
- ✓ Use commercially
- ✓ Use for any purpose

**Under the condition**:
- Attribution: Give appropriate credit, provide link to license

**You cannot**:
- ✗ Remove license terms from copies
- ✗ Modify license terms

---

### Code License: Apache 2.0

**Permissions**:
- Commercial use
- Modification  
- Distribution
- Private use

**Conditions**:
- License and copyright notice included
- State changes made

**Limitations**:
- Liability
- Warranty

---

## Data Handling & Privacy

### Training Data Provenance

```
English Data Sources:
├─ Wikipedia          (CC BY-SA 3.0, public domain)
├─ ArXiv              (CC BY 4.0, academic preprints)
├─ BookCorpus         (Custom license, educational use)
└─ Web Crawl (filtered)
   ├─ Removed PII before training
   ├─ Excluded adult/harmful content
   └─ Prioritized public domain/CC-licensed

Banso Data:
├─ Public domain texts (government, historical)
├─ Community-contributed (with explicit consent)
├─ Academic corpus (with permission)
└─ Synthetically augmented (from real data)
```

### User Data & Model Inference

**What we DON'T do**:
- Store user prompts (unless explicitly requested)
- Use inference data for model retraining
- Share data with third parties
- Link prompts to identity

**What you SHOULD do**:
- Inform users if you're using MarkGPT
- Don't process sensitive PII without consent
- Comply with local privacy regulations (GDPR, etc.)
- Implement appropriate access controls

---

## Responsible AI & Usage Guidelines

### Prohibited Uses

```
MarkGPT should NOT be used for:

1. MISINFORMATION
   - Creating false information
   - Impersonation
   - Conspiracy theories
   
2. HARM
   - Violent content
   - Self-harm instructions
   - Hate speech
   - Sexual abuse material
   
3. ILLEGAL ACTIVITY
   - Fraud
   - Drug trafficking
   - Terrorism
   - CSAM
   
4. DECEPTION
   - Automated spamming
   - Manipulation
   - Plagiarism
   - Unauthorized access
```

### Recommended Safety Practices

```
For Deployments:
✓ Content filtering on outputs
✓ Rate limiting to prevent abuse
✓ Logging for audit trails
✓ User authentication
✓ Abuse reporting mechanism

For Research:
✓ Document methodology
✓ Disclose limitations
✓ Include ethics review
✓ Plan for downstream harm
✓ Publish findings responsibly

For Education:
✓ Teach about AI limitations
✓ Discuss bias and fairness
✓ Emphasize human oversight
✓ Promote critical thinking
```

---

## Regulatory Compliance

### GDPR (EU)

**If you process EU citizen data**:
- [ ] Have a lawful basis (e.g., consent, contract)
- [ ] Data minimization (collect only what's needed)
- [ ] User rights (access, erasure, portability)
- [ ] Data protection (encryption, access controls)
- [ ] Privacy impact assessment (for high-risk)
- [ ] Data processing agreement (third-parties)

**Implementation**:
```python
# Example: Implement right to deletion
def delete_user_data(user_id):
    """GDPR delete user's personal data."""
    logging.info(f"Deleting data for GDPR subject access request: {user_id}")
    # Delete from database, cache, logs
    db.delete_user_records(user_id)
    cache.delete_user_data(user_id)
    # Audit trail maintained
```

### CCPA (California)

**If you process California resident data**:
- [ ] Disclose data collection practices
- [ ] Let users delete personal information
- [ ] Don't discriminate for exercising rights
- [ ] Validate deletion requests

### China's CAC Regulations

**If targeting Chinese users**:
- [ ] Content must comply with regulations
- [ ] Data localization requirements
- [ ] Government access capabilities
- [ ] Prior security assessment

---

## Ethical Considerations

### Bias & Fairness

**What we've tested**:
- [ ] Gender bias in completions (via embeddings)
- [ ] Racial bias in named entities
- [ ] Cultural stereotypes
- [ ] Professional stereotyping

**What we found**:
- Small models (70M-200M) show higher bias than large
- Banso model less biased (smaller training data, more recent)
- Context matters (prompt heavily influences outputs)

**What you should do**:
- Audit outputs for your use case
- Test on diverse demographics
- Implement human review for high-stakes
- Document observed biases

### Environmental Impact

**Training (one-time)**:
- 60 hours × 8 GPUs × 320W = ~150 MWh
- Equivalent to: ~10 tons CO2 equivalent

**Inference (ongoing)**:
- Per 1M tokens: ~0.0005 MWh
- Optimizing: Quantization saves 2-3x energy

**Our commitment**:
- [ ] Carbon-aware training (off-peak, renewable energy)
- [ ] Efficient models (small enough for edge)
- [ ] Open-sourcing to reduce duplicated research

---

## Model Limitations & Disclosure

### Known Limitations

```
Accuracy:
- Hallucinations: ~5-10% of outputs
- Factual errors common for obscure topics
- No access to real-time information

Language:
- English training dominant, Banso may be less fluent
- Transliteration errors possible
- Code generation weaker than specialized models
- Math reasoning limited

Bias & Fairness:
- Reflects biases in training data
- Performs worse on underrepresented groups
- Cultural specificity limited

Capabilities:
- No image/audio processing
- Limited reasoning over long documents
- No persistent memory between conversations
```

### Recommended Disclosure Language

For your service terms:

> "This service uses MarkGPT, an AI language model that may:
> - Generate inaccurate information
> - Perpetuate biases from training data
> - Have limited understanding of rare topics
> 
> Do not rely on outputs for high-stakes decisions.
> Always verify important information with authoritative sources."

---

## Incident Response

### If Harmful Content is Generated

**Immediate**:
1. Log incident (timestamp, input, output)
2. Disable that version if necessary
3. Notify affected users if applicable
4. Preserve evidence

**Investigation**:
1. Identify root cause
2. Assess scope (how many users affected?)
3. Check logs for patterns
4. Determine if training data or inference issue

**Response**:
1. Patch if code issue
2. Consider content filtering
3. Update documentation
4. Communicate transparency

---

## Annual Audit Checklist

- [ ] Data governance review
- [ ] Privacy compliance audit
- [ ] Ethics review of use cases
- [ ] Bias assessment on new data
- [ ] Security penetration testing
- [ ] User feedback analysis
- [ ] Regulatory update review
- [ ] Environmental impact calculation
- [ ] Incident review for patterns
- [ ] Model update evaluation

---

**Legal & Compliance v1.0**
**Last Updated**: 2024
**Status**: For reference, consult legal team for actual compliance
