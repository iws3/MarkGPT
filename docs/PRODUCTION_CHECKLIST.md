# Production Checklist

Complete checklist for deploying MarkGPT to production.

## Pre-Deployment (2-3 weeks before)

### Code Quality
- [ ] All tests passing locally and in CI
- [ ] Type checking passes (mypy --strict)
- [ ] Linting passes (black, ruff)
- [ ] No deprecated APIs used
- [ ] Code reviewed by 2+ maintainers

### Documentation
- [ ] API documentation complete
- [ ] Deployment guide written
- [ ] Known limitations documented
- [ ] Security considerations listed
- [ ] Troubleshooting guide prepared

### Performance
- [ ] Benchmarks run and documented
- [ ] Memory footprint within spec
- [ ] Latency acceptable for use case
- [ ] Throughput meets requirements
- [ ] Load tested with 2x expected QPS

### Security
- [ ] No secrets in code (credentials removed)
- [ ] Input validation implemented
- [ ] Rate limiting configured
- [ ] Audit logging enabled
- [ ] Security scan completed

## Deployment (Day of)

### Infrastructure
- [ ] Servers provisioned and tested
- [ ] Load balancers configured
- [ ] Monitoring fully enabled
- [ ] Alerting configured
- [ ] Backup systems ready

### Model
- [ ] Model quantized/optimized for deployment
- [ ] Model checksums verified
- [ ] Model version tagged and documented
- [ ] Fallback model ready
- [ ] Rollback procedure tested

### Configuration
- [ ] Production config file reviewed
- [ ] Environment variables set correctly
- [ ] Database connections tested
- [ ] Cache warmed up
- [ ] Logging level appropriate

### Deployment
- [ ] Blue-green deployment setup
- [ ] Canary deployment prepared (5% traffic initially)
- [ ] Health checks passing
- [ ] Smoke tests successful
- [ ] Team on-call available

## Post-Deployment (First 48 hours)

### Monitoring
- [ ] Error rate normal (<0.1%)
- [ ] Latency acceptable (<100ms p95)
- [ ] Memory usage stable
- [ ] CPU usage below 80%
- [ ] No OOM errors

### User Facing
- [ ] Gradual traffic increase (ramp to 100% over 24h)
- [ ] User feedback actively monitored
- [ ] Support team on alert
- [ ] Chat/Discord actively monitored
- [ ] Status page maintained

### Incident Response
- [ ] Runbooks prepared
- [ ] Escalation path clear
- [ ] Rollback tested and ready
- [ ] Post-mortem template ready
- [ ] Incident severity levels defined

## Production Operations

### Daily
- [ ] Monitor dashboards reviewed
- [ ] No critical alerts
- [ ] Error logs examined
- [ ] Performance metrics normal

### Weekly
- [ ] Performance trends analyzed
- [ ] User feedback reviewed
- [ ] Model performance on holdout test set verified
- [ ] Security scans run

### Monthly
- [ ] Disaster recovery drill completed
- [ ] Load test simulating peak traffic
- [ ] Model retraining evaluation
- [ ] Capacity planning review

## Scaling

### When to Scale Out (Add more instances)
- CPU usage > 70%
- Memory usage > 80%
- Latency p95 > 150ms
- Error rate increasing

### How to Scale Out
```bash
# Kubernetes example
kubectl scale deployment markgpt-api --replicas=8
```

### When to Optimize
- Throughput not meeting SLO
- Cost per request exceeding budget
- Memory footprint larger than expected

## Rollback Procedure

### If Critical Issues Discovered

```bash
# 1. Stop traffic to new version
kubectl set image deployment/markgpt-api markgpt=markgpt:previous-tag

# 2. Verify rollback successful
curl http://api.markgpt.ai/health

# 3. Monitor error rate
# Should drop to 0 within 2-3 minutes

# 4. Post-mortem
# Schedule within 24 hours
```

### Rollback Decision Tree
```
Is error rate > 1%?
├─ Yes → Rollback immediately
├─ No → Check latency

Is latency p95 > 500ms?
├─ Yes → Rollback immediately
├─ No → Check for user complaints

User complaints about functionality?
├─ Yes → Likely rollback
├─ No → Monitor 1 hour then decide
```

## Security Checklist

- [ ] HTTPS/TLS enforced
- [ ] API authentication required
- [ ] Rate limiting enabled
- [ ] Input sanitization complete
- [ ] SQL injection prevention implemented
- [ ] CORS properly configured
- [ ] Secrets rotated monthly
- [ ] Audit logs retained (90 days)
- [ ] Vulnerability scans passing
- [ ] Penetration testing completed

## Performance SLOs

```
API SLO:
- Availability: 99.9% (43 min downtime/month)
- Latency p95: < 100ms
- Error rate: < 0.1%
- Throughput: > 1000 req/s
```

## Communication Plan

### Launch Announcement
- [ ] Announcement sent to users 24h before
- [ ] Status page updated
- [ ] Team onboarded and briefed
- [ ] FAQ prepared

### During Deployment  
- [ ] Status updates every 15 minutes
- [ ] Post in Discord/Slack
- [ ] Email ready to send if issues

### Post-Launch
- [ ] Launch retrospective (if issues)
- [ ] Success announcement
- [ ] Thank you message to team

---

**Checklist Version**: 1.0
**Last Updated**: 2024
**Maintained by**: MarkGPT DevOps
