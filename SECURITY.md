# 🔒 Security Policy

## Reporting Security Vulnerabilities

MarkGPT takes security seriously. We appreciate your efforts and responsible disclosure of any security issues.

### ⚠️ Do NOT Create Public Issues for Security Vulnerabilities

If you discover a security vulnerability, **please do NOT open a public GitHub issue**. Instead, please disclose it responsibly:

## 🔐 How to Report a Vulnerability

**Email:** iwstechnical@gmail.com

**Subject:** [SECURITY] Brief description of vulnerability

**In your report, please include:**
1. **Description** - Clear description of the vulnerability
2. **Location** - Which file(s), module(s), or component(s) are affected
3. **Severity** - Impact assessment (Critical, High, Medium, Low)
4. **Reproduction** - Steps to reproduce or proof-of-concept code
5. **Impact** - What could an attacker do with this vulnerability?
6. **Suggested Fix** - If you have a fix, please describe it
7. **Your Details** - How you'd like us to credit you (optional)

## 📋 Vulnerability Types We Care About

**Examples of security issues we take seriously:**

- **Data Privacy:** Code that exposes personal information or training data
- **Dependency Vulnerabilities:** Known vulnerable versions of libraries
- **Authentication/Authorization:** Bypass mechanisms or access control issues
- **Injection Attacks:** Code injection, prompt injection in LLM contexts
- **Cryptography:** Weak encryption, misuse of cryptographic functions
- **Model Safety:** Model outputs that could cause harm
- **Infrastructure:** Cloud credential exposure, misconfigured resources
- **Documentation:** Security-related documentation errors that could mislead users

## 🛡️ What We Will Do

When we receive a vulnerability report:

1. **Acknowledge** your report within 48 hours
2. **Assess** the vulnerability severity
3. **Develop** a fix or mitigation strategy
4. **Test** the fix thoroughly
5. **Release** a security patch or updated version
6. **Credit** you publicly (unless you prefer anonymity)

## 📅 Response Timeline

| Severity | Initial Response | Target Fix | Public Disclosure |
|----------|------------------|------------|-------------------|
| Critical | 24 hours | 7 days | 30 days after fix |
| High | 48 hours | 14 days | 60 days after fix |
| Medium | 1 week | 30 days | 90 days after fix |
| Low | 2 weeks | 60 days | 120 days after fix |

*Note: These timelines are targets. Actual times depend on fix complexity.*

## ✅ Scope

**Security issues in scope:**
- Main MarkGPT codebase (src/, modules/)
- Project dependencies and imports
- Documentation related to security/data handling
- Configuration and deployment guidance
- GitHub infrastructure and workflows

**Out of scope:**
- Third-party services and dependencies (report to upstream maintainers)
- User-created vulnerabilities in forked versions
- Denial-of-service attacks on infrastructure
- Social engineering or phishing attacks

## 🎓 Security Best Practices for Contributors

When contributing, please keep security in mind:

### Code Security
- Never commit secrets (API keys, tokens, credentials)
- Follow the principle of least privilege
- Validate and sanitize user inputs
- Use `.gitignore` to exclude sensitive files
- Check dependencies for known vulnerabilities: `pip audit`

### Data Handling
- Be careful with personal data, especially:
  - Student learning data
  - Banso language speaker information
  - Any PII (Personally Identifiable Information)
- Follow data privacy regulations (GDPR, etc.)
- Document data sources and usage restrictions
- Don't include real personal data in examples

### Dependencies
- Keep dependencies up to date
- Review new dependencies for security
- Check the National Vulnerability Database (NVD)
- Use `pip audit` or `safety` to check for known vulnerabilities

### Documentation
- Document security assumptions
- Explain data flows and where data is stored
- Note any limitations or risks
- Don't document potential attack vectors

## 🔍 Security Scanning

Our project uses automated security scanning:
- GitHub Dependabot for dependency vulnerabilities
- SAST (Static Application Security Testing) tools
- Regular code audits

All security scan results are reviewed by maintainers.

## 📚 Security Resources for Contributors

- [OWASP Top 10](https://owasp.org/www-project-top-ten/) - Common web security issues
- [CWE Top 25](https://cwe.mitre.org/top25/) - Most dangerous software weaknesses  
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/)
- [Banso Language Data Privacy Considerations](../docs/BANSO_LINGUISTICS.md)
- [Data Collection Guide](../docs/DATA_COLLECTION_GUIDE.md)

## 🙏 Thank You

We're grateful for your dedication to keeping MarkGPT secure. Security researchers and contributors who help us improve security are invaluable to our community.

---

**Last Updated:** This policy will be reviewed and updated as the project evolves.

**Questions?** Email iwstechnical@gmail.com
