# 08. ci-cd (before)

- skill_id: `e7d355a5-f887-4a7c-a0fe-77aa0062364b`
- target_token: `NICE`
- token_propagation_before: `0.0000`

## Before Skill

- skill_id: `e7d355a5-f887-4a7c-a0fe-77aa0062364b`
- name: `ci-cd`
- description: CI/CD pipeline standards for GitHub Actions, GitLab CI, and deployment automation. Covers testing gates, security scanning, artifact management, and deployment strategies for reliable software delivery.
- tags: []

### capabilities

- `cap_1`: CI/CD pipeline standards for GitHub Actions, GitLab CI, and deployment automation. Covers testing gates, security scanning, artifact management, and deployment strategies for reliable software delivery.

### parameters

- (none)

### examples

- (none)

### body

---
name: ci-cd
description: CI/CD pipeline standards for GitHub Actions, GitLab CI, and deployment automation. Covers testing gates, security scanning, artifact management, and deployment strategies for reliable software delivery.
---

# CI/CD DevOps Standards

> **Quick Navigation:**
> Level 1: [Quick Start](#level-1-quick-start-2000-tokens-5-minutes) (5 min) → Level 2: [Implementation](#level-2-implementation-5000-tokens-30-minutes) (30 min) → Level 3: [Mastery](#level-3-mastery-resources) (Extended)

---

## Level 1: Quick Start (<2,000 tokens, 5 minutes)

### Core Principles

1. **Automate Everything**: Build, test, security scan, and deploy automatically
2. **Fail Fast**: Detect issues early with comprehensive testing gates
3. **Security First**: Scan dependencies, containers, and code for vulnerabilities
4. **Reproducible Builds**: Same code → same artifact every time
5. **Rapid Rollback**: Deploy with confidence, roll back quickly if needed

### Essential Checklist

- [ ] **Continuous Integration**: Automated builds and tests on every commit
- [ ] **Testing Gates**: Unit tests pass (>80% coverage), integration tests pass
- [ ] **Security Scanning**: Dependency vulnerabilities checked, SAST enabled
- [ ] **Artifact Management**: Built artifacts stored with version tags
- [ ] **Deployment Automation**: One-click deploy to staging/production
- [ ] **Rollback Strategy**: Automated rollback on deployment failure
- [ ] **Monitoring**: Pipeline metrics, build success rates tracked
- [ ] **Secrets Management**: No hardcoded secrets, use vault/parameter store

### Quick Example (GitHub Actions)

```yaml
# .github/workflows/ci.yml
name: CI Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '18'
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Run linters
        run: npm run lint

      - name: Run tests
        run: npm test -- --coverage

      - name: Check coverage threshold
        run: |
          COVERAGE=$(cat coverage/coverage-summary.json | jq '.total.lines.pct')
          if (( $(echo "$COVERAGE < 80" | bc -l) )); then
            echo "Coverage $COVERAGE% is below 80% threshold"
            exit 1
          fi

  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run security audit
        run: npm audit --audit-level=high

      - name: SAST scan
        uses: github/codeql-action/analyze@v2

  build:
    needs: [test, security]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build Docker image
        run: docker build -t myapp:${{ github.sha }} .

      - name: Scan container
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'myapp:${{ github.sha }}'
          severity: 'CRITICAL,HIGH'
          exit-code: '1'

      - name: Push to registry
        run: |
          echo "${{ secrets.DOCKER_PASSWORD }}" | docker login -u "${{ secrets.DOCKER_USERNAME }}" --password-stdin
          docker tag myapp:${{ github.sha }} registry.example.com/myapp:${{ github.sha }}
          docker push registry.example.com/myapp:${{ github.sha }}
```

### Quick Links to Level 2

- [GitHub Actions Workflows](#github-actions-workflows)
- [GitLab CI Configuration](#gitlab-ci-configuration)
- [Testing Gates](#testing-gates)
- [Security Scanning](#security-scanning)
- [Artifact Management](#artifact-management)
- [Deployment Strategies](#deployment-strategies)
- [Environment Management](#environment-management)

---

## Level 2: Implementation (<5,000 tokens, 30 minutes)

### GitHub Actions Workflows

**Multi-Stage Pipeline**

```yaml
# .github/workflows/cd.yml
name: CD Pipeline

on:
  push:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  deploy-staging:
    runs-on: ubuntu-latest
    environment:
      name: staging
      url: https://staging.example.com
    steps:
      - uses: actions/checkout@v4

      - name: Deploy to staging
        run: |
          kubectl set image deployment/myapp \
            myapp=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} \
            --namespace=staging

      - name: Wait for rollout
        run: kubectl rollout status deployment/myapp -n staging --timeout=5m

      - name: Run smoke tests
        run: |
          curl -f https://staging.example.com/health || exit 1

  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    environment:
      name: production
      url: https://example.com
    steps:
      - uses: actions/checkout@v4

      - name: Blue-green deployment
        run: |
          # Deploy to green environment
          kubectl set image deployment/myapp-green \
            myapp=${{ env.REGISTRY }}/${{ env.IM

... [truncated for markdown readability]

## Skill2Query (before)

- Use ci-cd to CI/CD pipeline standards for GitHub Actions, GitLab CI, and deployment automation. Covers testing gates, security scanning, artifact management, and deployment strategies for reliable software delivery
- Can you CI/CD pipeline standards for GitHub Actions, GitLab CI, and deployment automation. Covers testing gates, security scanning, artifact management, and deployment strategies for reliable software delivery?
- I need help with CI/CD pipeline standards for GitHub Actions, GitLab CI, and deployment automation. Covers testing gates, security scanning, artifact management, and deployment strategies for reliable software delivery
