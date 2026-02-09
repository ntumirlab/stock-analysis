# CI/CD Operations Guide

## Overview

This document explains how the professional CI/CD pipeline works and how to operate it.

---

## 🔄 Pipeline Flow

```
┌─────────────┐
│  Push to    │
│    main     │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────┐
│  Job 1: Build & Validate    │
│  - Checkout code            │
│  - Setup Docker Buildx      │
│  - Build image (no push)    │
│  - Cache layers             │
└──────┬──────────────────────┘
       │ ✅ Build Success
       ▼
┌─────────────────────────────┐
│  Job 2: Deploy              │
│  Step 1: Deploy via SSH     │
│  - Pull latest code         │
│  - Decode .env (Base64)     │
│  - Pull Docker images       │
│  - Restart services         │
└──────┬──────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│  Step 2: Health Check       │
│  - Verify containers        │
│  - Check health status      │
│  - Display resources        │
└──────┬──────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│  Step 3: Cleanup            │
│  - Remove dangling images   │
│  - Prune old containers     │
│  - Show disk usage          │
└─────────────────────────────┘
```

---

## 📊 Monitoring Deployments

### View Workflow Runs

1. Go to: `https://github.com/YOUR_REPO/actions`
2. Click on the latest "CI/CD Pipeline" run
3. Expand each job to see detailed logs

### Key Checkpoints

**Build Job:**
- ✅ Should complete in ~30-60 seconds (with cache)
- ❌ Fails = Dockerfile or dependency issues

**Deploy Step 1:**
- ✅ Shows: `🚀 Starting deployment...`
- ✅ Shows: `🔄 Restarting services...`
- ❌ Fails = SSH connection or Docker issues

**Health Check:**
- ✅ Shows: `✅ Container Status:` with all services `Up`
- ⚠️ Warns: If any containers are `unhealthy`
- ❌ Fails = No containers running

---

## 🚨 Troubleshooting

### Build Job Fails

**Symptom:** Red ❌ on "Build & Validate" job

**Common Causes:**
1. **Syntax error in Dockerfile**
   ```
   Error: failed to solve: failed to compute cache key
   ```
   - Check: Recent changes to `Dockerfile`
   - Fix: Review Dockerfile syntax

2. **Missing dependency**
   ```
   Error: Could not find a version that satisfies requirement
   ```
   - Check: `environment.yml` or `requirements.txt`
   - Fix: Ensure all packages are available

**How to Fix:**
```bash
# Test build locally first
docker build -t test .

# If it works locally, push the fix
git add Dockerfile
git commit -m "fix: correct Dockerfile syntax"
git push
```

---

### Deploy Fails - SSH Connection

**Symptom:**
```
Error: ssh: connect to host XX.XX.XX.XX port 10220: Connection refused
```

**Causes:**
- Server is down
- Firewall blocking port 10220
- SSH service not running

**How to Fix:**
```bash
# Test SSH manually
ssh -p 10220 user@your-server

# If connection fails, check server status
# Contact server administrator
```

---

### Deploy Fails - Container Won't Start

**Symptom:** Health check shows `❌ ERROR: No containers are running!`

**Debugging Steps:**

1. **SSH into server:**
   ```bash
   ssh -p 10220 user@your-server
   cd ~/stock-analysis
   ```

2. **Check container logs:**
   ```bash
   docker compose logs
   docker compose logs web  # Check specific service
   ```

3. **Common Issues:**

   **Port already in use:**
   ```
   Error: bind: address already in use
   ```
   Fix:
   ```bash
   # Find process using the port
   sudo lsof -i :8050
   
   # Kill old process or change port in docker-compose.yml
   ```

   **Missing environment variable:**
   ```
   Error: POSTGRES_PASSWORD is not set
   ```
   Fix:
   ```bash
   # Check .env file exists and is complete
   cat .env
   
   # Re-run the encode script and update GitHub Secret
   ```

4. **Manual restart:**
   ```bash
   docker compose down
   docker compose up -d
   docker compose ps
   ```

---

## 🔙 Emergency Rollback

If a deployment breaks production, rollback immediately:

### Option 1: Using Rollback Script (Recommended)

```bash
# SSH into server
ssh -p 10220 user@your-server
cd ~/stock-analysis

# Rollback to previous commit
.github/scripts/rollback.sh

# Or rollback 2 commits
.github/scripts/rollback.sh 2
```

### Option 2: Manual Rollback

```bash
# SSH into server
ssh -p 10220 user@your-server
cd ~/stock-analysis

# Check commit history
git log --oneline -5

# Rollback to specific commit
git reset --hard COMMIT_HASH
docker compose up -d --build

# Verify
docker compose ps
```

### Option 3: Revert via GitHub

```bash
# On your local machine
git revert HEAD
git push origin main

# Pipeline will auto-deploy the reverted version
```

---

## 🔐 Security Operations

### Rotating Secrets

**When to Rotate:**
- Every 90 days (recommended)
- After team member leaves
- If secret is compromised

**How to Rotate:**

1. **Update local .env file** with new secrets

2. **Re-encode:**
   ```bash
   .github/scripts/encode-env.sh
   ```

3. **Update GitHub Secret:**
   - Go to: Settings → Secrets → Actions
   - Click `ENV_FILE_BASE64`
   - Click "Update secret"
   - Paste new Base64 value

4. **Trigger deployment:**
   ```bash
   git commit --allow-empty -m "chore: rotate secrets"
   git push
   ```

---

## 📈 Performance Optimization

### Build Cache Hit Rate

**Good cache hit:** Build completes in ~30 seconds
**Cache miss:** Build takes 3-5 minutes

**Improve cache hits:**
- Don't change Dockerfile frequently
- Keep dependency versions stable
- Use layer-aware Dockerfile structure

### Deployment Speed

**Current:** ~2-3 minutes total
- Build: 30-60s
- Deploy: 60-90s
- Health check: 10-20s

**To speed up:**
1. **Pre-pull images** (already implemented in Phase 3)
2. **Reduce image size:**
   ```dockerfile
   # Use alpine base images
   FROM python:3.10-alpine
   ```
3. **Multi-stage builds** for smaller final images

---

## 📋 Maintenance Checklist

### Weekly
- [ ] Review failed deployments (if any)
- [ ] Check server disk usage
- [ ] Review Docker logs for errors

### Monthly
- [ ] Update dependencies
- [ ] Review and clean old Docker images
- [ ] Test rollback procedure

### Quarterly
- [ ] Rotate secrets
- [ ] Review and update documentation
- [ ] Audit GitHub Actions logs

---

## 🆘 Emergency Contacts

**Build Issues:**
- Check: GitHub Actions logs
- Contact: DevOps team

**Server Issues:**
- SSH access problems
- Contact: Infrastructure team

**Application Issues:**
- Container crashes
- Contact: Development team

---

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Compose Reference](https://docs.docker.com/compose/)
- [SSH Action Documentation](https://github.com/appleboy/ssh-action)

---

## Changelog

- **2026-02-10:** Phase 3 - Added health checks and resilience
- **2026-02-10:** Phase 2 - Added Base64 secret management
- **2026-02-10:** Phase 1 - Split build and deploy jobs
