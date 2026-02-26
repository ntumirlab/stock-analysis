# Version Tagging & Release Management

## Overview

This project uses **automatic semantic versioning** for all production deployments. Every successful deployment creates a Git tag and GitHub Release automatically.

---

## 📌 Versioning Scheme

### Format: `v1.0.{COMMIT_COUNT}`

Example: `v1.0.247`

- **Major (1):** Project version
- **Minor (0):** Feature set version
- **Patch (247):** Auto-incremented based on commit count

### Why This Approach?

✅ **Automatic:** No manual version bumping needed  
✅ **Unique:** Every deployment has a unique version  
✅ **Traceable:** Version number = exact commit in history  
✅ **Sortable:** Higher number = newer version

---

## 🏷️ How Tags Are Created

### Automatic Tagging Flow

```
Push to main
    ↓
Build passes
    ↓
Deploy succeeds
    ↓
Health check passes
    ↓
✨ Git tag created automatically
    ↓
🚀 GitHub Release published
```

### What Gets Tagged?

Only **successful production deployments** are tagged:
- ✅ Build validation passed
- ✅ Deployment completed
- ✅ Health checks passed

If ANY step fails → No tag created

---

## 📊 Viewing Deployment History

### On GitHub

**1. Releases Page:**
```
https://github.com/YOUR_USERNAME/stock-analysis/releases
```

Shows:
- Version number
- Deployment timestamp
- Who triggered it
- What changed
- Commit hash

**2. Tags Page:**
```
https://github.com/YOUR_USERNAME/stock-analysis/tags
```

Lists all version tags chronologically

**3. README Badges:**

The README shows:
- Latest deployed version
- CI/CD pipeline status
- Production deployment status

### On Server

**Check current version:**
```bash
ssh -p 10220 user@server
cd ~/stock-analysis
cat VERSION
# Output: v1.0.247
```

**Detailed version info:**
```bash
.github/scripts/show-version.sh
# Output:
# 🏷️  Current Production Version
# ================================
# Version:    v1.0.247
# Commit:     a1b2c3d
# Deployed:   2026-02-10T12:34:56Z
# ================================
```

**Via Docker:**
```bash
docker compose exec web cat /app/VERSION
```

---

## 🔍 Finding Specific Deployments

### By Version Number

```bash
# View tag details
git show v1.0.247

# Checkout specific version
git checkout v1.0.247

# Compare versions
git diff v1.0.240..v1.0.247
```

### By Date

```bash
# Find deployment on specific date
git log --tags --simplify-by-decoration --pretty="format:%ci %d" | grep "2026-02-10"
```

### By Commit

```bash
# Find version for commit
git describe --tags a1b2c3d
# Output: v1.0.247
```

---

## 🔙 Rollback to Specific Version

### Method 1: Using Rollback Script

```bash
ssh -p 10220 user@server
cd ~/stock-analysis

# Rollback to previous version
.github/scripts/rollback.sh
```

### Method 2: Deploy Specific Tag

```bash
# On your local machine
git fetch --all --tags
git checkout v1.0.240  # The version you want
git push origin HEAD:main --force

# This triggers deployment of that version
```

### Method 3: GitHub UI

1. Go to: Releases → Select version
2. Copy the commit hash
3. Create new branch from that commit
4. Open PR to main
5. Merge → Auto-deploy

---

## 📈 Version Analytics

### Count Deployments

```bash
# Total deployments
git tag | wc -l

# Deployments this month
git log --tags --since="1 month ago" --format="%ai %d" | grep "tag:" | wc -l
```

### Deployment Frequency

```bash
# Show last 10 deployments with dates
git log --tags --simplify-by-decoration --pretty="format:%ai %d" -10
```

### Average Time Between Deployments

```bash
# Show time between last 5 tags
git log --tags --simplify-by-decoration --pretty="format:%ai" -5 | \
  awk '{print $1}' | \
  xargs -I {} date -d {} +%s | \
  awk 'NR>1{print ($1-prev)/3600 " hours"} {prev=$1}'
```

---

## 🎯 Best Practices

### DO ✅

- **Let CI/CD handle versioning:** Don't create manual tags
- **Always deploy via main branch:** Tags are created for main deployments
- **Use releases for documentation:** Add release notes if needed
- **Monitor badge status:** Keep README badges updated

### DON'T ❌

- **Don't manually create version tags:** Automated process only
- **Don't delete tags:** They're production history
- **Don't force push to main:** Breaks version tracking
- **Don't skip health checks:** Tags need validation

---

## 🔧 Customization

### Change Version Format

Edit `.github/workflows/deploy.yml`:

```yaml
- name: Generate Version
  run: |
    # Change this line:
    VERSION="v1.0.${COMMIT_COUNT}"
    
    # Examples:
    # Date-based: VERSION="v$(date +%Y.%m).${COMMIT_COUNT}"
    # Git tags:   VERSION="$(git describe --tags)"
```

### Add Pre-release Versions

For staging deployments:

```yaml
VERSION="v1.0.${COMMIT_COUNT}-beta"
# Creates: v1.0.247-beta
```

### Include Branch Name

For feature branches:

```yaml
BRANCH=$(git rev-parse --abbrev-ref HEAD)
VERSION="v1.0.${COMMIT_COUNT}-${BRANCH}"
# Creates: v1.0.247-feat-auth
```

---

## 🐛 Troubleshooting

### "Tag already exists" Error

**Cause:** Trying to create duplicate tag

**Fix:**
```bash
# Delete local tag
git tag -d v1.0.247

# Delete remote tag (be careful!)
git push origin :refs/tags/v1.0.247

# Re-run deployment
```

### Tag Created but No Release

**Cause:** GitHub token permission issue

**Fix:**
1. Check workflow permissions: Settings → Actions → Workflow permissions
2. Enable "Read and write permissions"
3. Re-run failed workflow

### Version Not Showing on Server

**Cause:** VERSION file not created

**Fix:**
```bash
ssh -p 10220 user@server
cd ~/stock-analysis

# Manually create version file
echo "v1.0.247" > VERSION
echo '{"version":"v1.0.247","commit":"a1b2c3d","deployed_at":"2026-02-10T12:00:00Z"}' > version.json
```

---

## 📚 Related Documentation

- [CI/CD Operations Guide](./.github/OPERATIONS.md)
- [Deployment Workflow](../.github/workflows/deploy.yml)
- [Rollback Procedures](./.github/scripts/rollback.sh)

---

## Changelog

- **2026-02-10:** Added automatic versioning and release creation
- **2026-02-10:** Implemented version badges in README
- **2026-02-10:** Created version tracking scripts
