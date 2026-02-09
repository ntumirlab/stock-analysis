# Fast Rollback Guide

## Overview

The enhanced CI/CD pipeline now supports **instant rollback** by using version-tagged Docker images. No rebuild needed!

---

## 🚀 How Fast Rollback Works

### Traditional Rollback (OLD):
```
Rollback time: ~3-5 minutes
├─ Revert code: git reset
├─ Rebuild image: 2-3 min ⏳
└─ Restart: 30 sec
```

### Fast Rollback (NEW):
```
Rollback time: ~10 seconds ⚡
├─ List versions: instant
├─ Switch image tag: instant
└─ Restart: 10 sec
```

**Why it's fast:** We keep the last 3 pre-built images, so rollback is just swapping which image to run!

---

## 📦 Version Tagging

Every successful deployment creates:

```
Git Repository:
└─ Tag: v1.0.247

Docker Images on Server:
├─ stock-analysis:v1.0.247  (newest)
├─ stock-analysis:v1.0.246
├─ stock-analysis:v1.0.245  (oldest kept)
└─ stock-analysis:latest    (alias to v1.0.247)
```

**Retention Policy:** Last 3 versions kept automatically

---

## 🔙 Rollback Methods

### Method 1: List & Choose (Recommended)

**Step 1: See available versions**
```bash
ssh -p 10220 user@server
cd ~/stock-analysis
.github/scripts/list-versions.sh
```

Output:
```
📦 Available Rollback Versions
================================

📍 Current Production Version: v1.0.247

Available versions for rollback:
--------------------------------
#    VERSION         SIZE       CREATED
--------------------------------
1    v1.0.247        1.2GB      2026-02-10 14:30  ← CURRENT
2    v1.0.246        1.2GB      2026-02-10 12:00
3    v1.0.245        1.2GB      2026-02-09 18:00
--------------------------------
```

**Step 2: Rollback to chosen version**
```bash
# Rollback to previous version (1 step back)
.github/scripts/rollback.sh 1

# Or rollback to specific version
.github/scripts/rollback.sh v1.0.246
```

---

### Method 2: Quick Previous Version

```bash
ssh -p 10220 user@server
cd ~/stock-analysis

# Rollback to immediately previous version
.github/scripts/rollback.sh 1
```

This will:
1. Show what version you're rolling back to
2. Ask for confirmation
3. Switch images instantly (no rebuild!)
4. Restart services
5. Show health status

**Time: ~10 seconds total** ⚡

---

### Method 3: Specific Version by Number

```bash
# Go back 2 versions
.github/scripts/rollback.sh 2

# Go back 3 versions
.github/scripts/rollback.sh 3
```

Numbers refer to the list shown by `list-versions.sh`

---

## 🎯 Rollback Scenarios

### Scenario 1: New Deploy Has Bug

```bash
# Current: v1.0.247 (broken)
# Goal: Rollback to v1.0.246 (working)

.github/scripts/rollback.sh 1
# ⏱️  10 seconds → Service restored!
```

### Scenario 2: Need Specific Old Version

```bash
# Current: v1.0.247
# Goal: Go back to v1.0.245 (2 versions ago)

.github/scripts/list-versions.sh  # Find version number
.github/scripts/rollback.sh v1.0.245
# ⏱️  10 seconds → Service restored!
```

### Scenario 3: Test Then Roll Forward

```bash
# Rollback to test
.github/scripts/rollback.sh 1

# If it works, make permanent:
git push origin HEAD:main --force

# If rollback didn't help, go forward again:
git pull origin main --force
docker compose up -d --build
```

---

## 🔍 Verify Rollback Success

### Check Current Version

```bash
cat VERSION
# Output: v1.0.246

cat version.json
# Shows rolled_back_at timestamp
```

### Check Container Status

```bash
docker compose ps
# All services should show "Up"

docker compose logs -f
# Check for errors
```

### Check Image Being Used

```bash
docker compose images
# Should show stock-analysis:v1.0.246
```

---

## ⚠️ Important Notes

### Rollback Limitations

**Can only rollback to versions in the last 3 deployments:**
```
✅ Can rollback: v1.0.247 → v1.0.246 (available)
✅ Can rollback: v1.0.247 → v1.0.245 (available)
❌ Cannot rollback: v1.0.247 → v1.0.240 (image deleted)
```

**Why 3 versions?**
- Balance between disk space and rollback capability
- 3 versions = ~3.6GB disk space
- Covers 99% of rollback needs

**If you need older version:**
1. Deploy that version from Git
2. Let CI/CD rebuild it
3. Time: ~3-5 minutes (normal deploy)

---

### Rollback Does NOT Affect

❌ **Database:** Data persists across rollbacks  
❌ **Configuration:** .env file stays the same  
❌ **Logs:** Previous logs remain  
❌ **Git repository:** Still on latest commit (until you force push)

✅ **Only affects:** Application code and dependencies

---

### Making Rollback Permanent

After rollback, the server code is at an older version, but Git still points to latest.

**Next deploy will override your rollback!**

To make rollback permanent:
```bash
# After successful rollback
git push origin HEAD:main --force

# This updates main branch to match rolled-back version
# Now CI/CD won't override it
```

---

## 🎓 Advanced Usage

### Compare Two Versions

```bash
# See differences between versions
docker image inspect stock-analysis:v1.0.247 --format='{{index .Config.Labels "git.commit"}}'
# Output: a1b2c3d

docker image inspect stock-analysis:v1.0.246 --format='{{index .Config.Labels "git.commit"}}'
# Output: x9y8z7w

# Compare commits
git diff x9y8z7w a1b2c3d
```

### Rollback Specific Service

If you only want to rollback one service in docker-compose:

```bash
# Edit docker-compose.yml temporarily
vim docker-compose.yml

# Change:
#   image: stock-analysis:latest
# To:
#   image: stock-analysis:v1.0.246

# Restart just that service
docker compose up -d service_name
```

### Manual Image Tag Swap

```bash
# Tag an old version as latest
docker tag stock-analysis:v1.0.246 stock-analysis:latest

# Restart
docker compose up -d --no-build
```

---

## 📊 Monitoring Rollbacks

### Check Rollback History

```bash
cat version.json
```

Output:
```json
{
  "version": "v1.0.246",
  "commit": "x9y8z7w",
  "rolled_back_at": "2026-02-10T14:45:00Z",
  "rolled_back_from": "v1.0.247"
}
```

### Rollback Metrics

```bash
# How many times rolled back?
grep -c "rolled_back_at" version.json || echo "0"

# What version are we on?
cat VERSION
```

---

## 🐛 Troubleshooting

### "No version-tagged images found"

**Cause:** You haven't deployed with the new CI/CD yet

**Solution:** 
1. Deploy once with the new workflow
2. Images will be tagged automatically
3. Future rollbacks will work

### "Version not found"

**Cause:** Trying to rollback to a version older than last 3

**Solution:**
```bash
# List available versions
.github/scripts/list-versions.sh

# Choose from available versions only
```

### Rollback Failed - Service Won't Start

**Cause:** Image might be corrupted or incompatible

**Solution:**
```bash
# Try another version
.github/scripts/rollback.sh 2

# Or force rebuild
docker compose up -d --build
```

---

## 📋 Rollback Checklist

Before rollback:
- [ ] Identify which version to rollback to
- [ ] Verify version exists (run list-versions.sh)
- [ ] Notify team of upcoming rollback

During rollback:
- [ ] Run rollback script
- [ ] Confirm when prompted
- [ ] Wait ~10 seconds

After rollback:
- [ ] Check service status: `docker compose ps`
- [ ] Verify version: `cat VERSION`
- [ ] Test critical features
- [ ] Check logs: `docker compose logs -f`
- [ ] Monitor for 5-10 minutes
- [ ] Decide: Keep rollback or investigate issue

If keeping rollback:
- [ ] Force push to Git: `git push origin HEAD:main --force`
- [ ] Update team
- [ ] Plan fix for broken version

---

## 🆚 Comparison: Old vs New Rollback

| Feature | Old Rollback | New Fast Rollback |
|---------|-------------|-------------------|
| **Speed** | 3-5 minutes | ~10 seconds ⚡ |
| **Rebuild needed?** | Yes ❌ | No ✅ |
| **Can fail?** | Yes (build errors) | Rarely ✅ |
| **Disk space** | Low | Medium (+3GB) |
| **Versions available** | Any commit | Last 3 deploys |
| **Complexity** | Simple | Automated ✅ |
| **Reliability** | Medium | High ✅ |

---

## 📚 Related Scripts

- **list-versions.sh:** See available rollback versions
- **rollback.sh:** Execute fast rollback
- **show-version.sh:** Display current deployed version

---

## Summary

**Fast Rollback gives you:**
- ⚡ **10-second recovery** from bad deployments
- 🛡️ **Safety net** for risky changes
- 📦 **Last 3 versions** always ready
- 🔄 **No rebuild** needed
- ✅ **High reliability** (using proven images)

**Perfect for:**
- Production emergencies
- Testing new features
- Quick A/B comparisons
- Reverting risky changes

**Trade-off:**
- Uses ~3.6GB disk space for 3 images
- Can only rollback to recent versions
- Worth it for peace of mind! 🎯
