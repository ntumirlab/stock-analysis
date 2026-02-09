# CI/CD Flow Diagram

## Complete Deployment Flow with Versioning

```
┌─────────────────────────────────────────────────────────────┐
│  Developer Action: git push origin main                     │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  📦 JOB 1: BUILD & VALIDATE (GitHub Actions Runner)         │
├─────────────────────────────────────────────────────────────┤
│  1. Checkout code from repository                           │
│  2. Generate version: v1.0.{COMMIT_COUNT}                   │
│     Example: v1.0.247                                       │
│  3. Build Docker image (dry-run, no push)                   │
│  4. Validate Dockerfile syntax & dependencies               │
│  5. Cache Docker layers for faster builds                   │
│                                                              │
│  ✅ Output: version, sha_short                              │
└────────────────┬────────────────────────────────────────────┘
                 │ Build Success ✅
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  🚀 JOB 2: DEPLOY (SSH to Production Server)                │
├─────────────────────────────────────────────────────────────┤
│  Step 1: Deploy via SSH                                     │
│  ├─ Connect to server (port 10220)                          │
│  ├─ Pull latest code: git pull origin main                  │
│  ├─ Decode .env: base64 -d > .env                           │
│  ├─ Write version files:                                    │
│  │   • VERSION (simple: "v1.0.247")                         │
│  │   • version.json (detailed with timestamp)               │
│  ├─ Pull Docker images (if available)                       │
│  └─ Restart: docker compose up -d --build                   │
│                                                              │
│  Step 2: Health Check                                       │
│  ├─ Verify containers running                               │
│  ├─ Check health status                                     │
│  ├─ Display resource usage (CPU/Memory)                     │
│  └─ ✅ or ❌ based on container status                      │
│                                                              │
│  Step 3: Cleanup                                            │
│  ├─ Remove dangling images only                             │
│  ├─ Prune stopped containers                                │
│  └─ Show disk usage                                         │
└────────────────┬────────────────────────────────────────────┘
                 │ Deploy Success ✅
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  🏷️  JOB 3: TAG RELEASE (GitHub Actions Runner)            │
├─────────────────────────────────────────────────────────────┤
│  1. Create annotated Git tag: v1.0.247                      │
│  2. Push tag to repository: git push origin v1.0.247        │
│  3. Create GitHub Release with:                             │
│     • Version number                                        │
│     • Deployment timestamp                                  │
│     • Commit hash                                           │
│     • Who triggered deployment                              │
│     • What changed (commit message)                         │
│     • Status checkboxes (Build ✅ Deploy ✅ Health ✅)      │
│                                                              │
│  ✅ Output: Public release on GitHub                        │
└─────────────────────────────────────────────────────────────┘

                 ▼ ▼ ▼

┌─────────────────────────────────────────────────────────────┐
│  📊 VERSION VISIBILITY                                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  On GitHub:                                                  │
│  ├─ README badges show latest version                       │
│  ├─ Releases page lists all deployments                     │
│  └─ Tags page shows version history                         │
│                                                              │
│  On Server:                                                  │
│  ├─ VERSION file: "v1.0.247"                                │
│  ├─ version.json: detailed info                             │
│  └─ Access via: cat ~/stock-analysis/VERSION                │
│                                                              │
│  For Users:                                                  │
│  ├─ Dashboard can display version                           │
│  ├─ API endpoint can expose version                         │
│  └─ Logs include version in header                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Error Handling Flow

```
┌─────────────┐
│ Build Fails │
└──────┬──────┘
       │
       ├─ ❌ Dockerfile syntax error
       ├─ ❌ Dependency not found
       └─ ❌ Docker build timeout
       │
       ▼
   🛑 STOP - No Deploy
   🛑 STOP - No Tag Created


┌─────────────┐
│Deploy Fails │
└──────┬──────┘
       │
       ├─ ❌ SSH connection error
       ├─ ❌ Docker compose fails
       └─ ❌ Service won't start
       │
       ▼
   🛑 STOP - No Tag Created
   📧 Notification sent (optional)


┌──────────────────┐
│Health Check Fails│
└────────┬─────────┘
         │
         ├─ ❌ No containers running
         ├─ ❌ Container unhealthy
         └─ ❌ Resource check failed
         │
         ▼
     🛑 STOP - No Tag Created
     🔙 Manual rollback required
```

---

## Rollback Flow

```
┌──────────────────────────┐
│  Production Issue Found  │
└────────────┬─────────────┘
             │
             ▼
┌────────────────────────────────────────┐
│  Option 1: Using Rollback Script       │
├────────────────────────────────────────┤
│  1. SSH to server                      │
│  2. Run: .github/scripts/rollback.sh   │
│  3. Confirm: yes                       │
│  4. Script does:                       │
│     • git reset --hard HEAD~1          │
│     • docker compose up -d --build     │
│  5. Check: docker compose ps           │
│  ⏱️  Time: ~30 seconds                 │
└────────────────────────────────────────┘

             OR

┌────────────────────────────────────────┐
│  Option 2: Deploy Specific Version     │
├────────────────────────────────────────┤
│  1. Find good version:                 │
│     git tag | tail -5                  │
│  2. Checkout: git checkout v1.0.240    │
│  3. Force push: git push origin        │
│        HEAD:main --force               │
│  4. Wait for CI/CD to deploy           │
│  ⏱️  Time: ~3 minutes                  │
└────────────────────────────────────────┘

             OR

┌────────────────────────────────────────┐
│  Option 3: GitHub UI Rollback          │
├────────────────────────────────────────┤
│  1. Go to: Releases page               │
│  2. Find working version: v1.0.240     │
│  3. Copy commit hash                   │
│  4. Create branch from that commit     │
│  5. PR to main → Merge                 │
│  ⏱️  Time: ~5 minutes                  │
└────────────────────────────────────────┘
```

---

## Version Tracking Timeline

```
v1.0.240  ──→  v1.0.241  ──→  v1.0.242  ──→  v1.0.243
  │             │              │              │
  │             │              │              └─ Latest (HEAD)
  │             │              │                 ├─ Tagged: Yes ✅
  │             │              │                 ├─ Released: Yes ✅
  │             │              │                 └─ Deployed: 2026-02-10 14:30
  │             │              │
  │             │              └─ Production (current)
  │             │                 ├─ Tagged: Yes ✅
  │             │                 ├─ Released: Yes ✅
  │             │                 └─ Deployed: 2026-02-10 12:00
  │             │
  │             └─ Rolled back (had bug)
  │                ├─ Tagged: Yes ✅
  │                ├─ Released: Yes ✅
  │                └─ Deployed: 2026-02-10 10:00
  │                   (Rolled back after 30 min)
  │
  └─ Stable baseline
     ├─ Tagged: Yes ✅
     ├─ Released: Yes ✅
     └─ Deployed: 2026-02-09 18:00
```

---

## Directory Structure with Version Files

```
stock-analysis/
├── .github/
│   ├── workflows/
│   │   └── deploy.yml         # ← Modified: 3 jobs, version generation
│   ├── scripts/
│   │   ├── encode-env.sh      # Phase 2: Base64 encoding
│   │   ├── rollback.sh        # Phase 3: Emergency rollback
│   │   └── show-version.sh    # NEW: Display version info
│   ├── OPERATIONS.md          # Operations guide
│   ├── SECRETS_SETUP.md       # Secrets setup
│   └── VERSIONING.md          # NEW: Version guide
│
├── VERSION                    # NEW: Generated on deploy (gitignored)
├── version.json               # NEW: Generated on deploy (gitignored)
│
├── .gitignore                 # Modified: Added VERSION files
├── README.md                  # Modified: Added badges
└── ... (rest of your code)
```

---

## What Gets Created on Each Deploy?

### On GitHub:
```
📦 Git Tag: v1.0.247
   ├─ Annotated with deployment info
   └─ Pushed to repository

🚀 GitHub Release: v1.0.247
   ├─ Title: "Release v1.0.247"
   ├─ Body: Deployment details
   ├─ Created by: github-actions[bot]
   └─ Assets: None (server-based deployment)
```

### On Server:
```
📄 VERSION file:
   v1.0.247

📄 version.json file:
   {
     "version": "v1.0.247",
     "commit": "a1b2c3d",
     "deployed_at": "2026-02-10T12:34:56Z"
   }

🐳 Docker containers:
   ├─ Rebuilt with latest code
   └─ Running with new version
```

### In README Badges:
```
[![CI/CD Pipeline](https://...badge.svg)]
   Shows: ✅ passing or ❌ failing

[![Latest Release](https://...v/release/...)]
   Shows: v1.0.247

[![Production Status](https://...status-production...)]
   Shows: production (green)
```

---

## Time Breakdown

```
┌───────────────────────────────────────┐
│  Total Deploy Time: ~3-4 minutes      │
├───────────────────────────────────────┤
│  Build Job:           30-60 sec       │
│  Deploy Job:          60-120 sec      │
│  Health Check:        10-20 sec       │
│  Cleanup:             10-20 sec       │
│  Tag & Release:       10-30 sec       │
└───────────────────────────────────────┘
```

With caching:
- First deploy: ~4-5 minutes
- Subsequent deploys: ~2-3 minutes
