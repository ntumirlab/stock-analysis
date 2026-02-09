#!/bin/bash
# Helper script to encode .env file to Base64 for GitHub Secrets
# Usage: .github/scripts/encode-env.sh

set -e

if [ ! -f ".env" ]; then
    echo "❌ Error: .env file not found in current directory"
    echo "💡 Make sure you run this from the project root"
    exit 1
fi

echo "📋 Encoding .env file to Base64..."
echo ""

# Encode the .env file to base64
BASE64_CONTENT=$(base64 -i .env)

echo "✅ Base64 encoded successfully!"
echo ""
echo "================================================"
echo "🔐 Copy the content below to GitHub Secrets as:"
echo "    Secret name: ENV_FILE_BASE64"
echo "================================================"
echo ""
echo "$BASE64_CONTENT"
echo ""
echo "================================================"
echo ""
echo "📝 Next steps:"
echo "1. Go to: https://github.com/YOUR_REPO/settings/secrets/actions"
echo "2. Click 'New repository secret'"
echo "3. Name: ENV_FILE_BASE64"
echo "4. Value: Paste the Base64 string above"
echo "5. Click 'Add secret'"
echo ""
echo "⚠️  IMPORTANT: Delete the old ENV_FILE secret after verification!"
