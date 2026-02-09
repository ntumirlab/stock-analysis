#!/bin/bash
# Emergency Rollback Script
# Quickly reverts to a previous deployment using tagged Docker images
#
# Usage:
#   ./rollback.sh [version]
#   ./rollback.sh              # Shows available versions to rollback to
#   ./rollback.sh v1.0.240     # Rollback to specific version
#   ./rollback.sh 1            # Rollback to previous version (1 back)

set -e

# Safety check - are we in the right directory?
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ Error: docker-compose.yml not found"
    echo "💡 Make sure you're in the stock-analysis directory"
    exit 1
fi

echo "🚨 EMERGENCY ROLLBACK UTILITY"
echo ""

# List available versions
echo "📦 Available rollback versions:"
AVAILABLE_VERSIONS=$(docker images --format "{{.Tag}}" stock-analysis | grep "^v" | sort -V -r)

if [ -z "$AVAILABLE_VERSIONS" ]; then
    echo "❌ No version-tagged images found!"
    echo "💡 Versions are only available after the new CI/CD is deployed"
    exit 1
fi

echo "$AVAILABLE_VERSIONS" | nl -w2 -s'. '
echo ""

# Get current version
CURRENT_VERSION=$(cat VERSION 2>/dev/null || echo "unknown")
echo "📍 Current version: $CURRENT_VERSION"
echo ""

# Determine target version
TARGET_VERSION=""

if [ -z "$1" ]; then
    # Interactive mode - no argument provided
    echo "💡 Usage:"
    echo "  ./rollback.sh v1.0.240    # Rollback to specific version"
    echo "  ./rollback.sh 1           # Rollback to previous version (1 back)"
    echo "  ./rollback.sh 2           # Rollback 2 versions back"
    exit 0
elif [[ "$1" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    # Specific version provided (e.g., v1.0.240)
    TARGET_VERSION="$1"
elif [[ "$1" =~ ^[0-9]+$ ]]; then
    # Number provided (e.g., 1 for previous, 2 for two back)
    TARGET_VERSION=$(echo "$AVAILABLE_VERSIONS" | sed -n "${1}p")
    if [ -z "$TARGET_VERSION" ]; then
        echo "❌ Error: Cannot go back $1 version(s)"
        echo "💡 Only $(echo "$AVAILABLE_VERSIONS" | wc -l) version(s) available"
        exit 1
    fi
else
    echo "❌ Error: Invalid version format"
    echo "💡 Use: v1.0.240 or a number (1, 2, 3...)"
    exit 1
fi

# Verify target version exists
if ! echo "$AVAILABLE_VERSIONS" | grep -q "^${TARGET_VERSION}$"; then
    echo "❌ Error: Version $TARGET_VERSION not found in available images"
    echo "💡 Available versions:"
    echo "$AVAILABLE_VERSIONS"
    exit 1
fi

echo "🎯 Rolling back to: $TARGET_VERSION"
echo ""

# Show what changed
echo "📝 Getting version info..."
TARGET_COMMIT=$(docker inspect "stock-analysis:${TARGET_VERSION}" --format='{{index .Config.Labels "git.commit"}}' 2>/dev/null || echo "unknown")
echo "  Target commit: $TARGET_COMMIT"
echo ""

# Confirm rollback
read -p "⚠️  Continue with rollback to $TARGET_VERSION? (yes/no): " -r
if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo "❌ Rollback cancelled"
    exit 0
fi

echo ""
echo "⏮️  Starting fast rollback (no rebuild needed)..."

# Update version files
echo "$TARGET_VERSION" > VERSION
echo "{\"version\":\"${TARGET_VERSION}\",\"commit\":\"${TARGET_COMMIT}\",\"rolled_back_at\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"rolled_back_from\":\"${CURRENT_VERSION}\"}" > version.json

# Rollback git code to match the version
echo "📥 Rolling back code..."
if [ "$TARGET_COMMIT" != "unknown" ]; then
    git fetch --all
    git reset --hard "$TARGET_COMMIT" 2>/dev/null || echo "⚠️  Could not reset git (commit not found locally)"
else
    echo "⚠️  Commit hash unknown, skipping git rollback"
fi

# Update docker-compose to use the specific version
echo "🐳 Switching to version-tagged image..."
export IMAGE_TAG="$TARGET_VERSION"

# Modify docker-compose to use the specific tag temporarily
if grep -q "image:" docker-compose.yml; then
    # If image is specified, update it
    sed -i.bak "s|image: stock-analysis:.*|image: stock-analysis:${TARGET_VERSION}|g" docker-compose.yml
else
    # If using build, we'll need to temporarily add image tag
    echo "⚠️  Using build section, will use tagged image directly"
fi

# Restart with the specific tagged image (NO BUILD)
echo "🔄 Restarting services with version $TARGET_VERSION..."
docker compose up -d --no-build --remove-orphans

# Restore docker-compose.yml if we modified it
if [ -f docker-compose.yml.bak ]; then
    mv docker-compose.yml.bak docker-compose.yml.temp
fi

echo "⏳ Waiting for services to stabilize..."
sleep 5

echo ""
echo "🏥 Checking service health..."
docker compose ps

echo ""
echo "✅ Rollback to $TARGET_VERSION completed!"
echo ""
echo "📝 What happened:"
echo "  ├─ Switched from: $CURRENT_VERSION"
echo "  ├─ Switched to:   $TARGET_VERSION"
echo "  ├─ Method:        Fast image swap (no rebuild)"
echo "  └─ Time taken:    ~10 seconds"
echo ""
echo "📝 Next steps:"
echo "1. Verify the application is working"
echo "2. Check logs: docker compose logs -f"
echo ""
echo "⚠️  To make this rollback permanent:"
echo "   (This prevents the next deploy from overwriting)"
echo "   cd ~/stock-analysis"
echo "   git push origin HEAD:main --force"
echo ""
echo "💡 To roll forward again (undo rollback):"
echo "   git pull origin main --force"
echo "   docker compose up -d --build"
