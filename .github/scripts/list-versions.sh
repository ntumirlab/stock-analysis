#!/bin/bash
# List Available Rollback Versions
# Shows all Docker images tagged with versions that can be used for rollback

set -e

echo "📦 Available Rollback Versions"
echo "================================"
echo ""

# Check if any version-tagged images exist
VERSIONS=$(docker images --format "{{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}" stock-analysis 2>/dev/null | grep ":v" | sort -t: -k2 -V -r)

if [ -z "$VERSIONS" ]; then
    echo "❌ No version-tagged images found"
    echo ""
    echo "💡 Version-tagged images are created automatically by CI/CD"
    echo "💡 After the first deployment with the new workflow, you'll see versions here"
    exit 0
fi

# Get current version
CURRENT_VERSION=$(cat ~/stock-analysis/VERSION 2>/dev/null || echo "unknown")
echo "📍 Current Production Version: $CURRENT_VERSION"
echo ""

# Display versions in a table
echo "Available versions for rollback:"
echo "--------------------------------"
printf "%-4s %-15s %-10s %s\n" "#" "VERSION" "SIZE" "CREATED"
echo "--------------------------------"

COUNT=1
echo "$VERSIONS" | while IFS=$'\t' read -r IMAGE SIZE CREATED; do
    VERSION=$(echo "$IMAGE" | cut -d: -f2)
    
    # Mark current version
    if [ "$VERSION" = "$CURRENT_VERSION" ]; then
        printf "%-4s %-15s %-10s %s %s\n" "$COUNT" "$VERSION" "$SIZE" "$CREATED" "← CURRENT"
    else
        printf "%-4s %-15s %-10s %s\n" "$COUNT" "$VERSION" "$SIZE" "$CREATED"
    fi
    COUNT=$((COUNT + 1))
done

echo "--------------------------------"
echo ""

# Show rollback command
echo "🔙 To rollback, use:"
echo "   .github/scripts/rollback.sh <version>"
echo ""
echo "Examples:"
echo "   .github/scripts/rollback.sh v1.0.240    # Rollback to specific version"
echo "   .github/scripts/rollback.sh 1           # Rollback to previous version (1 step back)"
echo "   .github/scripts/rollback.sh 2           # Rollback 2 versions back"
echo ""

# Show detailed info about latest version
LATEST_VERSION=$(echo "$VERSIONS" | head -1 | cut -f1 | cut -d: -f2)
echo "📝 Latest Version Details: $LATEST_VERSION"
COMMIT=$(docker inspect "stock-analysis:${LATEST_VERSION}" --format='{{index .Config.Labels "git.commit"}}' 2>/dev/null || echo "unknown")
BUILD_DATE=$(docker inspect "stock-analysis:${LATEST_VERSION}" --format='{{index .Config.Labels "org.opencontainers.image.created"}}' 2>/dev/null || echo "unknown")
echo "   Commit: $COMMIT"
echo "   Built:  $BUILD_DATE"
echo ""

# Disk usage summary
echo "💿 Disk Usage Summary:"
docker images stock-analysis --format "table {{.Tag}}\t{{.Size}}" | grep -v "TAG" | awk '{sum+=$2} END {print "   Total: " NR " images"}'
