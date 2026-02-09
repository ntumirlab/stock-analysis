#!/bin/bash
# Script to fetch and display current deployed version
# Can be run locally or via web endpoint

set -e

VERSION_FILE="VERSION"
VERSION_JSON="version.json"

if [ -f "$VERSION_JSON" ]; then
    # Parse JSON for detailed info
    VERSION=$(cat "$VERSION_JSON" | grep -o '"version":"[^"]*"' | cut -d'"' -f4)
    COMMIT=$(cat "$VERSION_JSON" | grep -o '"commit":"[^"]*"' | cut -d'"' -f4)
    DEPLOYED=$(cat "$VERSION_JSON" | grep -o '"deployed_at":"[^"]*"' | cut -d'"' -f4)
    
    echo "🏷️  Current Production Version"
    echo "================================"
    echo "Version:    $VERSION"
    echo "Commit:     $COMMIT"
    echo "Deployed:   $DEPLOYED"
    echo "================================"
elif [ -f "$VERSION_FILE" ]; then
    # Fallback to simple version file
    VERSION=$(cat "$VERSION_FILE")
    echo "🏷️  Version: $VERSION"
else
    echo "⚠️  No version information found"
    exit 1
fi
