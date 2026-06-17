"""One-time script: extract the FinLab Svelte bundle from any existing HTML report.

Reads a sample HTML, strips lines 9-10 (reportJson / positionJson),
and writes the rest as a reusable template with placeholders.

Usage:
    python3 scripts/extract_report_template.py [path/to/any_report.html]

If no path is given, it picks the first .html from assets/GoldenAITWStrategyWeekly/.
Output: assets/finlab_report_template.html
"""
import os
import sys
import glob

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets')
OUTPUT = os.path.join(ASSETS, 'finlab_report_template.html')

PLACEHOLDER_LINES = (
    '    <script>const reportJson = {{REPORT_JSON}};</script>\n',
    '    <script>const positionJson = {{POSITION_JSON}};</script>\n',
)


def extract(source_html: str):
    with open(source_html, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if len(lines) < 11:
        print(f"ERROR: {source_html} has only {len(lines)} lines, expected 11+")
        sys.exit(1)

    # lines 1-8 (index 0-7) = head/body open
    # line 9 (index 8) = reportJson  → replace with placeholder
    # line 10 (index 9) = positionJson → replace with placeholder
    # lines 11+ (index 10+) = Svelte bundle + CSS + closing tags
    template_lines = lines[:8] + list(PLACEHOLDER_LINES) + lines[10:]

    with open(OUTPUT, 'w', encoding='utf-8') as f:
        f.writelines(template_lines)

    size_mb = os.path.getsize(OUTPUT) / (1024 * 1024)
    print(f"Wrote {OUTPUT} ({size_mb:.2f} MB, {len(template_lines)} lines)")


def main():
    if len(sys.argv) > 1:
        source = sys.argv[1]
    else:
        pattern = os.path.join(ASSETS, 'GoldenAITWStrategyWeekly', '*.html')
        candidates = sorted(glob.glob(pattern))
        if not candidates:
            print(f"ERROR: no HTML files found in {pattern}")
            sys.exit(1)
        source = candidates[0]

    print(f"Source: {source}")
    extract(source)


if __name__ == '__main__':
    main()
