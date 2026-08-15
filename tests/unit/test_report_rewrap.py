from core.report_rewrap import extract_report_data, rewrap_report_html

TEMPLATE = (
    '<html><head><title>Web Component Example</title></head><body>\n'
    '    <script>const reportJson = {{REPORT_JSON}}</script>\n'
    '    <script>const positionJson = {{POSITION_JSON}}</script>\n'
    '    <script>old bundle</script></body></html>'
)


def _finlab_html(report_json='{"a": 1}', position_json='[{"b": 2}]'):
    return (
        '<html><head><title>FinLab Report</title></head><body>'
        f'<script>const reportJson = {report_json}</script>'
        f'<script>const positionJson = {position_json}</script>'
        '<script>new bundle</script></body></html>'
    )


def test_rewrap_extracts_data_and_keeps_template_shell():
    out = rewrap_report_html(_finlab_html(), TEMPLATE)
    assert '<script>const reportJson = {"a": 1}</script>' in out
    assert '<script>const positionJson = [{"b": 2}]</script>' in out
    assert 'old bundle' in out
    assert 'new bundle' not in out


def test_multiline_json_is_extracted():
    out = rewrap_report_html(_finlab_html(report_json='{"a":\n1}'), TEMPLATE)
    assert '{"a":\n1}' in out


def test_missing_data_returns_none():
    assert rewrap_report_html('<html>no data scripts</html>', TEMPLATE) is None


def test_missing_report_json_only_returns_none():
    html = '<html><script>const positionJson = []</script></html>'
    assert rewrap_report_html(html, TEMPLATE) is None


def test_template_without_placeholders_returns_none():
    assert rewrap_report_html(_finlab_html(), '<html>no placeholder</html>') is None


def test_extract_returns_both_json_blobs():
    assert extract_report_data(_finlab_html()) == ('{"a": 1}', '[{"b": 2}]')


def test_extract_missing_data_returns_none_pair():
    assert extract_report_data('<html>no data scripts</html>') == (None, None)


def test_extract_is_independent_of_line_position():
    """finlab 2.x 把資料挪離第 9/10 行後，抽取仍須成功。"""
    padded = '\n'.join(['<div>padding</div>'] * 40) + '\n' + _finlab_html()
    assert extract_report_data(padded) == ('{"a": 1}', '[{"b": 2}]')
