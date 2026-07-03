"""將 finlab 報告 HTML 的資料重新包進固定模板 — 純字串邏輯，不 import finlab。

finlab 2.x 改版了報告前端外觀；為了讓廠商看到的報告維持 1.x 舊版排版，
把 finlab 產出 HTML 中的 reportJson / positionJson 抽出，填回
scripts/extract_report_template.py 抽出的舊版模板。
"""
import re

_REPORT_RE = re.compile(r'<script>const reportJson = (.*?)</script>', re.DOTALL)
_POSITION_RE = re.compile(r'<script>const positionJson = (.*?)</script>', re.DOTALL)

REPORT_PLACEHOLDER = '{{REPORT_JSON}}'
POSITION_PLACEHOLDER = '{{POSITION_JSON}}'


def rewrap_report_html(report_html: str, template_html: str) -> str | None:
    """從 finlab 產出的 report_html 抽出資料，填入模板的佔位符。

    回傳包裝後的 HTML；report_html 抽不到資料、或模板缺佔位符時
    回傳 None（呼叫端應保留 finlab 原始輸出）。
    """
    m_report = _REPORT_RE.search(report_html)
    m_position = _POSITION_RE.search(report_html)
    if not (m_report and m_position):
        return None
    if REPORT_PLACEHOLDER not in template_html or POSITION_PLACEHOLDER not in template_html:
        return None
    return (template_html
            .replace(REPORT_PLACEHOLDER, m_report.group(1))
            .replace(POSITION_PLACEHOLDER, m_position.group(1)))
