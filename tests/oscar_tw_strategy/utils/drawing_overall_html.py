#!/usr/bin/env python3
"""
Drawing Overall HTML Utility

將回測結果 DataFrame 轉換為可排序的互動式 HTML 表格
"""

import pandas as pd
from pathlib import Path
from typing import Optional


def dataframe_to_sortable_html(
    df: pd.DataFrame,
    output_path: str,
    title: str = "Backtest Results",
    is_grid_search_result: bool = False,
    parameter_meanings: Optional[dict] = None,
    tested_params: Optional[dict] = None,
) -> str:
    """
    將 DataFrame 轉換為可排序的 HTML 表格
    
    Args:
        df: 包含回測結果的 DataFrame
        output_path: HTML 輸出路徑
        title: 表格標題
        is_grid_search_result: 是否為網格搜尋結果
        parameter_meanings: 參數欄位說明
        tested_params: 測試參數範圍
        
    Returns:
        str: 輸出檔案路徑
    """
    output_path = Path(output_path)
    
    # 取得欄位名稱
    headers = df.columns.tolist()
    
    # 建立表頭 HTML
    thead_html = '<tr>\n'
    for header in headers:
        thead_html += f'                    <th>{header}</th>\n'
    thead_html += '                </tr>'
    
    # 建立表格內容 HTML
    tbody_html = ''
    for _, row in df.iterrows():
        tbody_html += '                <tr>\n'
        for header in headers:
            value = row[header]
            # 格式化數值
            if pd.isna(value):
                formatted_value = 'N/A'
            elif isinstance(value, float):
                if abs(value) < 1 and abs(value) > 0.001:
                    formatted_value = f'{value:.6f}'
                else:
                    formatted_value = f'{value:.2f}'
            else:
                formatted_value = str(value)
            tbody_html += f'                    <td>{formatted_value}</td>\n'
        tbody_html += '                </tr>\n'

    # 參數說明與測試範圍區塊（僅在有資料時顯示）
    info_panels_html = ''
    if parameter_meanings or tested_params:
        meaning_rows = ''
        tested_rows = ''

        if parameter_meanings:
            for key, desc in parameter_meanings.items():
                meaning_rows += (
                    '                        <tr>'
                    f'<td class="info-key">{key}</td>'
                    f'<td>{desc}</td>'
                    '</tr>\n'
                )

        if tested_params:
            for key, value in tested_params.items():
                tested_rows += (
                    '                        <tr>'
                    f'<td class="info-key">{key}</td>'
                    f'<td>{value}</td>'
                    '</tr>\n'
                )

        info_panels_html = f'''
        <div class="info-panels">
            <div class="info-card">
                <h2>Parameter Meaning</h2>
                <table class="info-table">
                    <tbody>
{meaning_rows if meaning_rows else '                        <tr><td colspan="2">N/A</td></tr>'}
                    </tbody>
                </table>
            </div>
            <div class="info-card">
                <h2>Tested Params</h2>
                <table class="info-table">
                    <tbody>
{tested_rows if tested_rows else '                        <tr><td colspan="2">N/A</td></tr>'}
                    </tbody>
                </table>
            </div>
        </div>
        '''
    
    # 完整的 HTML 模板
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    
    <!-- DataTables CSS -->
    <link rel="stylesheet" href="https://cdn.datatables.net/1.13.7/css/jquery.dataTables.min.css">
    <link rel="stylesheet" href="https://cdn.datatables.net/responsive/2.5.0/css/responsive.dataTables.min.css">
    
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        
        .container {{
            max-width: 100%;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        h1 {{
            color: #333;
            margin-bottom: 10px;
            font-size: 24px;
        }}
        
        .subtitle {{
            color: #666;
            margin-bottom: 30px;
            font-size: 14px;
        }}
        
        .summary-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}

        .info-panels {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}

        .info-card {{
            border: 1px solid #e6e6e6;
            border-radius: 8px;
            padding: 16px;
            background: #fafafa;
        }}

        .info-card h2 {{
            margin: 0 0 10px 0;
            font-size: 18px;
            color: #333;
        }}

        .info-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}

        .info-table td {{
            border-bottom: 1px solid #ececec;
            padding: 8px 6px;
            vertical-align: top;
        }}

        .info-key {{
            white-space: nowrap;
            width: 36%;
            font-weight: 600;
            color: #444;
        }}
        
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .stat-card.positive {{
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }}
        
        .stat-card.negative {{
            background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        }}
        
        .stat-label {{
            font-size: 12px;
            opacity: 0.9;
            margin-bottom: 5px;
        }}
        
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
        }}
        
        table.dataTable {{
            width: 100% !important;
            border-collapse: collapse;
        }}
        
        table.dataTable thead th {{
            background-color: #4CAF50;
            color: white;
            padding: 12px 8px;
            text-align: left;
            font-weight: 600;
            cursor: pointer;
        }}
        
        table.dataTable thead th:hover {{
            background-color: #45a049;
        }}
        
        table.dataTable tbody td {{
            padding: 10px 8px;
            border-bottom: 1px solid #ddd;
        }}
        
        table.dataTable tbody tr:hover {{
            background-color: #f5f5f5;
        }}
        
        table.dataTable.stripe tbody tr.odd {{
            background-color: #f9f9f9;
        }}
        
        .dataTables_wrapper .dataTables_length,
        .dataTables_wrapper .dataTables_filter {{
            padding: 10px 0;
        }}
        
        .dataTables_wrapper .dataTables_filter input {{
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 5px 10px;
            margin-left: 8px;
        }}
        
        .dataTables_wrapper .dataTables_length select {{
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 5px;
            margin: 0 5px;
        }}
        
        .dataTables_wrapper .dataTables_info {{
            padding-top: 15px;
            font-size: 13px;
            color: #666;
        }}
        
        .dataTables_wrapper .dataTables_paginate {{
            padding-top: 15px;
        }}
        
        td {{
            font-variant-numeric: tabular-nums;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <p class="subtitle">{'Grid Search Result' if is_grid_search_result else 'Backtest Result'} | Total Stocks Tested: {len(df)}</p>

{info_panels_html}
        
        <div class="summary-stats" id="summaryStats">
            <!-- Will be populated by JavaScript -->
        </div>
        
        <table id="resultsTable" class="display responsive nowrap" style="width:100%">
            <thead>
{thead_html}
            </thead>
            <tbody>
{tbody_html}
            </tbody>
        </table>
    </div>
    
    <!-- jQuery -->
    <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
    
    <!-- DataTables JS -->
    <script src="https://cdn.datatables.net/1.13.7/js/jquery.dataTables.min.js"></script>
    <script src="https://cdn.datatables.net/responsive/2.5.0/js/dataTables.responsive.min.js"></script>
    
    <script>
        $(document).ready(function() {{
            // Initialize DataTable
            var table = $('#resultsTable').DataTable({{
                responsive: true,
                pageLength: 25,
                lengthMenu: [[10, 25, 50, 100, -1], [10, 25, 50, 100, "All"]],
                order: [[1, 'desc']], // Default sort by annual_return
                columnDefs: [
                    {{
                        targets: '_all',
                        className: 'dt-body-left',
                        render: function(data, type, row) {{
                            if (type === 'sort' || type === 'type') {{
                                // For sorting, treat N/A as 0
                                if (data === 'N/A') {{
                                    return 0;
                                }}
                                // Try to parse as number
                                var num = parseFloat(data);
                                return isNaN(num) ? data : num;
                            }}
                            // For display, show original value
                            return data;
                        }}
                    }}
                ],
                language: {{
                    search: "Filter:",
                    lengthMenu: "Show _MENU_ entries",
                    info: "Showing _START_ to _END_ of _TOTAL_ stocks",
                    infoEmpty: "No stocks available",
                    infoFiltered: "(filtered from _MAX_ total stocks)"
                }}
            }});
            
            // Calculate and display summary statistics
            var data = table.rows().data();
            var totalStocks = data.length;
            var avgReturn = 0;
            var avgSharpe = 0;
            var positiveStocks = 0;
            var validSharpe = 0;
            
            for (var i = 0; i < totalStocks; i++) {{
                var row = data[i];
                var annualReturn = parseFloat(row[1]); // Assuming annual_return is column 1
                var sharpe = parseFloat(row[3]); // Assuming sharpe_ratio is column 3
                
                if (!isNaN(annualReturn)) {{
                    avgReturn += annualReturn;
                    if (annualReturn > 0) positiveStocks++;
                }}
                if (!isNaN(sharpe)) {{
                    avgSharpe += sharpe;
                    validSharpe++;
                }}
            }}
            
            avgReturn = totalStocks > 0 ? avgReturn / totalStocks : 0;
            avgSharpe = validSharpe > 0 ? avgSharpe / validSharpe : 0;
            var winRate = totalStocks > 0 ? (positiveStocks / totalStocks * 100) : 0;
            
            // Display summary cards
            var summaryHtml = `
                <div class="stat-card">
                    <div class="stat-label">Total Stocks</div>
                    <div class="stat-value">${{totalStocks}}</div>
                </div>
                <div class="stat-card ${{avgReturn > 0 ? 'positive' : 'negative'}}">
                    <div class="stat-label">Avg Annual Return</div>
                    <div class="stat-value">${{(avgReturn * 100).toFixed(2)}}%</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Avg Sharpe Ratio</div>
                    <div class="stat-value">${{avgSharpe.toFixed(2)}}</div>
                </div>
                <div class="stat-card ${{winRate >= 50 ? 'positive' : 'negative'}}">
                    <div class="stat-label">Positive Returns</div>
                    <div class="stat-value">${{positiveStocks}} (${{winRate.toFixed(1)}}%)</div>
                </div>
            `;
            
            $('#summaryStats').html(summaryHtml);
        }});
    </script>
</body>
</html>'''
    
    # 寫入 HTML 檔案
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    return str(output_path)
