#!/usr/bin/env python3
"""
Parameter Comparison Visualization Utility

為單一股票建立參數組合的績效比較表格（可排序，最佳參數高亮顯示）
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict


def create_param_comparison_chart(
    stock_id: str, param_results: List[Dict], output_path: str
) -> str:
    """
    建立參數比較表格（可排序的 HTML 表格）

    Args:
        stock_id: 股票代碼
        param_results: 參數測試結果列表，每個元素包含參數設定和績效指標
        output_path: HTML 輸出路徑

    Returns:
        str: 輸出檔案路徑
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 轉換為 DataFrame
    df = pd.DataFrame(param_results)

    # 相容新舊參數格式：
    # - 新格式: sar_signal_lag_min/sar_signal_lag_max
    # - 舊格式: sar_max_dots
    has_lag_window = {"sar_signal_lag_min", "sar_signal_lag_max"}.issubset(df.columns)

    # 選擇要顯示的欄位並排序
    if has_lag_window:
        display_columns = [
            "sar_signal_lag_min",
            "sar_signal_lag_max",
            "sar_accel",
            "sar_maximum",
            "macd_fast",
            "macd_slow",
            "macd_signal",
            "total_reward_amount",
            "annual_return",
            "max_drawdown",
            "sharpe_ratio",
            "total_trades",
        ]
    else:
        display_columns = [
            "sar_max_dots",
            "sar_accel",
            "sar_maximum",
            "macd_fast",
            "macd_slow",
            "macd_signal",
            "total_reward_amount",
            "annual_return",
            "max_drawdown",
            "sharpe_ratio",
            "total_trades",
        ]

    # 提取 SAR 參數到獨立欄位
    df["sar_accel"] = df["params"].apply(lambda x: x["sar_params"]["acceleration"])
    df["sar_maximum"] = df["params"].apply(lambda x: x["sar_params"]["maximum"])

    df = df[display_columns]

    # 依總報酬金額與夏普比率排序
    df = df.sort_values(
        by=["total_reward_amount", "sharpe_ratio"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)

    # 找出最佳值（用於高亮顯示）
    best_total_reward = df["total_reward_amount"].max()
    best_annual_return = df["annual_return"].max()
    best_sharpe = df["sharpe_ratio"].max() if df["sharpe_ratio"].notna().any() else None
    min_drawdown = df["max_drawdown"].max()  # 最大回檔越接近0越好（負值較小）

    # 建立表頭 HTML
    headers = {
        "sar_max_dots": "SAR Dots",
        "sar_signal_lag_min": "SAR Lag Min",
        "sar_signal_lag_max": "SAR Lag Max",
        "sar_accel": "SAR Accel",
        "sar_maximum": "SAR Max",
        "macd_fast": "MACD Fast",
        "macd_slow": "MACD Slow",
        "macd_signal": "MACD Signal",
        "total_reward_amount": "Total Reward",
        "annual_return": "Annual Return",
        "max_drawdown": "Max Drawdown",
        "sharpe_ratio": "Sharpe Ratio",
        "total_trades": "Total Trades",
    }

    thead_html = "<tr>\n"
    for col in display_columns:
        thead_html += f"                    <th>{headers[col]}</th>\n"
    thead_html += "                </tr>"

    # 建立表格內容 HTML（高亮最佳值）
    tbody_html = ""
    for idx, row in df.iterrows():
        # 判斷是否為最佳參數（第一行）
        is_best_row = idx == 0
        row_class = ' class="best-row"' if is_best_row else ""

        tbody_html += f"                <tr{row_class}>\n"

        for col in display_columns:
            value = row[col]

            # 格式化數值
            if pd.isna(value):
                formatted_value = "N/A"
                cell_class = ""
            elif col == "total_reward_amount":
                formatted_value = f"{value:.2f}"
                cell_class = (
                    ' class="highlight-best"'
                    if abs(value - best_total_reward) < 1e-6
                    else ""
                )
            elif col == "annual_return":
                formatted_value = f"{value:.2%}"
                # 高亮最佳年化報酬
                cell_class = (
                    ' class="highlight-best"'
                    if abs(value - best_annual_return) < 1e-6
                    else ""
                )
            elif col == "max_drawdown":
                formatted_value = f"{value:.2%}"
                # 高亮最小回檔
                cell_class = (
                    ' class="highlight-best"'
                    if abs(value - min_drawdown) < 1e-6
                    else ""
                )
            elif col == "sharpe_ratio":
                formatted_value = f"{value:.2f}"
                # 高亮最佳夏普比率
                cell_class = (
                    ' class="highlight-best"'
                    if best_sharpe and abs(value - best_sharpe) < 1e-6
                    else ""
                )
            elif col in [
                "sar_max_dots",
                "sar_signal_lag_min",
                "sar_signal_lag_max",
                "total_trades",
            ]:
                formatted_value = str(int(value))
                cell_class = ""
            elif col in ["sar_accel", "sar_maximum"]:
                formatted_value = f"{value:.2f}"
                cell_class = ""
            else:
                formatted_value = str(int(value))
                cell_class = ""

            tbody_html += (
                f"                    <td{cell_class}>{formatted_value}</td>\n"
            )

        tbody_html += "                </tr>\n"

    # 完整的 HTML 模板
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{stock_id} - Parameter Comparison</title>
    
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
        
        .info-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-size: 14px;
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
        
        /* 高亮最佳參數行 */
        table.dataTable tbody tr.best-row {{
            background-color: #e8f5e9 !important;
            font-weight: 600;
        }}
        
        table.dataTable tbody tr.best-row:hover {{
            background-color: #c8e6c9 !important;
        }}
        
        /* 高亮最佳指標值 */
        table.dataTable tbody td.highlight-best {{
            background-color: #fff9c4;
            font-weight: 700;
            color: #f57f17;
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
        <h1>{stock_id} - Parameter Optimization Results</h1>
        <p class="subtitle">Total Parameter Combinations Tested: {len(df)}</p>
        
        <div class="info-box">
            💡 <strong>Best Parameter:</strong> Row highlighted in green | 
            ⭐ <strong>Best Metrics:</strong> Individual best values highlighted in yellow
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
            $('#resultsTable').DataTable({{
                responsive: true,
                pageLength: 25,
                lengthMenu: [[10, 25, 50, -1], [10, 25, 50, "All"]],
                order: [[6, 'desc']], // Default sort by annual_return (index 6: sar_dots, sar_accel, sar_max, macd_fast, macd_slow, macd_signal, annual_return)
                columnDefs: [
                    {{
                        targets: '_all',
                        className: 'dt-body-left'
                    }},
                    {{
                        // SAR Accel and SAR Max columns - decimal formatting for sorting
                        targets: [1, 2], // sar_accel, sar_maximum
                        render: function(data, type, row) {{
                            if (type === 'sort' || type === 'type') {{
                                return parseFloat(data);
                            }}
                            return data;
                        }}
                    }},
                    {{
                        // Annual Return, Max Drawdown, Sharpe Ratio columns - treat N/A as 0
                        targets: [6, 7, 8], // annual_return, max_drawdown, sharpe_ratio
                        render: function(data, type, row) {{
                            if (type === 'sort' || type === 'type') {{
                                // For sorting, convert N/A to 0
                                if (data === 'N/A') {{
                                    return 0;
                                }}
                                // Remove % sign and convert to number
                                return parseFloat(data.replace('%', ''));
                            }}
                            // For display, show original value
                            return data;
                        }}
                    }}
                ],
                language: {{
                    search: "Filter:",
                    lengthMenu: "Show _MENU_ entries",
                    info: "Showing _START_ to _END_ of _TOTAL_ parameter combinations",
                    infoEmpty: "No parameters available",
                    infoFiltered: "(filtered from _MAX_ total combinations)"
                }}
            }});
        }});
    </script>
</body>
</html>"""

    # 寫入 HTML 檔案
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return str(output_path)
