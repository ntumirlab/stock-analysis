"""策略程式碼與 docs/*.md 的一致性測試。

以 AST 靜態解析策略檔（不 import，因此不需要 finlab / ta-lib，可在 CI 執行），
取出各子策略的實際參數，確認對應的 md 文件有寫到相同的值。

策略參數一改而文件忘了更新時，這個測試會失敗並指出缺漏的項目。
"""

import ast
import os
import re

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STRATEGY_DIR = os.path.join(_ROOT, 'strategy_class')
DOCS_DIR = os.path.join(_ROOT, 'docs')

BIAS_KEYS = ['bias_5', 'bias_10', 'bias_20', 'bias_60', 'bias_120', 'bias_240']

# md 檔名 → (策略檔名, 類別名)
STRATEGIES = [
    ('alan_tw_strategy_efg_simple.md', 'alan_tw_strategy_efg_simple.py', 'AlanTWStrategyEFGSimple'),
    ('alan_tw_strategy_efg95_full.md', 'alan_tw_strategy_efg95_full.py', 'AlanTWStrategyEFG95Full'),
    ('alan_tw_strategy_ace_simple.md', 'alan_tw_strategy_ace_simple.py', 'AlanTWStrategyACESimple'),
    ('alan_tw_strategy_efg_not_start.md', 'alan_tw_strategy_efg_not_start.py',
     'AlanTWStrategyEFGNotStart'),
]

# 基底類別的預設值（供子類未覆寫時回退）
BASE_DEFAULTS = {
    'AlanTWStrategyBase': {'entry_plus_di_min': 24, 'entry_minus_di_max': 21,
                           'sell_type': 'bare'},
    'AlanTWStrategyNotStartBase': {'entry_plus_di_min': None, 'entry_minus_di_max': None},
}


def _read(path):
    with open(path, encoding='utf-8') as fh:
        return fh.read()


def _parse_classes(filename):
    """回傳 {類別名: ClassDef}"""
    tree = ast.parse(_read(os.path.join(STRATEGY_DIR, filename)))
    return {n.name: n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}


def _class_attrs(node):
    """類別層級的簡單常數屬性"""
    attrs = {}
    for stmt in node.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
            if isinstance(target, ast.Name) and isinstance(stmt.value, ast.Constant):
                attrs[target.id] = stmt.value.value
    return attrs


def _resolve(node, attrs):
    """把 AST 節點轉成 Python 值；self.X 以 attrs 查表解析"""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
            and node.value.id == 'self':
        if node.attr not in attrs:
            raise KeyError(f'無法解析 self.{node.attr}')
        return attrs[node.attr]
    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(_resolve(e, attrs) for e in node.elts)
    if isinstance(node, ast.Dict):
        return {_resolve(k, attrs): _resolve(v, attrs)
                for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_resolve(node.operand, attrs)
    raise TypeError(f'不支援的節點: {ast.dump(node)[:60]}')


def _configs(class_node, attrs):
    """取出 get_strategy_configs() 回傳的設定清單"""
    for stmt in class_node.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == 'get_strategy_configs':
            ret = stmt.body[-1]
            assert isinstance(ret, ast.Return)
            return _resolve(ret.value, attrs)
    raise AssertionError(f'{class_node.name} 未定義 get_strategy_configs')


def _norm(text):
    """正規化全形減號與 ~ 前後空白，避免格式差異造成誤判"""
    return (text.replace('−', '-').replace('～', '~')
                .replace(' ~ ', '~').replace('% ~ ', '%~').replace(' ', ''))


def _pct(x):
    return f'{x:.0%}'


def _collect(py_file, cls_name):
    """取得該策略的 (configs, attrs)，attrs 含繼承回退後的 DMI 與 sell_type"""
    classes = _parse_classes(py_file)
    node = classes[cls_name]
    attrs = _class_attrs(node)
    for base in node.bases:
        base_name = base.id if isinstance(base, ast.Name) else None
        for key, val in BASE_DEFAULTS.get(base_name, {}).items():
            attrs.setdefault(key, val)
    return _configs(node, attrs), attrs


@pytest.mark.parametrize('md_file,py_file,cls_name', STRATEGIES)
def test_strategy_doc_matches_code(md_file, py_file, cls_name):
    configs, attrs = _collect(py_file, cls_name)
    md = _norm(_read(os.path.join(DOCS_DIR, md_file)))

    missing = []
    for cfg in configs:
        name = cfg['name']
        for key in BIAS_KEYS:
            lo, hi = cfg['bias_ranges'][key]
            token = f'{_pct(lo)}~{_pct(hi)}'
            if token not in md:
                missing.append(f'子策略{name} {key} = {token}')
        for key in ('top_n', 'new_high_days'):
            if str(cfg[key]) not in md:
                missing.append(f'子策略{name} {key} = {cfg[key]}')
        if cfg.get('extra_new_high'):
            days, pct = cfg['extra_new_high']
            if str(days) not in md or _pct(pct) not in md:
                missing.append(f'子策略{name} extra_new_high {days}天×{_pct(pct)}')

    for key in ('entry_plus_di_min', 'entry_minus_di_max'):
        val = attrs.get(key)
        if val is not None and str(val) not in md:
            missing.append(f'{key} = {val}')

    sell = attrs.get('sell_type')
    if sell == 'simple' and '-0.5%' not in md:
        missing.append('簡單出場的 -0.5%')
    if sell == 'full':
        for token in ('-3.5%', '-2.5%', '31'):
            if token not in md:
                missing.append(f'完整出場的 {token}')

    assert not missing, f'{md_file} 未涵蓋以下程式碼設定：\n  - ' + '\n  - '.join(missing)


def test_combo_doc_matches_code():
    """組合策略：兩個分量的參數與出場型別"""
    md_file = 'alan_tw_strategy_efg95_ace.md'
    md = _norm(_read(os.path.join(DOCS_DIR, md_file)))

    classes = _parse_classes('alan_tw_strategy_efg95_ace.py')
    combo = classes['AlanTWStrategyEFG95ACE']
    component_names = []
    for stmt in combo.body:
        if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name) \
                and stmt.targets[0].id == 'COMPONENTS':
            component_names = [e.id for e in stmt.value.elts]
    assert component_names, 'COMPONENTS 未定義'

    missing = []
    sell_types = set()
    for comp in component_names:
        if comp not in md.replace('`', ''):
            missing.append(f'分量類別 {comp} 未出現')
        if comp in classes:                      # 定義在本檔內的分量
            attrs = _class_attrs(classes[comp])
            sell_types.add(attrs.get('sell_type'))
            for key in ('extra_high_pct_a', 'extra_high_pct_c'):
                if key in attrs and _pct(attrs[key]) not in md:
                    missing.append(f'{comp}.{key} = {_pct(attrs[key])}')
        else:                                    # 引用其他檔案的策略
            _, attrs = _collect('alan_tw_strategy_efg95_full.py', comp)
            sell_types.add(attrs.get('sell_type'))

    if len(sell_types) == 1 and '簡單出場' in md:
        missing.append('兩分量出場已統一，md 不應提到簡單出場')

    # 「獨立版 vs 組合版」對照表：逐欄比對，避免兩個值寫反而未被發現
    raw = _read(os.path.join(DOCS_DIR, md_file))
    _, ace_attrs = _collect('alan_tw_strategy_ace_simple.py', 'AlanTWStrategyACESimple')
    combo_attrs = _class_attrs(classes['_ACE_A90C90Full'])
    for key in ('extra_high_pct_a', 'extra_high_pct_c'):
        for line in raw.split('\n'):
            if key not in line or not line.strip().startswith('|'):
                continue
            cells = [c.strip().strip('*` ') for c in line.strip().strip('|').split('|')]
            if len(cells) < 3:
                continue
            for idx, (label, expected) in enumerate(
                    [('獨立版', ace_attrs[key]), ('組合版', combo_attrs[key])], start=1):
                nums = re.findall(r'0\.\d+', cells[idx])
                if not nums:
                    missing.append(f'{key} 的{label}欄未列出數值')
                elif round(float(nums[0]), 4) != round(expected, 4):
                    missing.append(f'{key} 的{label}欄寫 {nums[0]}，程式碼為 {expected}')

    assert not missing, f'{md_file} 與程式碼不符：\n  - ' + '\n  - '.join(missing)
