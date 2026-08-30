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
    ('alan_tw_strategy_efg95_simple.md', 'alan_tw_strategy_efg95_simple.py',
     'AlanTWStrategyEFG95Simple'),
    ('alan_tw_strategy_efg95_full.md', 'alan_tw_strategy_efg95_full.py', 'AlanTWStrategyEFG95Full'),
    ('alan_tw_strategy_ace_simple.md', 'alan_tw_strategy_ace_simple.py', 'AlanTWStrategyACESimple'),
    ('alan_tw_strategy_efg_not_start.md', 'alan_tw_strategy_efg_not_start.py',
     'AlanTWStrategyEFGNotStart'),
]

# 類別名 → 檔名（跨檔繼承時解析 configs / 屬性用）
CLASS_FILES = {
    'AlanTWStrategyEFGSimple': 'alan_tw_strategy_efg_simple.py',
    'AlanTWStrategyEFG95Simple': 'alan_tw_strategy_efg95_simple.py',
    'AlanTWStrategyEFG95Full': 'alan_tw_strategy_efg95_full.py',
    'AlanTWStrategyACESimple': 'alan_tw_strategy_ace_simple.py',
    'AlanTWStrategyEFGNotStart': 'alan_tw_strategy_efg_not_start.py',
}

# 基底類別的預設值（供子類未覆寫時回退）
BASE_DEFAULTS = {
    'AlanTWStrategyBase': {'entry_plus_di_min': 24, 'entry_minus_di_max': 21,
                           'sell_type': 'bare',
                           'entry_low_ratio_days': 15, 'entry_low_ratio_max': 1.32},
    'AlanTWStrategyNotStartBase': {'entry_plus_di_min': None, 'entry_minus_di_max': None,
                                   'entry_low_ratio_days': 15, 'entry_low_ratio_max': 1.32},
}

# 未發動型類別（不套用發動型專屬檢查，如加權收盤 MACD）
NOT_START_CLASSES = {'AlanTWStrategyEFGNotStart'}


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


def _find_configs(class_node, attrs):
    """取出 get_strategy_configs() 回傳的設定清單；未定義時回傳 None"""
    for stmt in class_node.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == 'get_strategy_configs':
            ret = stmt.body[-1]
            assert isinstance(ret, ast.Return)
            return _resolve(ret.value, attrs)
    return None


def _norm(text):
    """正規化全形減號與 ~ 前後空白，避免格式差異造成誤判"""
    return (text.replace('−', '-').replace('～', '~')
                .replace(' ~ ', '~').replace('% ~ ', '%~').replace(' ', ''))


def _pct(x):
    return f'{x:.0%}'


def _collect(py_file, cls_name, need_configs=True):
    """取得該策略的 (configs, attrs)。

    attrs 含繼承回退後的 DMI 與 sell_type；configs 若子類未定義
    get_strategy_configs（如 EFG95Simple 繼承 EFG95Full），沿繼承鏈往上找。
    """
    classes = _parse_classes(py_file)
    node = classes[cls_name]
    attrs = _class_attrs(node)
    configs = _find_configs(node, attrs)

    base_name = None
    for base in node.bases:
        if isinstance(base, ast.Name):
            base_name = base.id

    # 沿繼承鏈往上補 configs 與屬性預設值
    visited = set()
    while base_name and base_name not in visited:
        visited.add(base_name)
        if base_name in BASE_DEFAULTS:
            for key, val in BASE_DEFAULTS[base_name].items():
                attrs.setdefault(key, val)
            break
        if base_name not in CLASS_FILES:
            break
        parent_classes = _parse_classes(CLASS_FILES[base_name])
        parent = parent_classes[base_name]
        parent_attrs = _class_attrs(parent)
        for key, val in parent_attrs.items():
            attrs.setdefault(key, val)
        if configs is None:
            configs = _find_configs(parent, {**parent_attrs, **attrs})
        base_name = None
        for base in parent.bases:
            if isinstance(base, ast.Name):
                base_name = base.id

    if need_configs:
        assert configs is not None, f'{cls_name} 的繼承鏈上找不到 get_strategy_configs'
    return configs, attrs


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

    # 統一進場條件：收盤 ÷ 近 N 日最低價 <= 上限
    low_days, low_max = attrs.get('entry_low_ratio_days'), attrs.get('entry_low_ratio_max')
    if str(low_days) not in md or str(low_max) not in md:
        missing.append(f'低點乖離上限 近{low_days}日×{low_max}')

    # 統一技術指標：發動型 MACD 以加權收盤價自算
    if cls_name not in NOT_START_CLASSES and '加權收盤價' not in md:
        missing.append('MACD 加權收盤價說明')

    # 籌碼面：投信已加入買進條件，不應再寫「不列入」
    if '投信' not in md:
        missing.append('籌碼面投信條件')
    if '不列入' in md:
        missing.append('投信已列入買進條件，md 不應再寫「不列入」')

    sell = attrs.get('sell_type')
    if sell == 'simple':
        for token in ('-0.5%', '-3.5%'):
            if token not in md:
                missing.append(f'簡單出場的 {token}')
    if sell == 'full':
        for token in ('-3.5%', '-2.5%', '31'):
            if token not in md:
                missing.append(f'完整出場的 {token}')

    assert not missing, f'{md_file} 未涵蓋以下程式碼設定：\n  - ' + '\n  - '.join(missing)


@pytest.mark.parametrize('combo_cls,expected_sell,md_file,py_file', [
    ('AlanTWStrategyEFG95ACEFull', 'full',
     'alan_tw_strategy_efg95_ace_full.md', 'alan_tw_strategy_efg95_ace_full.py'),
    ('AlanTWStrategyEFG95ACESimple', 'simple',
     'alan_tw_strategy_efg95_ace_simple.md', 'alan_tw_strategy_efg95_ace_simple.py'),
])
def test_combo_doc_matches_code(combo_cls, expected_sell, md_file, py_file):
    """組合策略：兩個分量的參數與出場型別"""
    md = _norm(_read(os.path.join(DOCS_DIR, md_file)))

    classes = _parse_classes(py_file)
    combo = classes[combo_cls]
    component_names = []
    for stmt in combo.body:
        if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name) \
                and stmt.targets[0].id == 'COMPONENTS':
            component_names = [e.id for e in stmt.value.elts]
    assert component_names, f'{combo_cls} COMPONENTS 未定義'

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
            _, attrs = _collect(CLASS_FILES[comp], comp, need_configs=False)
            sell_types.add(attrs.get('sell_type'))

    # 兩分量的出場型別必須一致，且符合該組合版本的定調
    if sell_types != {expected_sell}:
        missing.append(f'{combo_cls} 兩分量 sell_type 應皆為 {expected_sell}，實際 {sell_types}')
    label = '完整出場' if expected_sell == 'full' else '簡單出場'
    if label not in md:
        missing.append(f'md 未提及 {combo_cls} 的{label}')

    # 「獨立版 vs 組合版」對照表：逐欄比對，避免兩個值寫反而未被發現
    raw = _read(os.path.join(DOCS_DIR, md_file))
    _, ace_attrs = _collect('alan_tw_strategy_ace_simple.py', 'AlanTWStrategyACESimple',
                            need_configs=False)
    combo_ace = ('_ACE_A90C90Full' if expected_sell == 'full' else '_ACE_A90C90Simple')
    combo_attrs = _class_attrs(classes[combo_ace])
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
