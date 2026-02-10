import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.font_manager as fm

# 三個策略的年度報酬率資料
data = {
    'Year': [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    '吳Peter策略選股': [0.30, 0.58, 0.16, 0.31, 0.41, 0.37, -0.23, 0.71, 0.36],
    '吳Peter策略選股_10檔': [0.25, 0.32, -0.05, 0.00, 0.05, 0.42, -0.22, 0.48, 0.12],
    '藏敖_2016': [0.66, 0.44, -0.04, 1.37, 0.85, 1.91, 0.09, 0.33, 0.20],
}

# 將資料轉換為 DataFrame
df = pd.DataFrame(data)

# 設置字體路徑
font_path = 'SourceHanSans-Regular.otf'
prop = fm.FontProperties(fname=font_path)

# 加載字體
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = prop.get_name()
plt.rcParams['axes.unicode_minus'] = False

# 繪製年度報酬率趨勢圖
plt.figure(figsize=(12, 6))

plt.plot(df['Year'], df['吳Peter策略選股'], label='吳Peter策略選股', marker='o')
plt.plot(df['Year'], df['吳Peter策略選股_10檔'], label='吳Peter策略選股_10檔', marker='o')
plt.plot(df['Year'], df['藏敖_2016'], label='藏敖_2016', marker='o')

plt.xlabel('年份', fontproperties=prop)
plt.ylabel('年度報酬率', fontproperties=prop)
plt.title('不同策略的年度報酬率趨勢', fontproperties=prop)
plt.legend(prop=prop)
plt.grid(True)
plt.show()
