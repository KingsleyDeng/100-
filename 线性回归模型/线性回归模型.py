import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# 正常显示中文标签
mpl.rcParams['font.sans-serif'] = ['SimHei']

# 数据源路径
filepath = '广告费与商品销量.xlsx'

# 读取Excel文件
df = pd.read_excel(filepath, index_col='日期')

# 定义画图用的数据
df_group = df.groupby('广告费').mean()
x = np.array(df_group.index).reshape(-1, 1)
y = np.array(df_group.values)

# 用管道的方式调用算法 以便在线性回归拓展为多项式回归
poly_reg = Pipeline([
    ('ploy', PolynomialFeatures(degree=1)),
    ('lin_reg', LinearRegression())
])

# 拟合
poly_reg.fit(x, y)

# 斜率
coef = poly_reg.steps[1][1].coef_
# 截距
intercept = poly_reg.steps[1][1].intercept_
# 评分
score = poly_reg.score(x, y)

# 使用「面向对象」的方法画图，定义图片的大小
fig, ax = plt.subplots(figsize=(8, 6))

# 标注公式
formula = r'$y = ' + '%.2f' % coef[0][1] + 'x' + '%+.2f$' % intercept[0] + '，' + r'${R}^2 = ' + '%.5f$' % score

# 设置标题
ax.set_title('广告费每增加1万元，商品销量增加' + '%.0f' % (coef[0][1]) + '个\n'+formula, loc='left', size=26)

# 画气泡图
ax.scatter(x, y, color='#00589F', marker='.', s=100, zorder=1)

# 绘制预测线
y2 = poly_reg.predict(x)
ax.plot(x, y2, '-', c='#5D9BCF', zorder=2)

# 隐藏边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 设置坐标标签字体大小和颜色
ax.tick_params(labelsize=16)

# 设置坐标轴的标题
ax.text(ax.get_xlim()[0]-1.2, ax.get_ylim()[1]/2, '商\n品\n销\n量', va='center', fontsize=16)
ax.text(ax.get_xlim()[1]/2, ax.get_ylim()[0]-100, '广告费', ha='center', fontsize=16)

plt.show()










