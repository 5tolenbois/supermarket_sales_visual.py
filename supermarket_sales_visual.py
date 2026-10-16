import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========== 路径与全局字体配置 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "data", "超市购物数据集.csv")

# 设置中文字体，解决中文乱码与负号问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams["axes.unicode_minus"] = False

# ===================== 4.1 超市数据集介绍 =====================
"""
超市销售数据集
字段说明：
Date：销售日期
Branch：分店(A/B/C)
Product line：产品线
Payment：支付方式
Quantity：购买数量
Unit price：单价
Total：总销售额
gross income：毛收入
Rating：顾客评分
任务：使用Matplotlib完成多类型可视化：折线、饼图、柱状、条形、直方图、散点、气泡、热力图、雷达图、箱线图
"""

# ===================== 4.2 数据读取 =====================
data = pd.read_csv(csv_path)
print("="*70)
print("【数据集预览】")
print(data.head())
print("="*70)
print("【数据集基本信息】")
print(data.info())
print("="*70)
print("【数据集统计描述】")
print(data.describe())

# ===================== 4.3 绘制每日销量总和折线图 =====================
data1 = data.groupby('Date')['Quantity'].sum()
fig = plt.figure(figsize=(15, 8), dpi=100)
plt.plot(data1.index, data1.values, ".-g", alpha=0.8, markeredgecolor='r')
plt.xlabel('Date')
plt.xticks([])
plt.title('每日销量总和折线图')
plt.ylabel('销量')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "result", "daily_quantity_line.png"))
plt.show()

# ===================== 4.4 绘制不同支付方式数量占比饼图 =====================
data2 = data['Payment'].value_counts()
fig = plt.figure(figsize=(10, 8), dpi=100)
plt.pie(data2, labels=data2.index, autopct='%.2f%%')
plt.axis("equal")
plt.title("不同支付方式数量占比饼图")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "result", "payment_pie.png"))
plt.show()

# ===================== 4.5 绘制不同产品线数量占比饼图 =====================
data3 = data['Product line'].value_counts()
fig = plt.figure(figsize=(8, 8), dpi=100)
plt.pie(data3, labels=data3.index, autopct="%1.2f%%")
plt.axis("equal")
plt.title("不同产品线数量占比饼图")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "result", "product_line_pie.png"))
plt.show()

# ===================== 4.6 绘制不同产品线、支付方式的柱状图、条形图 =====================
data4 = data.groupby('Product line')['Quantity'].sum()
data5 = data.groupby('Payment')['Quantity'].sum()
fig = plt.figure(figsize=(20, 6), dpi=100)

# 子图1：支付方式销量柱状图
plt.subplot(121)
plt.bar(data5.index, data5.values, width=0.4, color='b', alpha=0.5)
plt.xlabel('Payment')
plt.title('不同支付方式销量柱状图')
plt.ylabel('销量')
plt.grid(linestyle="--", alpha=0.5)
for i, v in enumerate(data5.values):
    plt.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=10)

# 子图2：产品线销量条形图
plt.subplot(122)
plt.barh(data4.index, data4.values, height=0.5, color="g", alpha=0.5)
plt.xlabel('销量')
plt.title('不同产品线销量条形图')
plt.yticks(rotation=30)
plt.grid(linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "result", "product_payment_bar.png"))
plt.show()

# ===================== 4.7 绘制每日总销量直方图 =====================
data_hist = data.groupby('Date')['Quantity'].sum()
fig = plt.figure(figsize=(8, 6), dpi=100)
plt.hist(data_hist, color='green', alpha=0.7, bins=20)
plt.title('每日总销量直方图')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "result", "daily_quantity_hist.png"))
plt.show()

# ===================== 4.8 绘制不同分店每日总收入与单价、评分、销量关系的散点图 =====================
data6 = data.groupby(['Branch', 'Date']).sum(numeric_only=True)
df1 = data6.loc['A']
df2 = data6.loc['B']
df3 = data6.loc['C']

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), dpi=100)
ax1.scatter(df1['gross income'], df1['Unit price'], color='b')
ax1.set_title('A店总收入与单价关系散点图')
ax1.set_xlabel("gross income")
ax1.set_ylabel("Unit price")

ax2.scatter(df2['gross income'], df2['Rating'], color='g')
ax2.set_title('B店总收入与评分关系散点图')
ax2.set_xlabel("gross income")
ax2.set_ylabel("Rating")

ax3.scatter(df3['gross income'], df3['Quantity'], color='r')
ax3.set_title('C店总收入与销量关系散点图')
ax3.set_xlabel("gross income")
ax3.set_ylabel("Quantity")

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "result", "branch_scatter.png"))
plt.show()

# ===================== 4.9 绘制A店销售总额、销量、单价关系气泡图 =====================
data7 = data.groupby(['Branch', 'Date']).mean(numeric_only=True)
df7 = data7.loc['A']
fig = plt.figure(figsize=(10, 6), dpi=100)
scatter = plt.scatter(df7['Total'], df7['Quantity'],
                      s=np.power(df7['Quantity'].values * 4, 2),
                      c=df7['Unit price'].values,
                      cmap=plt.cm.RdYlBu, marker='o')
plt.colorbar(scatter, label="Unit price")
plt.title('A店销售总额、销量、单价关系气泡图')
plt.xlabel("Total 销售总额")
plt.ylabel("Quantity 销量")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "result", "shopA_bubble.png"))
plt.show()

# ===================== 4.10 绘制不同产品线使用不同支付方式的顾客评分热力图 =====================
heat_data = data.groupby(['Payment', 'Product line'])[['Rating']].mean()
da1 = heat_data.loc['Cash'][['Rating']]
da2 = heat_data.loc['Credit card'][['Rating']]
da3 = heat_data.loc['Ewallet'][['Rating']]
df_heat = pd.concat([da1, da2, da3], axis=1)

fig = plt.figure(figsize=(12, 6), dpi=100)
yLabel = ['Electronic accessories', 'Fashion accessories', 'Food and beverages',
          'Health and beauty', 'Home and lifestyle', 'Sports and travel']
xLabel = ['Cash', 'Credit card', 'Ewallet']
plt.yticks(range(len(yLabel)), yLabel)
plt.xticks(range(len(xLabel)), xLabel)
im = plt.imshow(df_heat, cmap=plt.cm.Blues)
plt.colorbar(im)
plt.title("不同产品线不同支付方式的评分热力图")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "result", "rating_heatmap.png"))
plt.show()

# ===================== 4.11 绘制超市指标雷达图 =====================
fig = plt.figure(figsize=(8, 8), dpi=100)
x = ['price', 'quality', 'popularity', 'location', 'rating', 'price']
y = [9, 4, 8, 10, 6, 9]
theta = np.linspace(0, np.pi * 2, 6)
plt.polar(theta, y, linestyle='--', marker='*', markerfacecolor='r', markersize=10)
plt.xticks(theta, x)
plt.title("超市指标雷达图")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "result", "radar_chart.png"))
plt.show()

# ===================== 4.12 绘制不同商店销量与销售总额对比图 =====================
data11 = data.groupby('Branch')[['Total', 'Quantity']].sum(numeric_only=True)
fig = plt.figure(figsize=(10, 6), dpi=100)
ax1 = fig.add_subplot(111)
ax1.plot(data11.index, data11[['Total']], 'or-', label='销售总额')
ax1.set_ylim(106000, 110600)
ax1.set_ylabel('销售总额', size=12)
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.set_ylim(1800, 1900)
plt.bar(data11.index, data11['Quantity'], label='销量', alpha=0.4, color='g', width=0.3)
ax2.set_ylabel('销量', size=12)
ax2.legend(loc='upper right')
plt.title('不同商店销量与销售总额对比图')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "result", "branch_total_quantity.png"))
plt.show()

# ===================== 4.13 不同产品线每日平均总销售额箱线图 =====================
d1 = data.groupby(['Product line', 'Date'])['Total'].mean()
k1 = d1.loc['Fashion accessories']
k2 = d1.loc['Food and beverages']
k3 = d1.loc['Electronic accessories']
k4 = d1.loc['Sports and travel']
k5 = d1.loc['Home and lifestyle']
k6 = d1.loc['Health and beauty']

fig = plt.figure(figsize=(10, 6), dpi=100)
x_data = [k1, k2, k3, k4, k5, k6]
label = ['Fashion accessories', 'Food and beverages', 'Electronic accessories',
         'Sports and travel', 'Home and lifestyle', 'Health and beauty']
plt.boxplot(x_data, notch=True, sym='o', patch_artist=True, labels=label)
plt.xticks(rotation=20)
plt.title('不同产品线每日平均总销售额箱线图')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "result", "product_boxplot.png"))
plt.show()

print("\n超市销售Matplotlib可视化全部绘图执行完成！图片保存到result文件夹")
