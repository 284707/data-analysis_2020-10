import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pyecharts.charts import Pie
from pyecharts.charts import Map
from pyecharts import options as opts
from pyecharts.charts import Bar3D
from multiprocessing import Pool
from statsmodels.formula.api import ols


"""
sns.图名(x='X轴 列名', y='Y轴 列名', data=原始数据df对象)
sns.图名(x='X轴 列名', y='Y轴 列名', hue='分组绘图参数', data=原始数据df对象)
sns.图名(x=np.array, y=np.array[, ...])
"""

# 导入数据与初始设置
path = './lagou.xlsx'
file = pd.read_excel(path)
pd.set_option("display.max_column", None)
pd.set_option("display.max_row", None)
plt.rcParams['font.sans-serif'] = ['SimHei']
# 选取需要分析的列
col = ['city', 'companyLabelList', 'positionName', 'companySize', 'district', 'education',
       'hitags', 'jobNature', 'salary', 'workYear', 'job_detail']
"""
数据预处理，首先对非实习数据分析师进行分析
"""
# 去重
file = file[col].drop_duplicates()


def pretreatment(file):
    # 结合jobNature与positionName去除实习生岗位
    cond_0 = file['jobNature'] == '全职'
    cond_1 = file['positionName'].str.contains('数据分析')
    cond_2 = file['positionName'].str.contains('实习') == False
    file = file[cond_0 & cond_1 & cond_2].copy()
    # 分析完成后去除此列
    file.drop(['jobNature', 'positionName'], axis=1, inplace=True)
    file.reset_index(drop=True, inplace=True)
    return file


def demand(file):
    """
    各个城市岗位需求量分析
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.countplot(y='city', order=file['city'].value_counts().index, data=file)
    plt.title('各个城市岗位需求量分析')
    plt.box(False)
    ax.xaxis.grid(which='both', linewidth=0.5, color='#3c7f99')
    plt.show()


def welfare(file):
    """
    探究数据分析岗最常有的福利
    """
    welfare_dic = {}
    # 实现各福利的统计
    for a in range(0, file.shape[0]):
        for b in eval(file['companyLabelList'][a]):
            welfare_dic[b] = welfare_dic.get(b, 0) + 1
    welfare_dic = sorted(
        welfare_dic.items(),
        key=lambda item: item[1],
        reverse=True)
    for i in range(0, 15):
        print([welfare_dic[i]])
    pie = (
        Pie()
        .add('', welfare_dic[0:10], center=['50%', '50%'],
             radius=['50%', '75%'])
        .set_global_opts(title_opts=opts.TitleOpts(title=''),
                         legend_opts=opts.LegendOpts(
            is_show=False
        )).set_series_opts(label_opts=opts.LabelOpts(formatter='{b}: {d}%')))
    pie.render('welfare.html')


def address(file, city):
    """
    各个城市各个区的岗位需求量
    """
    cond = file['city'].str.contains(city)
    df = file[cond]
    index = df['district'].value_counts().index.tolist()
    value = df['district'].value_counts().tolist()
    g = Map()
    data_pair = list(zip(index, value))
    g.add('', data_pair, maptype=city)
    g.set_series_opts(label_opts=opts.LabelOpts(is_show=True))
    g.set_global_opts(title_opts=opts.TitleOpts(title=str(city) + "各区块数据分析岗位分布", pos_left='center'),
                      visualmap_opts=opts.VisualMapOpts(min_=min(value), max_=max(value)))
    g.render('address.html')


def city_num(file):
    city_list = [
        '北京',
        '上海',
        '深圳',
        '广州',
        '杭州',
        '成都',
        '武汉',
        '苏州',
        '南京',
        '厦门',
        '西安',
        '长沙',
        '天津']
    for city in city_list:
        file[city] = file["city"].map(lambda x: 1 if (x == city) else 0)
    return file


def city_num_1(file):
    city_list = [
        '北京',
        '上海',
        '深圳',
        '广州',
        '杭州',
        '成都',
        '武汉',
        '苏州',
        '南京',
        '厦门',
        '西安',
        '长沙',
        '天津']
    num = 0
    for city in city_list:
        if file[city] == 1:
            file['city_num'] = num
        num += 1
    return file


def city_salary(file):
    # x轴：城市，y轴：薪酬，z轴：数量
    dic = {}
    file['salary'] = file["salary"].str.lower().str.extract(
        r'(\d+)[k]-(\d+)k').applymap(lambda x: int(x)).mean(axis=1)
    list1 = list(zip(file['city_num'].tolist(), file['salary'].tolist()))
    y_list = []
    tuple_data = []
    for i in range(0, 67):
        y_list.append(i / 2)
    x_list = [
        '北京',
        '上海',
        '深圳',
        '广州',
        '杭州',
        '成都',
        '武汉',
        '苏州',
        '南京',
        '厦门',
        '西安',
        '长沙',
        '天津']
    for i in range(0, len(list1)):
        dic[list1[i]] = dic.get(list1[i], 0) + 1
    for i in range(0, len(dic)):
        a = list(
            dic.keys())[i][0], list(
            dic.keys())[i][1], list(
            dic.values())[i]
        tuple_data.append(a)
    c = (
        Bar3D().add(
            "",
            tuple_data,
            xaxis3d_opts=opts.Axis3DOpts(
                x_list, type_="category", interval=0),
            yaxis3d_opts=opts.Axis3DOpts(
                y_list, type_="category", interval=4),
            zaxis3d_opts=opts.Axis3DOpts(type_="value"),
            grid3d_opts=opts.Grid3DOpts(width=200, height=100, depth=150))
        .set_global_opts(
            visualmap_opts=opts.VisualMapOpts(max_=50),
            title_opts=opts.TitleOpts(title="各城市薪酬3D柱状图"),
        ).render("city_salary.html")
    )


def education_salary(file):
    cond0 = file['education'].str.contains('大专')
    cond1 = file['education'].str.contains('本科')
    cond2 = file['education'].str.contains('硕士')
    cond3 = file['education'].str.contains('博士')
    file['salary'] = file["salary"].str.lower().str.extract(
        r'(\d+)[k]-(\d+)k').applymap(lambda x: int(x)).mean(axis=1)
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.text(x=0.04, y=0.90, s='——————不同学历要求的薪资分布——————', fontsize=32,
             weight='bold', color='white', backgroundcolor='#c5b783')
    sns.kdeplot(file[cond0]['salary'], shade=True, label='大专')
    sns.kdeplot(file[cond1]['salary'], shade=True, label='本科')
    sns.kdeplot(file[cond2 | cond3]['salary'], shade=True, label='研究生')
    plt.box(False)
    plt.xticks(np.arange(0, 61, 10), [str(i) + 'k' for i in range(0, 61, 10)])
    plt.yticks([])
    plt.legend(fontsize='xx-large', fancybox=None)
    plt.show()


def workyear_num(file):
    workyear_list = ['应届毕业生', '1年以下', '1-3年', '3-5年', '5-10年', '10年以上', '不限']
    for year in workyear_list:
        file[year] = file['workYear'].map(lambda x: 1 if (x == year) else 0)
    return file


def workyear_num_1(file):
    workyear_list = ['应届毕业生', '1年以下', '1-3年', '3-5年', '5-10年', '10年以上', '不限']
    workyear = [0, 0.5, 2, 4, 7.5, 12.5, '不限']
    i = 0
    # 明天改改
    for year in workyear_list:
        if file[year] == 1:
            file['workyear'] = workyear[i]
        i += 1
    return file


def education_workYear_salary(file):
    # 线性回归
    plt.rcParams['font.sans-serif'] = ['SimHei']
    cond0 = file['workyear'] != '不限'
    file = file[cond0]
    file['salary'] = file["salary"].str.lower().str.extract(
        r'(\d+)[k]-(\d+)k').applymap(lambda x: int(x)).mean(axis=1)
    formula = 'salary~workyear+education'
    model = ols(formula, data=file).fit()
    print(model.summary2())
    # 散点图
    cond1 = file['education'] != '不限'
    file = file[cond1]
    # Series转成float
    file['workyear'] = np.array(file['workyear'], dtype='float')
    file['salary'] = np.array(file['salary'], dtype='float')
    sns.lmplot(x='workyear', y='salary', hue='education', data=file)
    plt.xticks([i for i in range(0, 14)], [str(i) + '年' for i in range(0, 14)])
    plt.yticks([i * 5 for i in range(0, 15)],
               [str(i * 5) + 'k' for i in range(0, 15)])
    plt.xlabel('workyear')
    plt.ylabel('salary')
    plt.title('不同学历随着工龄的增长的薪酬变化')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    pool = Pool(8)
    file = pretreatment(file)
    demand(file) if input('各城市需求量分析?') == str(1) else ''
    welfare(file) if input('福利分析?') == str(1) else ''
    address(file, input("你想要分析哪所城市？(北京, 上海, 深圳, 广州, 杭州, 成都, 武汉, 苏州, 南京, 厦门, 西安, 长沙, 天津)")
            ) if input('地点分析?') == str(1) else ''
    city_salary(city_num(file).apply(city_num_1, axis=1)
                ) if input('城市薪酬分析？') == str(1) else ''
    education_workYear_salary(
        workyear_num(file).apply(
            workyear_num_1,
            axis=1)) if input('学历、工龄对薪资的影响?') == str(1) else ''
