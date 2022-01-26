import random
import matplotlib as matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

A = 5  # x方向有A个货位
B = 4  # y方向有B个货架
C = 6  # z方向有C层
l = 10  # 货位的长
r = 1  # 货位的宽
h = 20  # 货位的高
vx = 1  # 叉车的x速度
vy = 0.5  # 叉车的y速度
vz = 2  # 叉车的z速度
N = 20  # 货物的数量
L = 20  # 质量上限
m = 400  # 种群数量
cross_rate = 0.7
mutation = 0.01
w1 = 0.7  # 第一个规则的权值
w2 = 0.3  # 第二个规则的权值
num = 200  # 迭代次数


class GA():

    # input:
    #     nums: m * n  n是种群数，m是变量个数
    #     bound:n * 2  [(min, nax), (min, max), (min, max),...]
    #     DNA_SIZE is DNA大小
    def __init__(self, nums, func, cross_rate=0.8, mutation=0.003):
        nums = np.array(nums)
        # bound = np.array(bound)
        # self.bound = bound
        # if nums.shape[1] != bound.shape[0]:
        #     raise Exception(f'范围的数量与变量的数量不一致, 您有{nums.shape[1]}个变量，却有{bound.shape[0]}个范围')
        #
        # for var in nums:
        #     for index, var_curr in enumerate(var):
        #         if var_curr < bound[index][0] or var_curr > bound[index][1]:
        #             raise Exception(f'{var_curr}不在取值范围内')
        #
        # for min_bound, max_bound in bound:
        #     if max_bound < min_bound:
        #         raise Exception(f'抱歉，({min_bound}, {max_bound})不是合格的取值范围')

        # 所有变量的最小值和最大值
        # var_len为所有变量的取值范围大小
        # min_nums, max_nums = np.array(list(zip(*bound)))
        # self.var_len = var_len = max_nums-min_nums
        # bits = np.ceil(np.log2(var_len+1))

        # POP_SIZE为进化的种群数
        self.POP_SIZE = len(nums)
        m = nums.shape[0]
        n = nums.shape[1]
        POP = np.zeros((m, n // 3))
        for i in range(m):
            for j in range(n // 3):
                # 编码方式：
                POP[i, j] = nums[i, j] + nums[i, j + n // 3] * A * C + nums[i, j + 2 * n // 3] * A
        self.POP = POP
        # 用于后面重置（reset）
        self.copy_POP = POP.copy()
        self.cross_rate = cross_rate
        self.mutation = mutation
        self.func = func

    # save args对象保留参数：
    #        bound                取值范围
    #        var_len              取值范围大小
    #        POP_SIZE             种群大小
    #        POP                  编码后的种群[[[1,0,1,...],[1,1,0,...],...]]
    #                             一维元素是各个种群，二维元素是各个DNA[1,0,1,0]，三维元素是碱基对1/0
    #        copy_POP             复制的种群，用于重置
    #        cross_rate           染色体交换概率
    #        mutation             基因突变概率
    #        func                 适应度函数
    # 将编码后的DNA翻译回来（解码）
    def translateDNA(self):
        POP = self.POP
        m = POP.shape[0]
        n = POP.shape[1]
        vector = np.zeros((m, 3 * n))
        for i in range(m):
            for j in range(n):
                vector[i, j + n] = POP[i, j] // (A * C)
                copy = vector[i, j + n]
                vector[i, j + 2 * n] = (POP[i, j] - copy * (A * C)) // A
                cop = vector[i, j + 2 * n]
                vector[i, j] = POP[i, j] - cop * A - copy * A * C
        return vector

    # 得到适应度
    def get_fitness(self):
        result = 1 / self.func(self.translateDNA())
        return result

    # 自然选择
    def select(self):
        fitness = self.get_fitness()[0]
        self.POP = self.POP[np.random.choice(np.arange(self.POP.shape[0]), size=self.POP.shape[0], replace=True,
                                             p=fitness / np.sum(fitness))]

    # 染色体交叉
    def crossover(self):
        for people in self.POP:
            if np.random.rand() < self.cross_rate:
                i_ = np.random.randint(0, self.POP.shape[0], size=1)

                dd = np.random.randint(0, len(people), size=random.randint(0, len(people)))
                for d in dd:
                    if self.POP[i_, d] not in people:
                        people[d] = self.POP[i_, d]

    # 基因变异
    def mutate(self):
        for people in self.POP:
            for point in range(len(people)):
                if np.random.rand() < self.mutation:
                    x = np.random.randint(0, A * B * C)
                    people[point] = x if x not in people else people[point]

    # 进化
    def evolution(self):
        self.select()
        self.crossover()
        self.mutate()

    # 重置
    def reset(self):
        self.POP = self.copy_POP.copy()

    # 打印当前状态日志
    def log(self):
        d1 = self.translateDNA()
        d2 = self.get_fitness().reshape((len(self.POP), 1))
        d3 = np.hstack((d1, d2))

        return pd.DataFrame(d3)


p = np.mat(np.random.rand(N))  # 随机生成一组周转率
print(p)
g = np.mat(np.random.uniform(0, L, N))  # 随机生成一组质量
print(g)

# 生成初始种群
code = []

nums = np.zeros((m, 3 * N))
coo = np.zeros((m, N, 3))  # 以坐标的形式展现出来
for i in range(m):
    for j in range(N):
        code.append(random.sample(range(0, A * B * C), N))
code = np.array(code)
for i in range(m):
    for j in range(N):
        nums[i, j + N] = code[i, j] // (A * C)
        copy = nums[i, j + N]
        nums[i, j + 2 * N] = (code[i, j] - copy * (A * C)) // A
        cop = nums[i, j + 2 * N]
        nums[i, j] = code[i, j] - cop * A - copy * A * C
        coo[i, j] = [nums[i, j], nums[i, j + N], nums[i, j + 2 * N]]


# print(nums)
# print(coo)


def func(Vars):
    x = np.mat(Vars[:, 0:N])
    y = np.mat(Vars[:, N:2 * N])
    z = np.mat(Vars[:, 2 * N:3 * N])
    # 第一个目标函数，基于周转率
    t = x * l / vx + y * r / vy + z * h / vz
    f1 = np.dot(t, p.T)
    # 第二个目标函数，基于重量
    f2 = np.dot(z, g.T)
    # 总目标函数
    ObjV = w1 * f1 + w2 * f2  # 计算目标函数值，赋值给种群对象的ObjV属性
    objv_li = np.array(ObjV.T)
    return objv_li


ga = GA(nums=nums, func=func, cross_rate=cross_rate, mutation=mutation)

for i in range(num):
    ga.evolution()
    dd = ga.log()
    re = dd[dd.loc[:, N * 3] == dd.loc[:, N * 3].max()]
    print(re)
    res = np.array(re)[0]
    result = []
    for i in range(N):
        result.append([res[i], res[i + N], res[i + N * 2]])

print(result)

ds = []
for d in result:
    ds.append(str(d))
qq = set(ds)
print(len(qq))
