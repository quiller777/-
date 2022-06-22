# Task03 
# 概率论与基于Python的实施

## 1.随机现象与概率

（1）随机现象：一定条件下，并不总时出现相同结果的现象
（2）随机试验：可重复的随机现象 （不适用与不可重复，比如明年经济预期、比赛结果等）
（3）样本点：随机现象的可能结果（比如掷筛子出现不同点数的结果）
（4）样本空间：随机现象所有基本结果（样本点）的全体
（5）随机事件：随机现象某些基本结果组成的集合（如掷筛子出现奇数）
（6）事件关系：包含($(A \subset B)$)、相等(=)、互斥、必然事件与不可能事件
（7）事件运算：对立（$\bar{A}$ ）、并集（ $A \cup B$）、交集（$A \cap B$）、差集（$A-B$）
（8）事件概率：随机现象中，表示任一随机事件A发生可能性大小的实数为该事件概率，记为$P(A)$
\- 非负性 任一事件概率>=0
\- 正则性 必然事件概率为1
\- 可加性 若$A_{1}$与$A_{2}$互斥，则$P\left(A_{1} \cup A_{2}\right)=P\left(A_{1}\right)+P\left(A_{2}\right)$
（9）事件独立性：若有 $P(A B)=P(A) P(B)$， 则称事件 ${A}$ 与 ${B}$ 相互独立，简称 $A$ 与$B$独立。

## matplotlib 引入相关库

```
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use("ggplot")
import warnings 
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif']=['SimHei','Songti SC','STFangsong']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
import seaborn as sns
```

## 2.条件概率、乘法公式、全概率公式与贝叶斯公式

**(1) 条件概率**
两个事件 $A$ 与 $B$ ，在事件 $B$ 已发生的条件下，事件 $A$ 再发生的概率称为条件概率，记为 $P(A \mid B)$

- $B$不会影响$A$发生的概率（相互独立），$P(A \mid B)$=$P(A )$
- 如果会影响到，那么$P(A \mid B) \not= P(A)$

因为事件$B$的发生影响到事件的样本空间，使得$A$的概率受到影响

**定义:**
设 $A$ 与 $B$ 是基本空间 $\Omega$ 中的两个事件,且 $P(B)>0$,在事件 $B$ 已发生的条件下, 事件 $A$ 的条件概率 $P(A \mid B)$ 定义为 $P(A B) / P(B)$, 即

$
P(A \mid B)=\frac{P(A B)}{P(B)}
$

其中 $P(A \mid B)$ 也称为给定事件 $B$ 下事件 $A$ 的条件概率。

**(2) 乘法公式**

- 若 $P(B)>0$, 则$P(A B)=P(B) P(A \mid B)$
    先让B发生，计算B发生的概率，再计算B发生条件下A发生的概率
    
- 若 $P\left(A_{1} A_{2} \cdots A_{n-1}\right)>0$, 则$P\left(A_{1} A_{2} \cdots A_{n}\right)=P\left(A_{1}\right) P\left(A_{2} \mid A_{1}\right) P\left(A_{3} \mid A_{1} A_{2}\right) \cdots P\left(A_{n} \mid A_{1} A_{2} \cdots A_{n-1}\right)$
    从2个事件引申到多个事件
    

**(3) 全概率公式**
复杂概率计算化繁为简
设 $B_{1}, B_{2}, \cdots, B_{n}$ 是基本空间 $\Omega$ 的一个分割,则对 $\Omega$ 中任一事件 $A$,有(具体的解释如下图)

$$
P(A)=\sum_{i=1}^{n} P\left(A \mid B_{i}\right) P\left(B_{i}\right)




$$

![3.png](../_resources/3.png)

**(4) 贝叶斯公式**
通过简单的$P\left(A \mid B_{k}\right)$求解复杂的$P\left(B_{k} \mid A\right)$

$$
P\left(B_{k} \mid A\right) = \frac{P(AB_k)}{P(A)}


$$

其次，我们对分子分母分别使用乘法公式和全概率公式展开，即：

$$
P\left(B_{k} \mid A\right)=\frac{P\left(A \mid B_{k}\right) P\left(B_{k}\right)}{\sum_{i=1}^{n} P\left(A \mid B_{i}\right) P\left(B_{i}\right)}, \quad k=1,2, \cdots, n


$$

## 3.一维随机变量及其分布函数和密度函数

随机变量：取值带有随机性
离散随机变量：随机变量仅取数轴上有限个点或多个孤立点
连续随机变量：取值充满数轴上一个区间

随机概率：随机变量取这些值的概率（随机事件对应的随机变量取值概率）

- **直接计算法（利用分布函数计算）**
    设 $X$ 为一个随机变量,对任意实数 $x$,事件“ $X \leqslant x$ ”的概率 是 $x$ 的函数,
    记为 $F(x)=P(X \leqslant x)$
    这个函数称为 $X$ 的累积概率分布函数,简称分布函数
    ***分布函数是计算随机变量表示的随机事件的概率的直接方法***
    
- **间接及算法（利用密度函数计算）**
    **连续型随机变量密度函数：**
    对任意两个实数 $a$ 与 $b$,其中 $a<b$, 且 $a$ 可为 $-\infty, b$ 可为 $+\infty, X$ 在区间 $[a, b]$ 上取值的概率为曲线 $p(x)$ 在该区间上曲边梯形的面积,即
    $P(a \leqslant X \leqslant b)=\int_{a}^{b} p(x) d x $
    则称密度函数 $p(x)$ 为连续随机变量 $X$ 的概率分布,或简称 $p(x)$ 为 $X$ 的密度函数,
    记为 $X \sim p(x)$, 读作“ $X$ 服从密度 $p(x)$ ”
    **离散型随机变量密度函数：**
    设 $X$ 是一个离散随机变量,如果 $X$ 的所有可能取值是 $x_{1}, x_{2}, \cdots$, $x_{n}, \cdots$,
    则称 $X$ 取 $x_{i}$ 的概率
    $p_{i}=p\left(x_{i}\right)=P\left(X=x_{i}\right), i=1,2, \cdots, n, \cdots $
    为 $X$ 的概率分布列或简称为分布列, 记为 $X \sim\left\{p_{i}\right\}$
    分布列如下图：
    

$$
\begin{array}{c|ccccc}
X & x_{1} & x_{2} & \cdots & x_{n} & \cdots \\
\hline P & p\left(x_{1}\right) & p\left(x_{2}\right) & \cdots & p\left(x_{n}\right) & \cdots
\end{array}

$$

```
from sympy import *
x = symbols('x')
# 以下为根据密度函数求分布函数
p_x = 1/pi*(1/(1+x**2))
integrate(p_x, (x, -oo, x))
# 以下为根据分布函数求密度函数
f_x = 1/pi*(atan(x)+pi/2)
diff(f_x,x,1)

```

**常见的连续型随机变量 & 密度函数：**

- 均匀分布
    最简单最常见的分布
    
- 指数分布
    若随机变量 $X$ 的密度函数为
    $p(x)=\left\{\begin{aligned} \lambda e^{-\lambda x}, & x \geqslant 0 \\ 0, & x<0 \end{aligned}\right.$
    则称 $X$ 服从指数分布, 记作 $X \sim \operatorname{Exp}(\lambda)$, 其中参数 $\lambda>0$。
    其中 $\lambda$ 是根据实际背景而定的正参数。
    假如某连续随机变量 $X \sim \operatorname{Exp}(\lambda)$, 则表示 $X$ 仅可能取非负实数。
    指数分布的分布函数为：
    $F(x)= \begin{cases}1-\mathrm{e}^{-\lambda x}, & x \geqslant 0 \\ 0, & x<0\end{cases}$
    比如产品故障率随事件服从指数分布
    
- 正态分布（高斯分布）
    若随机变量 $X$ 的密度函数为
    $p(x)=\frac{1}{\sqrt{2 \pi} \sigma} \mathrm{e}^{-\frac{(x-\mu)^{2}}{2 \sigma^{2}}}, \quad-\infty<x<\infty,$
    则称 $X$ 服从正态分布， 称 $X$ 为正态变量， 记作 $X \sim N\left(\mu, \sigma^{2}\right)$。
    其中参数 $-\infty<\mu<\infty, \sigma>0$。
    
    - 如果固定 $\sigma$， 改变 $\mu$ 的值，则图形沿 $x$ 轴平移。
        也就是说正态密度函数的位置由参数 $\mu$ 所确定， 因此亦称 $\mu$ 为位置参数。
    - 如果固定 $\mu$， 改变 $\sigma$ 的值，则分布的位置不变,但 $\sigma$ 愈小，曲线呈高而瘦，分布较为集中; $\sigma$ 愈大，曲线呈矮而胖， 分布较为分散。
        也就是说正态密度函数的尺度由参数 $\sigma$ 所确定， 因此称 $\sigma$ 为尺度参数。

**常见的离散型随机变量 & 密度函数：**

- 0-1分布（伯努利分布）
    任何一个只有两种结果的随机现象都服从0-1分布。
    $P\{X=k\}=p^{k}(1-p)^{1-k}$
    其中 $k=0,1$ 。
    我们也可以写成分布列的表格形式：
    $\begin{array}{c|cccccc} X & 0 & 1 \\ \hline P & p & 1-p \end{array} $
    
- 二项式分布
    𝑛 重伯努里试验中成功出现 𝑘 次
    
- 泊松分布
    分布列为：$P(X=x)=\frac{\lambda^{x}}{x !} e^{-\lambda}$
    其中： $\lambda>0$ 是常数，是区间事件发生率的均值。
    泊松分布是一种常用的离散分布, 它常与单位时间 (或单位面积、单位产品等)上 的计数过程相联系,
    
    - 位于 $\lambda$ （均值）附近概率较大.
    - 随着 $\lambda$ 的增加, 分布逐渐趋于对称

## 4.一维随机变量的数字特征

对随机变量的不同侧面进行描述

### （1）数学期望

数学期望简称期望、期望值或均值，是用概率分布算得的一种加权平均
表示有根据的希望，或发生可能性较大的希望

- 离散型随机变量数学期望
    $X$ 的分布列为
    $P=\left(X=x_{i}\right)=p\left(x_{i}\right), \quad i=1,2, \cdots, n$
    则：
    $E(X)=\sum_{i=1}^{n} x_{i} p\left(x_{i}\right) $
    
- 连续型随机变量数学期望
    $X$ 有密度函数 $p(x)$,
    如果积分$\int_{-\infty}^{\infty}|x| p(x) d x$有限,
    则
    $E(X)=\int_{-\infty}^{\infty} x p(x) d x $
    如果积分无限，则数学期望不存在
    

**数学期望的意义是消除随机性**

**常见的分布的数学期望：**

- 0-1分布：$E(X)=p$
- 二项分布：$E(X) = np$
- 泊松分布：$E(X) = \lambda$
- 均匀分布：$E(X) = \frac{a+b}{2}$
- 指数分布：$E(X) = \frac{1}{\lambda}$
- 正态分布：$E(X) = \mu$

### （2）方差与标准差

期望$E(X)$ 是分布的一种**位置特征数**， 它刻画了 $X$ 的取值总在 $E(X)$ 周围波动。
但这个位置特征数无法反映出随机变量取值的“波动大小”
可能出现数学期望相同，但是波动性存在差异的情况，这时通过方差和标准差进行评估

方差是衡量随机变量波动程度的数学量（波动大小的特征数）

$$
\operatorname{Var}(X)=E(X-E(X))^{2}

$$

离散型随机变量的方差：

$$
\sum_{i}\left(x_{i}-E(X)\right)^{2} p\left(x_{i}\right)

$$

连续型随机变量的方差：

$$
\int_{-\infty}^{\infty}(x-E(X))^{2} p(x) \mathrm{d} x

$$

方差的正平方根 $[\operatorname{Var}(X)]^{1 / 2}$ 称为随机变量 $X($ 或相应分布 $)$ 的标准差,记为 $\sigma_{X}$ 或 $\sigma(X)$

**方差的性质：**

- **最重要的性质：$\operatorname{Var}(X)=E\left(X^{2}\right)-[E(X)]^{2}$**
- 常数的方差为 0 ， 即 $\operatorname{Var}(c)=0$， 其中 $c$ 是常数。
- 若 $a, b$ 是常数，则 $\operatorname{Var}(a X+b)=a^{2} \operatorname{Var}(X)$。
- 若随机变量 $X$ 与 $Y$ 相互独立， 则有$\\operatorname{Var}(X \\pm Y)=\\operatorname{Var}(X)+\\operatorname{Var}(Y) $

**常见分布的方差：**

- 0-1分布：$Var(X) = p(1-p)$
- 二项分布：$Var(X) = np(1-p)$
- 泊松分布：$Var(X) = \lambda$
- 均匀分布：$Var(X) = \frac{(b-a)^{2}}{12}$
- 正态分布：$Var(X) = \sigma^2$
- 指数分布：$Var(X) = \frac{1}{\lambda^{2}}$

```
# 使用scipy计算常见分布的均值与方差：(如果忘记公式的话直接查，不需要查书了)
from scipy.stats import bernoulli   # 0-1分布
from scipy.stats import binom   # 二项分布
from scipy.stats import poisson  # 泊松分布
from scipy.stats import rv_discrete # 自定义离散随机变量
from scipy.stats import uniform # 均匀分布
from scipy.stats import expon # 指数分布
from scipy.stats import norm # 正态分布
from scipy.stats import rv_continuous  # 自定义连续随机变量
```

### （3）分位数和中位数

主要是基于连续随机变量来讨论（离散随机变量分位数或中位数可能不唯一或不存在）
描述随机变量位置的数字特征
累计概率等于p所对应的随机变量取值x为p分位数

设连续随机变量 $X$ 的分布函数为 $F(x)$，密度函数为 $p(x)$。 对任意 $p \in(0,1)$， 称满足条件

$$
F\left(x_{p}\right)=\int_{-\infty}^{x_{p}} p(x) \mathrm{d} x=p

$$

的 $x_{p}$ 为此分布的 $p$ 分位数， 又称下侧 $p$ 分位数。

分位数与上侧分位数是可以相互转换的， 其转换公式如下：

$$
x_{p}^{\prime}=x_{1-p}, \quad x_{p}=x_{1-p}^{\prime} 

$$

**中位数就是p=0.5时的分位数点**
数学期望与中位数都是属于位置特征数，但是有时候中位数可能比期望更能说明问题
中位数相较均值更稳健，不易收到极端数据影响
（中位数计算比较麻烦，接受起来也比较麻烦；
均值会受到极端数据的影响，但是计算简单，也容易接受）

## 5.多维随机变量及其联合分布、边际分布、条件分布

### （1）多维随机变量

若随机变量 $X_{1}(\omega), X_{2}(\omega), \cdots, X_{n}(\omega)$ 定义在同一个基本空间 $\Omega=\{\omega\}$ 上， 则称

$$
\boldsymbol{X}(\omega)=\left(X_{1}(\omega), X_{2}(\omega), \cdots, X_{n}(\omega)\right)

$$

是一个多维随机变量，也称为n维随机向量。

- **n维随机变量的联合分布函数**
    一维随机变量的分布函数是一个关于x的函数，而n维随机变量的联合分布函数就是关于n个自变量的函数：
    设 $X=\left(X_{1}, X_{2}, \cdots, X_{n}\right)$ 是 $n$ 维随机变量，
    对任意 $n$ 个实数 $x_{1}, x_{2}, \cdots, x_{n}$ 所组成的 $n$ 个事件$X_{1} \leqslant x_{1},X_{2} \leqslant x_{2} , \cdots, X_{n} \leqslant x_{n}$
    同时发生的概率
    $F\left(x_{1}, x_{2}, \cdots, x_{n}\right)=P\left(X_{1} \leqslant x_{1}, X_{2} \leqslant x_{2}, \cdots, X_{n} \leqslant x_{n}\right) $
    称为 $n$ 维随机变量 $\boldsymbol{X}$ 的联合分布函数。
    
- **多维连续随机变量的联合密度函数**
    分布函数是密度函数的积分，那推广至多维随机变量也是如此
    设二维随机变量 $(X, Y)$ 的分布函数为 $F(x, y)$ 。假如各分量 $X$ 和 $Y$ 都是一维连续随机变量，并存在定义在平面上的非负函数 $p(x, y)$，使得
    $F(x, y)=\int_{-\infty}^{x} \int_{-\infty}^{y} p(x, y) d x d y$
    则称 $(X, Y)$ 为二维连续随机变量，
    $p(x, y)$ 称为 $(X, Y)$ 的联合概率密度函数， 或简称联合密度。
    在 $F(x, y)$ 偏导数存在的点上有
    

$$
p(x, y)=\frac{\partial^{2}}{\partial x \partial y} F(x, y) 

$$

- **多维离散随机变量的联合分布列**
    如果二维随机变量 $(X, Y)$ 只取有限个或可列个数对 $\left(x_{i}, y_{j}\right)$，
    则称 $(X, Y)$ 为二维离散随机变量，
    称$p_{i j}=P\left(X=x_{i}, Y=y_{j}\right), \quad i, j=1,2, \cdots $为 $(X, Y)$ 的联合分布列

### （2）多维随机变联

- 边际分布函数
    每个分量的分布 (每个分量的所有信息), 即边际分布。
    多维随机向量中的其中一个随机变量 𝑋 排除其他随机变量影响的分布，即 𝑋 自身的分布
    
- 协方差和相关系数
    多维随机向量中，每个元素即单一随机变量都可能受到这组向量其他随机变量的影响，这种影响可以通过协方差来反映
    两个分量之间的关联程度， 用协方差和相关系数来描述。 （后面介绍）
    
- 条件分布
    给定一个分量时，另一个分量的分布, 即条件分布。
    

#### 1）边际分布函数

二维随机变量 $(X, Y)$ 的联合分布函数 $F(x, y)$ 中令 $y \rightarrow \infty$， 由于 $\{Y<\infty\}$ 为必然事件， 故可得

$$
\lim _{y \rightarrow \infty} F(x, y)=P(X \leqslant x, Y<\infty)=P(X \leqslant x),

$$

这是由 $(X, Y)$ 的联合分布函数 $F(x, y)$ 求得的 $X$ 的分布函数， 被称为 $X$ 的边际分布, 记为

$$
F_{X}(x)=F(x, \infty)

$$

类似地， 在 $F(x, y)$ 中令 $x \rightarrow \infty$， 可得 $Y$ 的边际分布

$$
F_{Y}(y)=F(\infty, y) 

$$

#### 2）边际密度函数

二维连续随机变量 $(X, Y)$ 的联合密度函数为 $p(x, y)$， 因为

$$
\begin{aligned}
&F_{X}(x)=F(x, \infty)=\int_{-\infty}^{x}\left(\int_{-\infty}^{\infty} p(u, v) \mathrm{d} v\right) \mathrm{d} u=\int_{-\infty}^{x} p_{X}(u) \mathrm{d} u \\
&F_{Y}(y)=F(\infty, y)=\int_{-\infty}^{y}\left(\int_{-\infty}^{\infty} p(u, v) \mathrm{d} u\right) \mathrm{d} v=\int_{-\infty}^{y} p_{Y}(v) \mathrm{d} v
\end{aligned}

$$

其中 $p_{X}(x)$ 和 $p_{Y}(y)$ 分别为

$$
\begin{aligned}
&p_{X}(x)=\int_{-\infty}^{\infty} p(x, y) \mathrm{d} y \\
&p_{Y}(y)=\int_{-\infty}^{\infty} p(x, y) \mathrm{d} x
\end{aligned}

$$

#### 3）边际分布列

### （3）条件分布

- 多维连续随机变量的条件密度函数
    对一切使 $p_{y}(y)>0$ 的 $y$, 给定 $Y=y$ 条件下 $X$ 的条件密度函数分别为

$$
\begin{aligned}
p(x \mid y)=\frac{p(x, y)}{p_{Y}(y)} .
\end{aligned}

$$

- 全概率公式（扩展）

$$
\begin{aligned}
&p_{Y}(y)=\int_{-\infty}^{\infty} p_{X}(x) p(y \mid x) \mathrm{d} x \\
&p_{\chi}(x)=\int_{-\infty}^{\infty} p_{Y}(y) p(x \mid y) \mathrm{d} y .
\end{aligned}

$$

- 贝叶斯公式（扩展)

$$
\begin{aligned}
&p(x \mid y)=\frac{p_{X}(x) p(y \mid x)}{\int_{-\infty}^{\infty} p_{X}(x) p(y \mid x) \mathrm{d} x},\\
&p(y \mid x)=\frac{p_{Y}(y) p(x \mid y)}{\int_{-\infty}^{\infty} p_{Y}(y) p(x \mid y) \mathrm{d} y} .
\end{aligned}

$$

## 6\. 多维随机变量的数字特征

### (1) 期望向量

期望在多维随机变量的推广

$n$ 维随机向量为 $\boldsymbol{X}=\left(X_{1}, X_{2}, \cdots, X_{n}\right)^{\prime}$, 若其每个分量的数学期望都存在， 则称

$$
E(\boldsymbol{X})=\left(E\left(X_{1}\right), E\left(X_{2}\right), \cdots, E\left(X_{n}\right)\right)^{\prime}

$$

为 $n$ 维随机向量 $\boldsymbol{X}$ 的数学期望向量（一般为列向量）， 简称为 $\boldsymbol{X}$ 的数学期望

### (2) 协方差

衡量两个随机变量之间的相互关联的程度的指标

$$
\operatorname{Cov}(X, Y)=E[(X-E(X))(Y-E(Y))]

$$

当 $\operatorname{Cov}(X, Y)>0$ 时， 称 $X$ 与 $Y$ 正相关
当 $\operatorname{Cov}(X, Y)<0$ 时， 称 $X$ 与 $Y$ 负相关
当 $\operatorname{Cov}(X, Y)=0$ 时， 称 $X$ 与 $Y$ 不相关（毫无关联或存在某种非线性关系）

协方差性质：

- $\operatorname{Cov}(X, Y)=E(X Y)-E(X) E(Y)$
- 若随机变量 $X$ 与 $Y$ 相互独立， 则 $\operatorname{Cov}(X, Y)=0$， 反之不成立。
- （**最重要**）对任意二维随机变量 $(X, Y)$， 有

$$
\operatorname{Var}(X \pm Y)=\operatorname{Var}(X)+\operatorname{Var}(Y) \pm 2 \operatorname{Cov}(X, Y)

$$

这个性质表明: 在 $X$ 与 $Y$ 相关的场合,和的方差不等于方差的和。
$X$ 与 $Y$ 的正相关会增加和的方差,负相关会减少和的方差，
而在 $X$ 与 $Y$ 不相关的场合，和的方差等于方差的和，
即：**若 $X$ 与 $Y$ 不相关**， 则 $\operatorname{Var}(X \pm Y)=\operatorname{Var}(X)+\operatorname{Var}(Y)$。

- 协方差 $\operatorname{Cov}(X, Y)$ 的计算与 $X, Y$ 的次序无关， 即

$$
\operatorname{Cov}(X, Y)=\operatorname{Cov}(Y, X) .

$$

- 任意随机变量 $X$ 与常数 $a$ 的协方差为零，即

$$
\operatorname{Cov}(X, a)=0 

$$

- 对任意常数 $a, b$， 有

$$
\operatorname{Cov}(a X, b Y)=a b \operatorname{Cov}(X, Y) .

$$

- 设 $X, Y, Z$ 是任意三个随机变量,则

$$
\operatorname{Cov}(X+Y, Z)=\operatorname{Cov}(X, Z)+\operatorname{Cov}(Y, Z) 

$$

**总结**：

- $\operatorname{Var}(X)=E(X-E(X))^{2}=E\left(X^{2}\right)-[E(X)]^{2}$
- $\operatorname{Var}(Y)=E(Y-E(Y))^{2}=E\left(Y^{2}\right)-[E(Y)]^{2}$
- $\operatorname{Cov}(X, Y)=E(X Y)-E(X) E(Y)$

### (3) 协方差矩阵

假设$n$ 维随机向量为 $\boldsymbol{X}=\left(X_{1}, X_{2}, \cdots, X_{n}\right)^{\prime}$的期望向量为：

$$
E(\boldsymbol{X})=\left(E\left(X_{1}\right), E\left(X_{2}\right), \cdots, E\left(X_{n}\right)\right)^{\prime}

$$

那么，我们把

$$
\begin{aligned}
& E\left[(\boldsymbol{X}-E(\boldsymbol{X}))(\boldsymbol{X}-E(\boldsymbol{X}))^{\prime}\right] \\
=&\left(\begin{array}{cccc}
\operatorname{Var}\left(X_{1}\right) & \operatorname{Cov}\left(X_{1}, X_{2}\right) & \cdots & \operatorname{Cov}\left(X_{1}, X_{n}\right) \\
\operatorname{Cov}\left(X_{2}, X_{1}\right) & \operatorname{Var}\left(X_{2}\right) & \cdots & \operatorname{Cov}\left(X_{2}, X_{n}\right) \\
\vdots & \vdots & & \vdots \\
\operatorname{Cov}\left(X_{n}, X_{1}\right) & \operatorname{Cov}\left(X_{n}, X_{2}\right) & \cdots & \operatorname{Var}\left(X_{n}\right)
\end{array}\right)
\end{aligned}

$$

为该随机向量的方差-协方差矩阵，简称协方差阵，记为 $\operatorname{Cov}(\boldsymbol{X})$。

注意：$n$ 维随机向量的协方差矩阵 $\operatorname{Cov}(\boldsymbol{X})=\left(\operatorname{Cov}\left(X_{i}, X_{j}\right)\right)_{n \times n}$ 是一个**对称的非负定矩阵**。

### (4) 相关系数

与协方差相同点：
都是衡量随机变量的相关性大小

与协方差不同点
协方差没有排除量纲对数值大小的影响，这样的缺点就是两个协方差之间无法比较相关性的大小
相关系数就是去除量纲影响后的协方差

设 $(X, Y)$ 是一个二维随机变量， 且 $\operatorname{Var}(X)=\sigma_{X}^{2}>0, \operatorname{Var}(Y)=\sigma_{Y}^{2}>0$.
则称

$$
\operatorname{Corr}(X, Y)=\frac{\operatorname{Cov}(X, Y)}{\sqrt{\operatorname{Var}(X)} \sqrt{\operatorname{Var}(Y)}}=\frac{\operatorname{Cov}(X, Y)}{\sigma_{X} \sigma_{Y}}

$$

为 $X$ 与 $Y$ 的 **(线性)** 相关系数。

相关系数的性质：

- $-1 \leqslant \operatorname{Corr}(X, Y) \leqslant 1$， 或 $|\operatorname{Corr}(X, Y)| \leqslant 1$。
- $\operatorname{Corr}(X, Y)=\pm 1$ 的充要条件是 $X$ 与 $Y$ 间几乎处处有线性关系, 即存 在 $a(\neq 0)$ 与 $b$， 使得

$$
 P(Y=a X+b)=1 

$$

- 相关系数 $\operatorname{Corr}(X, Y)$ 刻画了 $X$ 与 $Y$ 之间的线性关系强弱， 因此也常称其为 “线性相关系数”。
- 若 $\operatorname{Corr}(X, Y)=0$， 则称 $X$ 与 $Y$ 不相关。不相关是指 $X$ 与 $Y$ 之间没有线性关系， 但 $X$ 与 $Y$ 之间可能有其他的函数关系， 譬如平方关系、对数关系等。
- 若 $\operatorname{Corr}(X, Y)=1$， 则称 $X$ 与 $Y$ 完全正相关； 若 $\operatorname{Corr}(X, Y)=-1$， 则称 $X$ 与 $Y$ 完全负相关。
- 若 $0<|\operatorname{Corr}(X, Y)|<1$， 则称 $X$ 与 $Y$ 有 “一定程度” 的线性关系。
    $|\operatorname{Corr}(X, Y)|$ 越接近于 1， 则线性相关程度越高；
    $|\operatorname{Corr}(X, Y)|$ 越接近于 0 ，则线性相关程度越低。
    而协方差看不出这一点， 若协方差很小， 而其两个标准差 $\sigma_{X}$ 和 $\sigma_{Y}$ 也很小， 则其比值就不一定很小。

### (5) 相关系数矩阵

类似于协方差矩阵，相关系数矩阵就是把协方差矩阵中每个元素替换成相关系数，具体来说就是：

$$
\begin{aligned}
& \operatorname{Corr}(X, Y)=\frac{\operatorname{Cov}(X, Y)}{\sqrt{\operatorname{Var}(X)} \sqrt{\operatorname{Var}(Y)}}=\frac{\operatorname{Cov}(X, Y)}{\sigma_{X} \sigma_{Y}} \\
=&\left(\begin{array}{cccc}
1 & \operatorname{Corr}\left(X_{1}, X_{2}\right) & \cdots & \operatorname{Corr}\left(X_{1}, X_{n}\right) \\
\operatorname{Corr}\left(X_{2}, X_{1}\right) & 1 & \cdots & \operatorname{Corr}\left(X_{2}, X_{n}\right) \\
\vdots & \vdots & & \vdots \\
\operatorname{Corr}\left(X_{n}, X_{1}\right) & \operatorname{Corr}\left(X_{n}, X_{2}\right) & \cdots & 1
\end{array}\right)
\end{aligned}

$$

## 7\. 随机变量的收敛状态

![28ae2afa49528bb7554ae11f0086e704.png](../_resources/28ae2afa49528bb7554ae11f0086e704.png)

### （1） 依概率收敛

随着 𝑛 不断增大， 大偏差发生的可能性会越来越小

### （2） 依分布收敛

随机变量分布函数序列按分布弱收敛于极限分布函数
随机变量序列也按分布弱收敛

## 8\. 大数定律

伯努利大数定理
当随机事件发生的次数足够多时，随机事件发生的频率 𝑣𝑛 趋近于预期的概率 𝑝
样本数量越多，频率越接近于期望值（概率值） ，其实是依概率收敛的概念

大数定律的条件：独立重复事件与重复次数足够多

蒙特卡洛模拟法（随机投点法）：通过大数定律计算定积分的值

辛钦大数定理

## 9\. 中心极限定理
