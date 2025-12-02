#场论 
Voigt记号（也称为Voigt标记法）是连续介质力学、固体力学和材料科学中一种广泛应用的方法，用于简化对称高阶张量的表示和计算，特别是二阶对称张量（如应力和应变）和四阶张量（如刚度或柔度张量）。
基本思想是：利用对称性进行降维。
# 二阶张量
二阶张量$T_{ij}$原本有$3\times 3=9$个分量。
但当二阶张量具有对称性$T_{ij}=T_{ji}$时，**独立的分量从9个变成6个**。
我们约定
$$        \begin{aligned}
        ij & \rightarrow \alpha \\
        11 & \rightarrow 1 \\
        22 & \rightarrow 2 \\
        33 & \rightarrow 3 \\
        23 \text{ 或 } 32 & \rightarrow 4 \\
        13 \text{ 或 } 31 & \rightarrow 5 \\
        12 \text{ 或 } 21 & \rightarrow 6
        \end{aligned}$$
由此将两个指标$ij$化为一个指标$\alpha$。
在物理学中，这样对称二阶张量的实例有应力$\sigma_{ij}=\sigma_{\alpha}$，应变$\varepsilon_{ij}=\varepsilon_{\alpha}$。
# 三阶张量
三阶张量$P_{ijk}$原本有$3\times3\times 3=27$个分量。
但当三阶张量的其中两个具有指标对称性$P_{ijk}=P_{ikj}$时，**独立的分量从27个变成18个**。
我们约定
$$        \begin{aligned}
        jk & \rightarrow \beta \\
        11 & \rightarrow 1 \\
        22 & \rightarrow 2 \\
        33 & \rightarrow 3 \\
        23 \text{ 或 } 32 & \rightarrow 4 \\
        13 \text{ 或 } 31 & \rightarrow 5 \\
        12 \text{ 或 } 21 & \rightarrow 6
        \end{aligned}$$
由此将三个指标$ijk$化为两个指标$i\beta$。
在物理学中，这样对称二阶张量的实例有压电常数$d_{ijk}=d_{i\beta}$。
# 四阶张量
四阶张量$E_{ijkl}$原本有$3\times3\times 3\times 3=81$个分量。
四阶在使用Voigt记号简化时，需要满足以下对称性之一。
## 亚对称性
当只有其中两个指标满足对称性$E_{ijkl}=E_{jikl}$时，独立的变量从81个变成54个。
按和二阶张量一样的规则，由此将四个指标$ijkl$化为三个指标$\alpha kl$。
## 小对称性
当其中两对指标满足对称性$E_{ijkl}=E_{jikl}=E_{ijlk}$时，独立的变量从81个变成36个。
把前两个指标$ij$按照Voigt规则映射为$\alpha$，把后两个指标$kl$按照Voigt规则映射为$\beta$。
由此将四个指标$ijkl$化为两个指标$\alpha \beta$。