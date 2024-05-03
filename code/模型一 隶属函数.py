import numpy as np
import sympy as sp

z = np.array([100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300,
             310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530])
std_deviation = np.std(z)
mean = np.mean(z)
formatted_mean = "{:.4f}".format(mean)
formatted_std_deviation = "{:.4f}".format(std_deviation)

print("平均值: " + formatted_mean + "\t标准差: " + formatted_std_deviation)


def probability(i):
    j = 0.5 * (1 + sp.erf((i - mean) / (std_deviation * sp.sqrt(2))))
    x = sp.symbols('x')
    f = (1 / (0.1 * sp.sqrt(2 * sp.pi))) * \
        sp.exp(-0.5 * ((x - j) / 0.1) ** 2)
    p = [
        sp.integrate(f, (x, 0, 0.2)),
        sp.integrate(f, (x, 0.2, 0.4)),
        sp.integrate(f, (x, 0.4, 0.6)),
        sp.integrate(f, (x, 0.6, 0.8)),
        sp.integrate(f, (x, 0.8, 1.0)),
    ]

    p = [val / sum(p) for val in p]
    p_rounded = [round(val.evalf(), 4) for val in p]
    p_rounded.reverse()
    return p_rounded


input_value = float(input("请输入自变量数值："))
result = probability(input_value)
print("非常需要保险，需要保险，可有可无，不太需要，非常不需要")
print(result)
