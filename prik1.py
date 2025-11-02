
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

x, y = sp.symbols('x y', real=True)


f1 = -y
f2 = x - 3*x**2


def equilibria(f1: sp.Expr, f2: sp.Expr) -> List[Tuple[float, float]]:
    sols = sp.solve([sp.Eq(f1, 0), sp.Eq(f2, 0)], [x, y], dict=True)
    out = []
    for s in sols:
        xs, ys = s.get(x), s.get(y)
        try:
            out.append((float(sp.N(xs)), float(sp.N(ys))))
        except Exception:
            out.append((xs, ys))
    return out

def J_matrix(f1: sp.Expr, f2: sp.Expr) -> sp.Matrix:
    return sp.Matrix([[sp.diff(f1, x), sp.diff(f1, y)],
                      [sp.diff(f2, x), sp.diff(f2, y)]])

def classify(A: np.ndarray) -> str:
    lam = np.linalg.eigvals(A)
    tr  = np.trace(A)
    det = np.linalg.det(A)
    disc = tr**2 - 4*det
    eps = 1e-9

    if disc >= -1e-9:
        r1, r2 = lam[0].real, lam[1].real
        i1, i2 = lam[0].imag, lam[1].imag
        if abs(i1) < eps and abs(i2) < eps:
            if det < 0:            return "saddle (сідло)"
            if r1 < 0 and r2 < 0:  return "stable node (стійкий вузол)"
            if r1 > 0 and r2 > 0:  return "unstable node (нестійкий вузол)"
            return "degenerate (вироджена)"
    if det > 0:
        if abs(tr) < eps:  return "center (центр)"
        if tr < 0:         return "stable focus (стійкий фокус)"
        if tr > 0:         return "unstable focus (нестійкий фокус)"
    return "degenerate (вироджена)"


print("Система:  x' = -y,   y' = x - 3x^2")
eqs = equilibria(f1, f2)
print("Точки рівноваги:", eqs)

J = J_matrix(f1, f2)
print("\nJ(x,y) =")
sp.pprint(J)

for i, (xe, ye) in enumerate(eqs, 1):
    A = np.array(J.subs({x: xe, y: ye}), dtype=float)
    lam = np.linalg.eigvals(A)
    tr, det = float(np.trace(A)), float(np.linalg.det(A))
    disc = tr**2 - 4*det
    print(f"\n— Точка {i}: ({xe}, {ye})")
    print("  J(x*,y*) =\n", A)
    print("  eigenvalues =", lam)
    print(f"  trace={tr:.6g}, det={det:.6g}, disc={disc:.6g}")
    print("  тип:", classify(A))

f1l = sp.lambdify((x, y), f1, 'numpy')
f2l = sp.lambdify((x, y), f2, 'numpy')

X = np.linspace(-1.5, 1.5, 31)
Y = np.linspace(-1.5, 1.5, 31)
XX, YY = np.meshgrid(X, Y)
U = f1l(XX, YY)
V = f2l(XX, YY)

plt.figure(figsize=(6, 6))
plt.streamplot(XX, YY, U, V, density=1.2, linewidth=1)
plt.xlabel("x"); plt.ylabel("y")
plt.title("Фазовий портрет: x'=-y,  y'=x-3x^2")
for xe, ye in eqs:
    plt.plot([xe], [ye], 'o')
plt.axhline(0, lw=0.5); plt.axvline(0, lw=0.5)
plt.xlim(-1.5, 1.5); plt.ylim(-1.5, 1.5)
plt.tight_layout()
plt.show()
