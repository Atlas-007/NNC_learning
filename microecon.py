import numpy as np
import matplotlib.pyplot as plt

# simple settings
C_MAX, P_MAX = 5, 5
out_file = "indifference_plot_simple.png"

# continuous grid for nice contours (note: p on x-axis, c on y-axis)
p_vals = np.linspace(0.1, P_MAX, 200)
c_vals = np.linspace(0.1, C_MAX, 200)
P, C = np.meshgrid(p_vals, c_vals)        # P = x, C = y
U = np.sqrt(C * P)                        # utility = sqrt(c * p)

# contour levels: use the utility at every integer (p,c) grid point
levels = np.unique([np.sqrt(p * c) for p in range(1, P_MAX + 1) for c in range(1, C_MAX + 1)])

plt.figure(figsize=(7,6))
cs = plt.contour(P, C, U, levels=levels)  # indifference curves through integer points
plt.clabel(cs, fmt="%.2f", inline=True, fontsize=8)

plt.xlabel("p")
plt.ylabel("c")
plt.title(r"Indifference curves for $u=\sqrt{c\cdot p}$ (points at $(p,c)$)")

# plot integer points (p, c) and label with utility
for p in range(1, P_MAX+1):
    for c in range(1, C_MAX+1):
        u = (c * p) ** 0.5
        plt.scatter(p, c, marker='x')
        plt.text(p + 0.06, c + 0.06, f"{u:.2f}", fontsize=8)

plt.xlim(0.8, P_MAX + 0.2)
plt.ylim(0.8, C_MAX + 0.2)
plt.grid(alpha=0.4)
plt.tight_layout()
plt.savefig(out_file, dpi=150)
plt.show()
print("Saved to", out_file)






