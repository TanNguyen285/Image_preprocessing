import matplotlib.pyplot as plt
import numpy as np

# ===== 5 LẦN TEST =====

tests = ["Test0", "Test1", "Test2", "Test3"]

# DCE++ Enhancement (ms)
dce_enh = [39.3, 41.67, 42.33, 45.84]

# SCI Enhancement (ms)
sci_enh = [8.67, 9.79, 9.14, 9.58]

x = np.arange(len(tests))
width = 0.35

plt.figure()

bars1 = plt.bar(x - width/2, dce_enh, width, label="DCE++")
bars2 = plt.bar(x + width/2, sci_enh, width, label="SCI")

plt.xticks(x, tests)
plt.ylabel("Enhancement Time (ms)")
plt.title("CPU_640x480")
plt.legend()

# ===== HIỂN THỊ SỐ TRÊN CỘT =====
for bar in bars1:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2,
             height,
             f"{height:.2f}",
             ha='center',
             va='bottom')

for bar in bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2,
             height,
             f"{height:.2f}",
             ha='center',
             va='bottom')

plt.tight_layout()
plt.show()