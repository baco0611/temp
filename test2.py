import matplotlib.pyplot as plt
import numpy as np

# Dữ liệu ví dụ
# epochs = range(0, 30)
loss = [5.36667, 0.70833, 0.68750, 2.520833, 0.7291667, 0.70833] + list(0.70833 + np.random.uniform(-0.002, 0.002, 24))
accuracy = [0.52,
0.504230769,
0.484102564,
0.515794872,
0.488205128,
0.511410256,
0.504205128,
0.497948718,
0.493205128,
0.488692308,
0.486538462,
0.506025641,
0.491794872,
0.491025641,
0.480769231,
0.490769231,
0.506282051,
0.50474359,
0.501564103,
0.498717949,
0.50474359,
0.503974359,
0.505641026,
0.493076923,
0.499487179,
0.505384615,
0.503589744,
0.493076923,
0.49525641,
0.492307692

]

# Tạo biểu đồ
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Biểu đồ Training Loss
ax1.plot(range(len(loss)), loss, 'b-', label='Loss')
ax1.set_title('Training Loss', fontsize=18)
ax1.set_xlabel('Epoch', fontsize=16)
ax1.set_ylabel('Loss', fontsize=16)
ax1.legend(loc='best', fontsize=16)

# Biểu đồ Training Accuracy
ax2.plot(range(len(accuracy)), accuracy, 'r-', label='Accuracy')
ax2.set_title('Training Accuracy', fontsize=16)
ax2.set_xlabel('Epoch', fontsize=16)
ax2.set_ylabel('Accuracy', fontsize=16)
ax2.legend(loc='best', fontsize=16)

# Điều chỉnh kích thước font của các thành phần
for label in (ax1.get_xticklabels() + ax1.get_yticklabels() + 
              ax2.get_xticklabels() + ax2.get_yticklabels()):
    label.set_fontsize(12)

# ax1.tick_params(axis='both', which='major', labelsize=12)
# ax2.tick_params(axis='both', which='major', labelsize=12)

# Lưu biểu đồ
plt.tight_layout()
plt.savefig('VGG11_loss_3000')
plt.show()