import matplotlib.pyplot as plt
import numpy as np

logs = ['logu_FTUNetFormer_sam_v1', 'logu_FTUNetFormer_sam_v1']
for log in logs[1:]:
    with open("logs/unetformer/"+log, 'r') as f, open("logs/unetformer/"+log+"_loss", "w") as fin:
        lines = f.readlines()
        for line in lines:
            if 'Loss' in line:
                fin.write(line)
    f.close()
    fin.close()

##  Vaihingen
fig_M = 500
fig_N = 500
fig_loss_0 = np.zeros([fig_M])
fig_loss_1 = np.zeros([fig_M])

it = 0
with open("logs/unetformer/"+logs[1]+'_loss', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.replace('\n', '').split(' ')
        loss = float(line[5].replace('\tLoss_boundary:', ''))
        fig_loss_0[it] = loss
        it = it + 1
f.close()

it = 0
with open("logs/unetformer/"+logs[1]+'_loss', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.replace('\n', '').split(' ')
        loss = float(line[6].replace('\tLoss_object:', ''))
        fig_loss_1[it] = loss
        it = it + 1
f.close()

fig_loss_0 = fig_loss_0[:fig_N]
fig_loss_1 = fig_loss_1[:fig_N]

meant = 500
fig_loss_0 = np.reshape(fig_loss_0, (meant, 1))
fig_loss_0 = np.mean(fig_loss_0, axis=1)
fig_loss_1 = np.reshape(fig_loss_1, (meant, 1))
fig_loss_1 = np.mean(fig_loss_1, axis=1)

fig, ax1 = plt.subplots()
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# 设置坐标标签字体大小
ax1.set_xlabel(..., fontsize=20)
ax1.set_ylabel(..., fontsize=20)

lns1 = ax1.plot(np.arange(meant), fig_loss_0, label="train_acc", color='b', linestyle='-')
lns2 = ax1.plot(np.arange(meant), fig_loss_1, label="train_acc", color='r', linestyle='-')

ax1.set_xlabel('iteration (epoch)', fontsize=26)
ax1.set_ylabel('loss', fontsize=26)
lns = lns1 + lns2
labels = ["log0", "log1"]
plt.legend(lns, labels, loc=0, fontsize=20)

plt.show()
