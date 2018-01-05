import numpy as np
import matplotlib.pyplot as plt
# 读取数据
with open('/home/yan/文档/小论文实验/seq-NNGN') as f:
    text=f.readlines()
text=text[1:]
seq_2=np.ones([20])
seq_3=np.ones([20])
seq_4=np.ones([20])
for index,line in enumerate(text):
    nums=line.split()
    seq_2[index]=nums[3]
    seq_3[index] = nums[1]
    seq_4[index] = nums[2]

with open('/home/yan/文档/小论文实验/bow-NNGN') as f:
    text=f.readlines()
text=text[1:]
bow_2=np.ones([20])
bow_3=np.ones([20])
bow_4=np.ones([20])
bow_5=np.ones([20])
bow_6=np.ones([20])
bow_7=np.ones([20])
for index,line in enumerate(text):
    nums=line.split()
    bow_2[index]=nums[1]
    bow_3[index] = nums[2]
    bow_4[index] = nums[3]
    bow_5[index] = nums[4]
    bow_6[index] = nums[5]
    bow_7[index] = nums[6]


x_ax=np.linspace(100,2000,20,True)

x=[1,2,3]
y1=[0.1,0.2,0.3]
y2=[0.3,0.4,0.5]
plt.figure(figsize=(10,6),dpi=300)


plt.subplot(1,2,1)
plt.plot(x_ax,seq_2,label='2-ngram')
plt.plot(x_ax,seq_3,label='3-ngram')
plt.plot(x_ax,seq_4,label='4-ngram')
plt.xlabel('embedding dimension')
plt.ylabel('accuracy')
plt.title('seq-NNGN')
plt.legend()

plt.subplot(1,2,2)
plt.plot(x_ax,bow_2,label='2-ngram')
plt.plot(x_ax,bow_3,label='3-ngram')
plt.plot(x_ax,bow_4,label='4-ngram')
plt.plot(x_ax,bow_5,label='5-ngram')
plt.plot(x_ax,bow_6,label='6-ngram')
plt.plot(x_ax,bow_7,label='7-ngram')
plt.xlabel('embedding dimension')
plt.ylabel('accuracy')
plt.title('bow-NNGN')
plt.legend()

plt.savefig('/home/yan/图片/Figure_1.png')
plt.show()