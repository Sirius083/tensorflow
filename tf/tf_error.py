# 没有你犯不了的错误TOT
# 每次训练的结果都是一样的，每个batch基本都选的是一样的好么
# sirius: 这里有问题：：
for j in range(num_step): # each epoch
    x_batch = train_epoch[j:j+train_batch_size,:]
    y_true_batch = labels_epoch[j:j+train_batch_size]
   
   
   
# 训练最后一层的全连接层加上了relu的激活函数
