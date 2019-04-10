
# Since tf.matmul is a time-consuming op,
# A better solution is using element-wise multiply, reduce_sum and reshape

# 在可视化给定神经元的感受野时可以在定义conv时候就计算出每层对应的receptive field
# 除了指定神经元，剩下的元素全部标为零（图像可视化）

# cur_feats: 指定卷积层的输出 current feature
# self._pick_feat: 指定channel的index
self.max_act = tf.reduce_max(cur_feats[:, :, :, self._pick_feat]) # 找到给定channel最大的激活值
cond =  tf.equal(cur_feats, tf.ones(tf.shape(cur_feats)) * self.max_act) # boolean matrix(same shape as cur_feats)
out = tf.where(cond, cur_feats, tf.zeros(tf.shape(cur_feats))) # only location with max value non zero
