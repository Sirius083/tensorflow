# Note for official cliqueNet network architecture
# https://github.com/iboing/CliqueNet

'''
问题
最后一个block的 0_1, 0_2, 0_3, 0_4, 0_5 的variable的大小都是 (3,3,320,64)?? 
'''

'''
parameter meaning
if_a: attentional transition
if_b: bottleneck
if_c: compression
T: the sum of layers in all blocks
k: number of filters per layer

param_dict: 所有定义的卷积层的字典（总变量个数是A_n_(2) + n）
            将所有卷积层参数存在param_dict
blob_dict: blob_dict[i]: 初始化阶段 经过i次conv之后的网络输出
           将每一个物理层的输出存在blob_dict中
layer_num: 当前block中的总层数

param_dict:
1_2,2_1,1_3,3_1,1_4,4_1,1_5,5_1,2_3,3_2,2_4,4_2,2_5,5_2,3_4,4_3,3_5,5_3,4_5,5_4,
0_1,0_2,0_3,0_4,0_5
每个variable的大小都是(3,3,64,64)

blob_dict:
1,2,3,4,5
每个variable的大小都是(N,32,32,64) --> block 1
                     (N,16,16,64) --> block 2
                     (N,8,8,64)   --> block 3

'''

'''
Network Architecture:
conv: BN --> relu --> conv
transition layer: 1x1 conv --> 2x2 average pooling
x_4_(2): x_5_(1), x_1_(2), x_2_(2), x_3_(2)
         previous layer, # larger than it
         current  block, # smaller than it 

multi-scale feature strategy to compose final representation
with the block in different map size

难点：
block: 一个block包含多个stage
stage: 参数更新的不同阶段(相当于readout time t)
       stage-1: 首次更新
       stage-2: 第二次更新（其他层的输出作为另一个层的输入）
相同stage前面层的输出集合作为当前层的输入
前stage后面层的输出作为当前层的输入

input layer & stage-2 feature (concate) --> global pooling --> loss function
stage-2 feature is the input of the next block (denote as x_0)
'''


'''
Techniques:
increase performance and accuracy
1. channel-wise attention: defined in SENet (only at transition layer)
2. bottleneck: for deep network & large dataset(imagenet)
3. compression: filter compression

Advantages:
1. weights are updated alternatively
2. representation learning: combination of recurrent structure and feedback mechanism

Two main structure:
1. Clique Block: ebavke feature refinement
2. multi-scale feature strategy: facilitate parameters efficiency

CliqueNet version:
Clique_I_I: only consider stage-1 features
clique_I_II: stage-1 to loss, stage-2 to next block
clique_II_II: x_0 @ stage-2 to loss, stage-2 to next block
'''


'''
parameter output:
param_dict:
1_2 <tf.Variable 'b0-1_2:0' shape=(3, 3, 64, 64) dtype=float32_ref>
2_1 <tf.Variable 'b0-2_1:0' shape=(3, 3, 64, 64) dtype=float32_ref>
1_3 <tf.Variable 'b0-1_3:0' shape=(3, 3, 64, 64) dtype=float32_ref>
3_1 <tf.Variable 'b0-3_1:0' shape=(3, 3, 64, 64) dtype=float32_ref>
1_4 <tf.Variable 'b0-1_4:0' shape=(3, 3, 64, 64) dtype=float32_ref>
4_1 <tf.Variable 'b0-4_1:0' shape=(3, 3, 64, 64) dtype=float32_ref>
1_5 <tf.Variable 'b0-1_5:0' shape=(3, 3, 64, 64) dtype=float32_ref>
5_1 <tf.Variable 'b0-5_1:0' shape=(3, 3, 64, 64) dtype=float32_ref>
2_3 <tf.Variable 'b0-2_3:0' shape=(3, 3, 64, 64) dtype=float32_ref>
3_2 <tf.Variable 'b0-3_2:0' shape=(3, 3, 64, 64) dtype=float32_ref>
2_4 <tf.Variable 'b0-2_4:0' shape=(3, 3, 64, 64) dtype=float32_ref>
4_2 <tf.Variable 'b0-4_2:0' shape=(3, 3, 64, 64) dtype=float32_ref>
2_5 <tf.Variable 'b0-2_5:0' shape=(3, 3, 64, 64) dtype=float32_ref>
5_2 <tf.Variable 'b0-5_2:0' shape=(3, 3, 64, 64) dtype=float32_ref>
3_4 <tf.Variable 'b0-3_4:0' shape=(3, 3, 64, 64) dtype=float32_ref>
4_3 <tf.Variable 'b0-4_3:0' shape=(3, 3, 64, 64) dtype=float32_ref>
3_5 <tf.Variable 'b0-3_5:0' shape=(3, 3, 64, 64) dtype=float32_ref>
5_3 <tf.Variable 'b0-5_3:0' shape=(3, 3, 64, 64) dtype=float32_ref>
4_5 <tf.Variable 'b0-4_5:0' shape=(3, 3, 64, 64) dtype=float32_ref>
5_4 <tf.Variable 'b0-5_4:0' shape=(3, 3, 64, 64) dtype=float32_ref>
0_1 <tf.Variable 'b0-0_1:0' shape=(3, 3, 64, 64) dtype=float32_ref>
0_2 <tf.Variable 'b0-0_2:0' shape=(3, 3, 64, 64) dtype=float32_ref>
0_3 <tf.Variable 'b0-0_3:0' shape=(3, 3, 64, 64) dtype=float32_ref>
0_4 <tf.Variable 'b0-0_4:0' shape=(3, 3, 64, 64) dtype=float32_ref>
0_5 <tf.Variable 'b0-0_5:0' shape=(3, 3, 64, 64) dtype=float32_ref>

blob_dict:
1 Tensor("dropout_5/mul:0", shape=(?, 32, 32, 64), dtype=float32)
2 Tensor("dropout_6/mul:0", shape=(?, 32, 32, 64), dtype=float32)
3 Tensor("dropout_7/mul:0", shape=(?, 32, 32, 64), dtype=float32)
4 Tensor("dropout_8/mul:0", shape=(?, 32, 32, 64), dtype=float32)
5 Tensor("dropout_9/mul:0", shape=(?, 32, 32, 64), dtype=float32)
'''

'''
构建网络：
stage_1 ================================
layer_id = 1
bottom_layer = input_layer = (N,32,32,64)_0
bottom_param : 0-1
mid_layer(bn-relu-conv-dropout) = (N,32,32,64)
block_dict['1'] = (N,32,32,64)_1


layer_id = 2
bottom_layer = input_layer = (N,32,32,64)_0
bottom_param : 0-2
layer_id_id = 1
bottom_blob = (N,32,32,64)_0 @ (N,32,32,64)_1 = (N,32,32,128)
bottom_param  0-2 @ 1-2 # axis = 2 # (3,3,64,64) @ (3,3,64,64) = (3,3,128,64)
mid_layer = (N,32,32,128) * (3,3,128,64) = (N,32,32,64)
blob_dict['2'] = (N,32,32,64)_2

layer_id = 3
bottom_layer = (N,32,32,64)_0
bottom_param : 0-3
layer_id_id = 1
bottom_blob = (N,32,32,64)_0 @ (N,32,32,64)_1
bottom_param = 0-3 @ 1-3
layer_id_id = 2
bottom_blob = (N,32,32,64)_0 @ (N,32,32,64)_1 @ (N,32,32,64)_2 --> (N,32,32,192)
bottom_param = 0-3 @ 1-3 @ 2-3 --> (3,3,64,64) @ (3,3,64,64) @ (3,3,64,64) --> (3,3,192,64)
blob_dict['3'] = (N,32,32,64)_3

layer_id = 4

layer_id = 5

stage_2====================================
loop_num = 1
loop_id = 0

layer_id = 1
layer_list = ['2','3','4','5']
init_blob = blob_dict['2']
init_param = param_dict['2_1']
blob: blob_dict['2'] @ blob_dict['3']
param: param_dict['2_1'] @ param_dict['3_1']
blob: blob_dict['2'] @ blob_dict['3'] @ blob_dict['4']
param: param_dict['2_1'] @ param_dict['3_1'] @ param_dict['4_1']
blob: blob_dict['2'] @ blob_dict['3'] @ blob_dict['4'] @ blob_dict['5'] --> (N,32,32,256)
param: param_dict['2_1'] @ param_dict['3_1'] @ param_dict['4_1'] @ param_dict['5_1'] --> (3,3,256,64)
blob_dict['1'] = (N,32,32,64)_1

layer_id = 2
layer_list = ['1','3','4','5']
init_blob = blob_dict['1']
init_param = param_dict['1_2']
blobs = blob['1'] @ blob['3'] @ blob['4'] @ blob['5']
params = params['1_2'] @ params['3_2'] @ params['4_2'] @ params['5_2']
blob_dict['2'] = (N,32,32,64)_2

layer_id = 3
layer_list = ['1','2','4','5']
blobs = blob['1'] @ blob['2'] @ blob['4'] @ blob['5']
params = params['1_3'] @ params['2_3'] @ params['4_3'] @ params['5_3']
block_dict['3'] = (N,32,32,64)_3

layer_id = 4

layer_id = 5

transition_layer =====================================
block_feature: input_layer @ blob_dict['1'] @ blob_dict['2'] @ blob_dict['3'] @ blob_dict['4'] @ blob_dict['5']
transition_feature: blob_dict['1'] @ blob_dict['2'] @ blob_dict['3'] @ blob_dict['4'] @ blob_dict['5']
'''

