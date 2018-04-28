# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import re
import time
import datetime

def clean_data(string):
    """
    from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    数据清洗
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\`\']", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"\'t", " \'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string

def load_data(positive_data,negative_data):
    """
    数据载入
    """
    positive_text = [s.strip() for s in open(positive_data,"r",encoding="UTF-8").readlines()]
    negative_text = [s.strip() for s in open(negative_data,"r",encoding="UTF-8").readlines()]
    x_text = [clean_data(s) for s in (positive_text + negative_text)]

    positive_label = [[0,1] for _ in positive_text]
    negative_label = [[1,0] for _ in negative_text]
    y = positive_label + negative_label

    return [x_text,y]


def get_weights_and_biases(w_shape,b_shape):
    """
    权重、偏置初始化函数
    """
    w_form = tf.truncated_normal(shape = w_shape,stddev= 0.1)
    b_form = tf.constant(0.1,shape = b_shape)
    return [tf.Variable(w_form),tf.Variable(b_form)]

def batch_generator(data,batch_size,train_epoch):
    """
    数据batch生成函数
    """
    data = np.array(data)
    data_size = len(data)
    batch_num = int( (data_size-1)/batch_size) + 1
    for epoch in range(train_epoch):
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffle_data = data[shuffle_indices]
        for batch in range(batch_num):
            start_index = batch * batch_size
            end_index   = min( (batch+1)*batch_size, data_size )
            yield shuffle_data[start_index:end_index]


def text_cnn(SENTENCE_LEN,VOCAB_SIZE,x_train,y_train,x_val,y_val):
    """
    CNN结构
    """
    OUTPUT_SIZE    = 2
    EMBEDDING_SIZE = 128
    FILTER_HEIGHTS = [3,4,5]
    FILTER_NUM     = 128
    L2NORM_RATE    = 0.0001
    LEARNING_RATE  = 0.001
    TRAIN_EPOCH    = 100
    BATCH_SIZE     = 100

    #定义输入占位
    input_x = tf.placeholder(tf.int32, shape= [None,SENTENCE_LEN], name = "input_x")
    input_y = tf.placeholder(tf.float32, shape= [None,OUTPUT_SIZE],name = "input_y")
    dropout_keep_prob = tf.placeholder(tf.float32,name = "dropout_keep_prob")
    l2_loss = tf.constant(0.0)

    #将输入句子转换为随机的词向量
    with tf.name_scope("embedding"):
        vocab_embedding = tf.Variable( tf.random_uniform(shape= [VOCAB_SIZE,EMBEDDING_SIZE]) )
        x_embedding_pre = tf.nn.embedding_lookup(vocab_embedding, input_x)
        x_embedding = tf.expand_dims(x_embedding_pre,-1)

    #三组不同大小的filter，形成三种卷积层
    pooling2_out = []
    for FILTER_HEIGHT in FILTER_HEIGHTS:
        #第一层卷积层:conv1
        with tf.name_scope("conv1-%s" % FILTER_HEIGHT):
            w,b = get_weights_and_biases([FILTER_HEIGHT,EMBEDDING_SIZE,1,FILTER_NUM],[FILTER_NUM])
            conv1_op = tf.nn.conv2d(
                input = x_embedding,
                filter = w,
                strides = [1,1,1,1],
                padding = "VALID",
                name = "conv1_op")
            conv1 = tf.nn.relu( tf.nn.bias_add(conv1_op,b,name = "conv1") )

        #第二层池化层:pooling2
        with tf.name_scope("pooling2-%s" % FILTER_HEIGHT):
            pooling2  = tf.nn.max_pool(
                value = conv1,
                ksize = [1,SENTENCE_LEN - FILTER_HEIGHT +1,1,1],
                strides = [1,1,1,1],
                padding = "VALID",
                name = "pooling2")
            pooling2_out.append(pooling2)
    #池化结果展开
    pooling2_concat = tf.concat(pooling2_out,3)
    pooling2_flat = tf.reshape(pooling2_concat,[-1,len(FILTER_HEIGHTS) * FILTER_NUM])
    pooling2_flat_dropout = tf.nn.dropout(pooling2_flat,dropout_keep_prob)

    #第三层全连接层：fc3
    with tf.name_scope("fc3"):
        w,b = get_weights_and_biases([len(FILTER_HEIGHTS) * FILTER_NUM,OUTPUT_SIZE],[OUTPUT_SIZE])
        y_hat = tf.nn.xw_plus_b(pooling2_flat_dropout,w,b)
        l2_loss += tf.nn.l2_loss(w) + tf.nn.l2_loss(b)

    #定义交叉熵+正则化约束loss
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = y_hat, labels = input_y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + L2NORM_RATE * l2_loss

    #定义准确率
    correct_predictions = tf.equal( tf.argmax(y_hat,1), tf.argmax(input_y,1) )
    accuracy = tf.reduce_mean( tf.cast(correct_predictions,tf.float32) )

    #使用AdamOptimizer优化
    global_step = tf.Variable(0,trainable=False)
    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss,global_step = global_step)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        #验证数据集
        val_feed = {
            input_x : x_val,
            input_y : y_val,
            dropout_keep_prob : 1.0
        }

        batches = batch_generator(list(zip(x_train,y_train)) , BATCH_SIZE, TRAIN_EPOCH)
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            #训练数据batch
            feed_dict = {
                input_x: x_batch,
                input_y: y_batch,
                dropout_keep_prob : 0.5
            }

            _,train_step,train_loss,train_acc = sess.run(
                [train_op, global_step, loss, accuracy], feed_dict = feed_dict)

            if train_step % 10 == 0:
                print( "step:{} , loss:{:g} ,acc:{:g}".format(train_step,train_loss,train_acc))

            if train_step % 500 == 0:
                val_loss, val_acc = sess.run([loss,accuracy],feed_dict = val_feed)
                print( "After {} steps, in validation dataset, loss is {:g}, acc is {:g}".format(train_step,val_loss,val_acc))

        final_acc = sess.run(accuracy,feed_dict=val_feed)
        print("After {} epochs, in validation dataset, final accuracy is {:g}".format(TRAIN_EPOCH, final_acc))


if __name__ == "__main__":

    VALIDATION_RATE = 0.1
    x_text,y = load_data(r"data/rt-polaritydata/rt-polarity.pos",r"data/rt-polaritydata/rt-polarity.neg")
    y = np.array(y)

    #将输入文本转换为词典表示
    max_document_length = max( [ len(s.split()) for s in x_text] )
    #print((max_doc_length))
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))
    vocab_size = len(vocab_processor.vocabulary_)

    #生成训练、验证集
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    split_indice = -1 * int(VALIDATION_RATE * float(len(y)))
    x_train, x_val = x_shuffled[:split_indice],x_shuffled[split_indice:]
    y_train, y_val = y_shuffled[:split_indice],y_shuffled[split_indice:]

    del x_shuffled,y_shuffled,x,y

    #训练CNN
    text_cnn(max_document_length,vocab_size,x_train,y_train,x_val,y_val)
