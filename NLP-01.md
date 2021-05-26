# 自然语言处理
## 文本的编码
```
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
#通过分词器对句子进行编码，它可以快速地产生词典并创建词向量
#preprocessing能够将文本转化为tokens流
sentences=[
    'I love my dog',
    'I love my cat',
    'You love my dog!'     # 句子中的标点符号会自动删除，编码会忽略大小写
]
tokenizer=Tokenizer(num_words=100)#表示要建立一个100个单词的词典，当文本比较大时，分词器将选取词频最大的100个单词，放入词典进行编码
#因为那些词频低的单词，往往对神经网络的训练精度影响很小，但是会极大地增加训练时间
tokenizer.fit_on_texts(sentences)#将sentences数组中的单词按照出现的频率进行排序，将前100个单词放入分词器进行编码
word_index=tokenizer.word_index   #查看词典中的编码数据
print(word_index)
```
## 文本的序列化  将文本按照编码转化为矩阵

```
#序列化   序列化，其实就是将文本按照编码方式生成一个矩阵，每一个句子生成一行
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


sentences=[
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
]
tokenizer=Tokenizer(num_words=100,oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index=tokenizer.word_index
sentences=tokenizer.texts_to_sequences(sentences)  #将文本序列化

#将序列化的补齐
padded=pad_sequences(sentences)   #就是在前面填充0
#如果需要将0填充在后面，则可以在后面加上padding='post'
# padded=pad_sequences(sentences,padding='post')
#可以通过maxlen指定矩阵每一行的长度,这样会丢失一部分信息，是从什么地方开始丢失信息，需要指定，默认也是从前面开始丢失信息
#通过truncating=指定从哪里开始丢失信息
# padded=pad_sequences(sentences,padding='post',maxlen=5,truncating='post')


print(word_index)
print(sentences)

print(padded)

#如果文本中存在词典中没有的词的编码，则会丢失
test_data=[
    'i really love my dog',   #really不在编码的词典中，所以会在编码的矩阵中丢失
    'my dog loves my manatee' #loves也会丢失
]
test_seq=tokenizer.texts_to_sequences(test_data)
print(test_seq)

# 可以使用一个符号标记未在编码词典中出现的词oov_token="<OOV>"  <OOV>会被编码为1，所以文本中未出现的单词都会编码为1

#将序列化的到的矩阵变成规则矩阵，就是将每一行都变成相等的长度
#使用pad_sequences来保持编码后句子长度的一致性

```

## 例子
+ 词的编码
+ 词的序列化  截断对齐
+ 词嵌入 神经网络第一层为嵌入层

```
'''
sarcasm讽刺数据集：下载：https://rishabhmisra.github.io/publications/
projector.tensorflow.org

首先将单词映射到矢量空间，然后将启用于神经网络的训练
'''
'''
构建一个可视化的情感分类器
tensorflow内置数据集：
tensorflow有一个名为Tensorflow Data Services简称tfds的库，里面包含了很多的数据集
如果tensorflow为1.x版本，则需要使用tf.enable_eager_execution()

安装tensorflow数据集： !pip install -q tensorflow-datasets

'''

import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#返回两个数据集imdb数据集和info数据集，这里只使用imdb数据集
imdb,info=tfds.load("imdb_reviews",with_info=True,as_supervised=True)
#获得imdb_reviews数据集，整个数据集被分成了两个部分，其中25000个样本用于训练，25000个样本用于测试
train_data,test_data=imdb['train'],imdb['test']
#训练数据和测试数据均包含了25000个句子和相应的标签
training_sentences=[]
training_labels=[]

testing_sentences=[]
testing_labels=[]

for s,l in train_data:
    training_sentences.append(str(s.numpy()))
    training_labels.append(l.numpy())

for s,l in test_data:
    testing_sentences.append(str(s.numpy()))
    testing_labels.append(l.numpy())

training_labels_final=np.array(training_labels)
testing_labels_final=np.array(testing_labels)


vocab_size=10000
embedding_dim=16
max_length=120
trunc_type='post'
oov_tok="<OOV>"

tokenizer=Tokenizer(num_words=vocab_size,oov_token=oov_tok)

tokenizer.fit_on_texts(training_sentences)
word_index=tokenizer.word_index

sequences=tokenizer.texts_to_sequences(training_sentences)
padded=pad_sequences(sequences,maxlen=max_length,truncating=trunc_type)

testing_sequences=tokenizer.texts_to_sequences(testing_sentences)   #测试数据的序列化生成所使用的词典是使用训练数据生成的
testing_padded=pad_sequences(testing_sequences,maxlen=max_length)

print("==============================")
#查看编码前后的   编码之后的句子和原始句子之间的差别
reverse_word_index=dict([(value,key) for (key,value) in word_index.items()])
def decode_review(text):
    return ' '.join([reverse_word_index.get(i,'?') for i in text])
print(decode_review(padded[1]))
print(training_sentences[1])

print("=================================")



model=tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_length),
    tf.keras.layers.Flatten(),
    #也可以使用另外一种方式将神经网络层展平
    # tf.keras.layers.GlobalAveragePooling1D(),#全局平均池化层
    tf.keras.layers.Dense(6,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

num_epochs=10
model.fit(padded,
          training_labels_final,
          epochs=num_epochs,
          validation_data=(testing_padded,testing_labels_final)
          )

e=model.layers[0]
weights=e.get_weights()[0]
print(weights.shape)



#将训练完成以后的结果可视化
import io
out_v=io.open("vecs.tsv",'w',encoding='utf-8')
out_m=io.open('meta.tsv','w',encoding='utf-8')
for word_num in range(1,vocab_size):
    word=reverse_word_index[word_num]
    embeddings=weights[word_num]
    out_m.write(word+"\n")
    out_v.write('\t'.join([str(x) for x in embeddings])+"\n")
out_v.close()
out_m.close()


```


