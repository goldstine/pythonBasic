# pythonBasic
```
默认情况下jupyternotebook是没有代码提示的，如果需要提示可以在开始位置加上
%config IPComplater.greedy=True   #TAB键代码自动提示

加载fasion_mnist数据集的方法
fasion_mnist=keras.datasets.fasion_mnist
(train_images,train_labels),(test_images,test_labels)=fasion_mnist.load_data()

如果是数值型的输出，损失函数就使用spare_categorical_crossentropy
如果是one_hot编码的输出，损失函数就是用categorical_crossentropy

在图片分类问题中，可以将图片的像素点数据在输入之前，归一化train_images=train_images/255,归一化以后的训练效果会更好，损失函数值更小精度更高
如果在训练的时候做了归一化，那么在评估的时候测试集也要做归一化  test_train=test_train/255

对单张图片进行预测 model.predict([[test_images[0]/255]])
通过numpy的argmax获得图片所属的类别：   import numpy as np     np.argmax(model.predict([[test_images[0]/255]]))

训练次数不是越多越好，会出现过拟合的现象
所以需要相应的终止条件及时地终止训练过程，通过回调函数及时地终止训练过程
callback的使用方式，直接继承自callback，定义一个自定义的子类
然后将该类的实例传给model.fit（）的训练参数

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self,epoch,logs={}):
    if(logs.get('loss')<0.4):
      print("\n Loss is low so cancelling training!")
      self.model.stop_training=True
      
callbacks=myCallback()

model.fit(train_images_scaled,train_labels,epochs=5,callbacks=[callbacks])    #每次训练以后都会调用该回调函数进行检查是否可以终止


```
# 卷积神经网络
+ 之所以出现CNN，一个原因是一个像素点周围的点相关，所以特征提取使用卷积核
+ 还有一个原因是因为实际中的图片的内容在图片中的位置是不同的，不一定居中，有旋转角度，所以使用CNN进行特征提取

```
卷积神经网络的构建
modelmodel=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)),
    tf.keras.layers.MaxPool2D(2,2),  #池化层的作用就是增强特征，减少数据
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
])

输入的数据在输入卷积层之前：reshape(-1,28,28,1)
model.fit(train_images_scaled.reshape(-1,28,28,1),train_labels,epochs=5)

卷积神经网络实际上训练的参数是卷积核，加的偏置值也是在每一个卷积核filter上加上一个相同的值

把中间层取出来
观察卷积网络发生的过程
import matplotlib.pyplot as plt
layer_outputs=[layer.output for layer in model.layers]
activation_model=tf.keras.models.Model(inputs=model.input,outputs=layer_outputs)
pred=activation_model.predict(test_images[0].reshape(1,28,28,1))
plt.imshow(pred[][0,:,:,1])

```
# 项目实战
## 数据准备
+ 首先下载数据解压，所有的数据都按照文件夹名称进行打标签，所有图片文件都放在images文件夹下
+ images
+ ---------train:
                human:
                horse:
           validation:
                human:
                horse:
 
 
## 数据预处理
+ 真实数据的特点
（1）图片尺寸大小不一，需要裁剪成一样的大小
（2）数据量比较大，不能一下子装入内存    数据通道,之前的时候都是直接load_data()的方式将所有数据加载进内存
（3）进场需要修改参数，例如输出的尺寸，增补图像拉伸等

ImageDataGenerator:

创建数据生成器：
train_datagen=ImageDataGenerator(rescale=1/255)
validation_datagen=ImageDataGenerator(rescale=1/255)

train_generator=train_datagen.flow_from_directory(
    '/tmp/horse-or-human/',   #训练数据所在的文件夹
    target_size=(300,300),    #指定输出尺寸
    batch_size=32,          
    class_mode='binary'      #指定二分类

)

validation_generator=validation_datagen.flow_from_directory(
  '/tmp/validation-horse-or-human/',
  target_size=(300,300),
  batch_size=32,
  class_mode='binary'
)

## 模型的构建与训练

## 参数优化
+ 自己手工一个一个参数优化试
+ 写一个循环函数对需要优化的参数进行循环试
+ 通过库kerastuner          

from kerastuner.tuners import Hyperband
from kerastuner.engine.hyperparameters import HyperParames
将需要探索得值，使用
hp=HyperParameters()
def build_model(hp):
  model=tf.keras.models.Sequential()
  model.add(tf.keras.layers.Conv2D(hp.choice('num_filters_layer0',values=[16,64],default=16),(3,3),activation='relu',input_shape=(300,300,3)))
  
  for i in range(hp.Int("num_conv_layers",1,3)):
    model.add(tf.keras.layers.Conv2D(hp.choice('num_filters_layer0',values=[16,64],default=16),(3,3),activation='relu',input_shape=(300,300,3)))
    model.add(tf.keras.layers.MaxPooling2D(2,2))    #需要多少个卷积池化层
  ...
  model.add(tf.keras.layers.Dense(hp.Int("hidden_units",128,512,step=32),activation='relu'))
  return model
  
  
tuner=Hyperband(
  build_model,
  objective='val_acc',
  max_epochs=15,
  directory='horse_human_params',     #指定本地目录，在参数有的时候会将相关的参数保存到该本地目录中
  hyperparameters=hp,
  project_name='my_horse_human_project'
)
# tuner.search()参数和model.fit()的参数差不多
tuner.search(train_generator,epochs=10,validation_data=validation_generator)

在训练的时候会出现OOM错误，需要将训练的卷积核数量调小一点

读取训练优化参数
best_hps=tuner.get_best_hyperparameters(1)[0]
print(best_hps.values)

然后根据参数将模型构建出来
model=tuner.hypermodel.build(best_hps)
model.summary()

除了卷积核的个数，卷积层和池化层的层数，全连接层的隐藏层神经元个数，其他的如学习率也可以进行调节
调整的参数越多，训练的时间也越长



# 机器视觉的应用
（1）糖尿病的诊断可以通过视网膜 的图片
（2）透彻影像辅助病理师进行癌症筛查


















