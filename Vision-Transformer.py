import tensorflow as tf
from einops.layers.tensorflow import Rearrange
import math
import six

def gelu(x):
    cdf = 0.5 * (1.0 + tf.tanh(
        (math.sqrt(2 / math.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf

def get_activation(identifier):
    if isinstance(identifier, six.string_types):
        name_to_fn = {"gelu": gelu}
        identifier = str(identifier).lower()
        if identifier in name_to_fn:
            return tf.keras.activations.get(name_to_fn[identifier])
    return tf.keras.activations.get(identifier)

class Residual(tf.keras.Model):
    def __init__(self,fn):
        super(Residual, self).__init__()
        self.fn = fn
    def call(self, inputs, training=None, mask=None):
        return self.fn(inputs) + inputs

class PreNorm(tf.keras.Model):
    def __init__(self,dim,fn):
        super(PreNorm, self).__init__()
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.fn = fn
    def call(self, inputs, training=None, mask=None):
        return self.fn(self.norm(inputs))

class FeedForward(tf.keras.Model):
    def __init__(self,dim, hidden_dim):
        super(FeedForward, self).__init__()
        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation=get_activation('gelu')),
            tf.keras.layers.Dense(dim)
        ])
    def call(self, inputs, training=None, mask=None):
        return self.net(inputs)


class Attention(tf.keras.Model):

    def __init__(self, dim, heads = 8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = tf.keras.layers.Dense(dim * 3, use_bias=False)
        self.to_out = tf.keras.layers.Dense(dim)

        self.rearrange_qkv = Rearrange('b n (qkv h d) -> qkv b h n d', qkv = 3, h = self.heads)
        self.rearrange_out = Rearrange('b h n d -> b n (h d)')

    def call(self, x):
        qkv = self.to_qkv(x)
        qkv = self.rearrange_qkv(qkv)
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]

        dots = tf.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = tf.nn.softmax(dots, axis=-1)

        out = tf.einsum('bhij,bhjd->bhid', attn, v)
        out = self.rearrange_out(out)
        out = self.to_out(out)
        return out


class Transformer(tf.keras.Model):

    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend([
                Residual(PreNorm(dim, Attention(dim, heads = heads))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim)))
            ])
        self.net = tf.keras.Sequential(layers)

    def call(self, x):
        return self.net(x)

class ViT(tf.keras.Model):
    def __init__(self, image_size, patch_size, dim,num_classes, depth, heads,mlp_dim,channels=3):
        super(ViT, self).__init__()
        assert image_size % patch_size == 0, 'image dim must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.dim = dim
        self.pos_embedding = self.add_weight("position_embeddings",
                                             shape=[num_patches + 1, dim],
                                             initializer=tf.keras.initializers.RandomNormal(),
                                             dtype = tf.float32)
        self.patch_to_embedding = tf.keras.layers.Dense(dim)
        self.cls_token = self.add_weight("cls_token",
                                         shape=[1,1,dim], initializer=tf.keras.initializers.RandomNormal(),
                                         dtype= tf.float32)
        # self.transpose = Rearrange('b w h c -> b c w h')
        self.rearrange = Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
        self.transformer = Transformer(dim, depth, heads, mlp_dim)
        self.to_cls_token = tf.identity
        self.mlp_head = tf.keras.Sequential([tf.keras.layers.Dense(mlp_dim, activation=get_activation('gelu')),
                                             tf.keras.layers.Dense(num_classes,activation='softmax')])

    def call(self, inputs, training=None, mask=None):
        shape = tf.shape(inputs)
        # inputs = self.transpose(inputs)
        x = self.rearrange(inputs)

        x = self.patch_to_embedding(x)
        cls_tokens = tf.broadcast_to(self.cls_token, (shape[0],1,self.dim))
        x = tf.concat((cls_tokens, x),axis=1)
        x += self.pos_embedding
        x = self.transformer(x)

        x = self.to_cls_token(x[:,0])

        return self.mlp_head(x)




net = ViT(28,7,4,10,64,4,128,1)
# x = tf.ones(shape=(1,28,28,1))
# print(net(x).shape)


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print(x_train.shape)

x_train = x_train[...,tf.newaxis]
x_test = x_test[...,tf.newaxis]
print(x_train.shape)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_dataset = train_dataset.shuffle(10000).batch(32)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(32)


loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name = 'train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc')

EPOCHS = 20
for epoch in range(EPOCHS):
    for data, label in train_dataset:
        with tf.GradientTape() as tape:
            pred = net(data)
            loss = loss_obj(label, pred)

        gradient = tape.gradient(loss, net.trainable_variables)
        optimizer.apply_gradients(zip(gradient, net.trainable_variables))
        train_loss(loss)
        train_accuracy(label,pred)

        template = 'EPOCH {}, LOSS:{}, Acc:{}\n'
        print(template.format(epoch+1,
                              train_loss.result(),
                              train_accuracy.result() * 100))

        train_loss.reset_states()
        train_accuracy.reset_states()
