from convnet_drawer import Model, Conv2D, MaxPooling2D, Flatten, Dense
from pptx_util import save_model_to_pptx
from matplotlib_util import save_model_to_file

# model = Model(input_shape=(33, 55, 1))
# model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='valid'))
# model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
# model.add(Conv2D(64, (3, 3), padding="same"))
# model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
# model.add(Conv2D(64, (3, 3), padding="same"))
# model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
# model.add(Conv2D(64, (3, 3), padding="same"))
# model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
# model.add(Flatten())
# model.add(Dense(128))
# model.add(Dense(64))
# model.add(Dense(16))

model = Model(input_shape=(96, 1, 1))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(16))
model.add(Dense(8))
# save as svg file
model.save_fig("example2.svg")

# save as pptx file
save_model_to_pptx(model, "example2.pptx")

# save via matplotlib
save_model_to_file(model, "example2.pdf")