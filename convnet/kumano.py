from convnet_drawer import Model, Conv2D, MaxPooling2D, Flatten, Dense
from pptx_util import save_model_to_pptx
from matplotlib_util import save_model_to_file

# model = Model(input_shape=(499, 2, 1))
# model.add(Conv2D(32, (3, 1), (1, 1)))
# model.add(MaxPooling2D((2, 1)))
# model.add(Conv2D(4, (3, 1), (1, 1), padding="valid"))
# model.add(MaxPooling2D((2, 1)))
# model.add(Flatten())
# model.add(Dense(300))

# model = Model(input_shape=(101, 31, 2))
# model.add(Conv2D(64, (6, 6), (1, 1)))
# model.add(MaxPooling2D((3, 3)))
# model.add(Flatten())
# model.add(Dense(300))

model = Model(input_shape=(600, 1, 1))
model.add(Flatten())
model.add(Dense(200))
model.add(Dense(8))

# save as svg file
model.save_fig("kumano2.svg")

# save as pptx file
save_model_to_pptx(model, "kumano2.pptx")

# save via matplotlib
save_model_to_file(model, "kumano2.pdf")