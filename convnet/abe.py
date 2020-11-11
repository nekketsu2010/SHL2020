from convnet_drawer import Model, Conv2D, MaxPooling2D, Flatten, Dense
from pptx_util import save_model_to_pptx
from matplotlib_util import save_model_to_file

model = Model(input_shape=(500, 5, 1))
model.add(Conv2D(32, (4, 5), (4, 1)))
model.add(Conv2D(32, (5, 1), (1, 1)))
model.add(MaxPooling2D((4, 1)))
model.add(Flatten())
model.add(Dense(512))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(8))
# save as svg file
model.save_fig("abe.svg")

# save as pptx file
save_model_to_pptx(model, "abe.pptx")

# save via matplotlib
save_model_to_file(model, "abe.pdf")