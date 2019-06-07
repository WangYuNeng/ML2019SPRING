from create_model import *
from transform import load_weight_16
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import sys

model_path = "m-127-0.702-0.627_16.npz"
test_x = np.load("test_feature.npy")
test_y = np.zeros((test_x.shape[0], 7))

train_datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
train_datagen.fit(test_x)

model = create_model(input_shape=(48, 48, 1),
              alpha=0.5,
              depth_multiplier=1,
              classes=7)
model = load_weight_16(model, model_path)

tta_steps = 10
bs = 512
test_y = []

for i in range(tta_steps):
    preds = model.predict_generator(train_datagen.flow(test_x, batch_size=bs, shuffle=False, seed=i), steps = len(test_x)/bs)
    test_y.append(preds)
    
test_y = np.mean(test_y, axis=0)
print(test_y.shape)
with open(sys.argv[1], 'w') as f:
    f.write("id,label\n")
    i = 0
    for row in test_y:
        max_l = 0
        label = 0
        for j in range(7):
            if row[j] > max_l:
                label = j
                max_l = row[j]
        line = str(i) + "," + str(label) + '\n'
        f.write(line)
        i += 1