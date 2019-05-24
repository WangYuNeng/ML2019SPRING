from sklearn import cluster
from sklearn.decomposition import PCA
from keras.models import Sequential, Model, load_model
import skimage
import numpy as np
import os
import sys

img_dir = sys.argv[1]
test_case_file = sys.argv[2]

file_names = list()
for f in os.listdir(img_dir):
    if f.endswith(".jpg") and not f.startswith("."):
        file_names.append(os.path.join(img_dir, f))
num_files = len(file_names)
file_names.sort()

train_x = list()
for f in file_names:
    train_x.append(skimage.io.imread(f, as_gray=False).reshape((32,32,3)))

train_x = np.array(train_x) / 255

autoencoder = load_model("best.h5")

latent_layer = "latent"
laten_model = Model(inputs=autoencoder.input,
                    outputs=autoencoder.get_layer(latent_layer).output)
latent_codes = laten_model.predict(train_x)
latent_codes = latent_codes.reshape((latent_codes.shape[0], -1))


pca = PCA(n_components=256,whiten=True,random_state=0)
pca_imgs = pca.fit_transform(latent_codes)
y_pred = cluster.KMeans(n_clusters=2,max_iter=4000, random_state=0, n_jobs=4).fit_predict(pca_imgs)

import pandas as pd
df = pd.read_csv(test_case_file)
i1 = df["image1_name"]
i2 = df["image2_name"]
output_path = sys.argv[3]

with open(output_path, "w") as f:
  f.write("id,label\n")
  for i in range(len(i1)):
    label = 0;
    if y_pred[int(i1[i])-1] == y_pred[int(i2[i])-1]:
        label = 1
    line = str(i) + "," + str(label) + '\n'
    f.write(line)

    