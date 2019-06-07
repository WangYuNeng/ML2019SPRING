from create_model import create_model
import numpy as np

# credit to Ariel Yang
def save_weight_16(model, model_path):
    model.load_weights(model_path)
    w = model.get_weights()
    w_16 = list()
    for i in range(len(w)):
        w_16.append(w[i].astype(np.float16))
    np.savez_compressed(model_path[:-3]+"_16.npz", w_16=w_16)
    
def load_weight_16(model, model_path):
    w_16 = np.load(model_path)["w_16"]
    w = list()
    for i in range(w_16.shape[0]):
        w.append(w_16[i].astype(np.float32))
    model.set_weights(w) 
    return model
    
if __name__ == "__main__":
    model_path = "model/m-127-0.702-0.627.h5"
    model = create_model(input_shape=(48, 48, 1),
              alpha=0.5,
              depth_multiplier=1,
              classes=7)
    save_weight_16(model, model_path)
    