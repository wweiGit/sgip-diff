import net
import torch
import os
from face_alignment import align
import numpy as np
from PIL import Image
adaface_models = {
    'ir_50':"/root/autodl-tmp/face_recognition/AdaFace/adaface_ir50_ms1mv2.ckpt",
}

def load_pretrained_model(architecture='ir_50'):
    # load model and pretrained statedict
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    statedict = torch.load(adaface_models[architecture])['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model

def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
    tensor = torch.tensor([brg_img.transpose(2,0,1)]).float()
    return tensor

if __name__ == '__main__':
    model = load_pretrained_model('ir_50')
    #feature, norm = model(torch.randn(2,3,112,112))
    test_image_path = '/root/autodl-tmp/project/SGIP-dev-v2/prepare_data/swinir_s1_p0195_e17/stage1_val'
    out_path = "/root/autodl-tmp/project/SGIP-dev-v2/prepare_data/swinir_s1_p0195_e17/id_val"
    os.makedirs(out_path,exist_ok=True)
    for fname in sorted(os.listdir(test_image_path)):
        path = os.path.join(test_image_path, fname)
        img = Image.open(path).convert('RGB')
        img = img.resize((112,112))
        bgr_tensor_input = to_input(img)
        feature, _ = model(bgr_tensor_input)
        feature = feature.reshape(512)
        torch.save(feature, os.path.join(out_path, fname.split(".")[0]+".bin"))
