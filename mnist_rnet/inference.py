import sys
import mnist_rnet
import os

_dir = os.path.dirname(mnist_rnet.__file__)
sys.path.append(_dir)

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, Pad
from mnist_rnet.models.model import MnistResNet

# save_3/49
try:
    save_path = os.path.join(_dir, "_model_ep_78.t7")
    print("Initialized from", save_path)
except KeyError:
    raise ValueError("Please set MNIST_MODEL_PATH environment varible to saved onmt model path, example:\
            export MNIST_MODEL_PATH='models/onmt/demo-model_step_989120.pt'")

# restore model
# model = torch.load(save_path)
model = MnistResNet()
model.load_state_dict(torch.load(save_path))

model.to(torch.device('cpu'))
torch.set_grad_enabled(False)
model.eval()

# transformations
valid_data_transform = Compose([
    # RandomAffine(degrees=6, translate=(0.15, 0.15)),
    Pad(padding=(15, 15, 15, 15), fill=255),
    Resize((36, 36)),
    ToTensor(),
])


def normalize_label(int_predicted):
    if int_predicted == 10:
        return "¥"
    else:
        return str(int_predicted)


def predict(img_fn, bias_to_yen=False):
    if type(img_fn) is str:
        image = Image.open(img_fn).convert('L')
    else:
        image = Image.fromarray(img_fn).convert('L')

    transformed_image = valid_data_transform(image)
    transformed_image = torch.unsqueeze(transformed_image, dim=0)

    outputs = model(transformed_image)

    if bias_to_yen:  # "_0" in img_fn and '05_' in img_fn:
        outputs[-1][10] += outputs.std()

    # outputs = F.softmax(outputs,dim=-1)
    probability, predicted_classes = torch.max(outputs, 1)

    # convert to numpy
    probability = probability.numpy()[0]
    predicted_classes = predicted_classes.numpy()[0]

    return normalize_label(predicted_classes), probability


import xlsxwriter, glob
import json


def _update_table_dct_v1(table, image_fn, value):
    _image_fn = image_fn.split('/')[-1]
    _id = _image_fn.rfind('_')

    _key = _image_fn[:_id]

    if _key in table:
        table[_key] += [(_image_fn, value)]
    else:
        table[_key] = [(_image_fn, value)]


def predict_from_folder(folder_name, output_excel_fn="./report_2.xlsx",
                        labels_fn="/home/vanph/Desktop/up/flax_prod_ffg/kan_test_2/_debug02/_labels.json"):
    image_fns = glob.glob("%s/*.png" % folder_name)
    labels = json.load(open(labels_fn, 'r'))

    print("Total:", len(image_fns), " images")

    workbook = xlsxwriter.Workbook(output_excel_fn)
    worksheet = workbook.add_worksheet(name='input_and_predict')

    worksheet.set_column('A:A', 30)
    worksheet.set_column('B:B', 50)
    worksheet.set_column('C:C', 30)
    worksheet.set_column('D:D', 30)

    MAX_LEN = 3000
    image_fns = image_fns[:MAX_LEN]
    _table_pred = {}
    _table_gt = {}

    _count = 0
    for id, image_fn in enumerate(sorted(image_fns)):
        worksheet.set_row(id, 80)
        predicted_classes, _ = predict(image_fn)

        _image_fn = image_fn.split('/')[-1]
        _image_lbl = labels.get(_image_fn, "-1")
        # _table[_image_fn] = (str(predicted_classes), str(_image_lbl))

        _update_table_dct_v1(_table_pred, image_fn, predicted_classes)

        # test
        if _image_lbl != '-1' and _image_lbl != str(predicted_classes):
            # print (image_fn, predicted_classes, _image_lbl)

            id = _count  # must be commented

            worksheet.write("A%d" % (id + 1), image_fn)
            worksheet.insert_image("B%d" % (id + 1), image_fn,
                                   {'x_scale': 1.0, 'y_scale': 1.0, 'x_offset': 5, 'y_offset': 5})
            worksheet.write("C%d" % (id + 1), predicted_classes)
            worksheet.write("D%d" % (id + 1), _image_lbl)

            _count += 1  # must be commented

    workbook.close()

    return _table_pred


if __name__ == "__main__":
    # print (predict(img_fn='/home/vanph/Desktop/debug2/05_Img00048_0.png'))
    # exit()

    _table_pred = predict_from_folder(folder_name="/home/vanph/Desktop/debug2")

    _final_table_pred = {}
    for k, vs in _table_pred.items():
        v = "".join([_[1] for _ in vs])
        _final_table_pred[k] = v

    json.dump(_final_table_pred, open("./pred.json", 'w'))

    _pred_json = json.load(open("./pred.json", "r"))
    _gt_json = json.load(open("/home/vanph/Desktop/up/flax_prod_ffg/kan_test_2/_debug02/_labels.json", "r"))

    _ks = list(_pred_json.keys())
    for _k in _ks:
        _pred, _gt = _pred_json[_k], _gt_json[_k]

        _pred = _pred.replace("¥", "")

        if _pred != _gt:
            print(_k, "pred: ", _pred, "gt:", _gt)

    print("finished")
