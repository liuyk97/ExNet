import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from model.Net import CDNet
import model.Net as net
from model.Net import adaptive_instance_normalization
from Dataloader.CD_dataset import LEVID_CDset, WHU_CDset, SYSU_CDset, CDD_set
from tqdm import tqdm
from torch.utils.data import DataLoader


def style_transfer(vgg, decoder, content, style, alpha=1.0):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)


parser = argparse.ArgumentParser()
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default='/data/sdu08_lyk/data/style_cd/output',
                    help='Directory to save the output image(s)')
parser.add_argument('--dataset', default='LEVIR-CD')
parser.add_argument('--decoder', type=str, default='logs/LEVIR-CD/style/best_model_decoder.pth')
# Advanced options
parser.add_argument('--preserve_color', action='store_true',
                    help='If specified, preserve color of the content image')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='The weight that controls the degree of \
                             stylization. Should be between 0 and 1')
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True, parents=True)

decoder = net.decoder
vgg = net.vgg

decoder.eval()
vgg.eval()

decoder.load_state_dict(torch.load(args.decoder))

vgg.to(device)
decoder.to(device)

print("----Loading Datasets----")
if args.dataset == 'LEVIR-CD':
    test_set = LEVID_CDset(mode='test')
    print("get {} images from LEVIR-CD test set".format(len(test_set)))

test_loader = DataLoader(test_set, batch_size=1, num_workers=2, shuffle=False)
pbar = tqdm(test_loader, unit='pair')
for step, data in enumerate(pbar):
    image_A = data['A'].cuda()
    image_B = data['B'].cuda()
    path = data['path']
    with torch.no_grad():
        output = style_transfer(vgg, decoder, image_A, image_B,
                                args.alpha)
    output = output.cpu()

    output_name = args.output + '/{:s}_stylized{:s}'.format(
        path[0], args.save_ext)
    save_image(output, str(output_name))
