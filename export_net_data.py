import os
import numpy as np
import argparse
import json
import torch
import shutil

from main import main as quantlab_main

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--exp_id', help='experiment identification', type=int, default=999)
parser.add_argument('-s', '--sample', help='index of the sample', type=int, default=0)
parser.add_argument('--train', help='Train network', action='store_true')

args = parser.parse_args()

exp_folder = f'BCI-CompIV-2a/logs/exp{args.exp_id:03}'
output_file = 'export/{}.npz'
output_config_file = "export/config.json"

# train the network
if args.train:
    # delete the exp folder
    try:
        shutil.rmtree(exp_folder)
        print('exp folder was deleted!')
    except:
        print('exp folder does not exist, skipping deletion')
    quantlab_main('BCI-CompIV-2a', 'EEGNet', exp_id=args.exp_id, ckpt_every=1, num_workers=1,
                  do_validPreTrain=False, use_single_gpu=True)

# import the EEGnet folder
exec(open('quantlab/BCI-CompIV-2a/EEGNet/preprocess.py').read())
exec(open('quantlab/BCI-CompIV-2a/EEGNet/eegnet.py').read())

exp_folder = f'BCI-CompIV-2a/logs/exp{args.exp_id:03}'

# load the configuration file
with open(f'{exp_folder}/config.json') as _f:
    config = json.load(_f)

# get data loader
_, _, dataset = load_data_sets('BCI-CompIV-2a/data', config['treat']['data'])

# load the model
ckpts = os.listdir(f'{exp_folder}/saves')
ckpts = [x for x in ckpts if "epoch" in x]
ckpts.sort()
last_epoch = int(ckpts[-1].replace('epoch', '').replace('.ckpt', ''))
ckpt = torch.load(f'{exp_folder}/saves/{ckpts[-1]}')
model = EEGNet(**config['indiv']['net']['params'])
model.load_state_dict(ckpt['indiv']['net'])
for module in model.steController.modules:
    module.started = True

# export all weights
weights = {key: value.cpu().detach().numpy() for key, value in ckpt['indiv']['net'].items()}
np.savez(output_file.format("net"), **weights)

# save input data
np.savez(output_file.format("input"), input=dataset[args.sample][0].numpy())

# prepare verification data
verification = {}
# do forward pass and compute the result of the network
model.train(False)
with torch.no_grad():
    x = dataset[args.sample][0]
    verification['input'] = x.numpy()
    x = x.reshape(1, 1, 22, 1125)
    x = model.quant1(x)
    verification['input_quant'] = x.numpy()
    x = model.conv1_pad(x)
    x = model.conv1(x)
    verification['layer1_conv_out'] = x.numpy()
    x = model.batch_norm1(x)
    verification['layer1_bn_out'] = x.numpy()
    x = model.quant2(x)
    verification['layer1_activ'] = x.numpy()
    x = model.conv2(x)
    verification['layer2_conv_out'] = x.numpy()
    x = model.batch_norm2(x)
    verification['layer2_bn_out'] = x.numpy()
    x = model.activation1(x)
    verification['layer2_relu_out'] = x.numpy()
    x = model.pool1(x)
    verification['layer2_pool_out'] = x.numpy()
    x = model.quant3(x)
    verification['layer2_activ'] = x.numpy()
    x = model.sep_conv_pad(x)
    x = model.sep_conv1(x)
    verification['layer3_conv_out'] = x.numpy()
    x = model.quant4(x)
    verification['layer3_activ'] = x.numpy()
    x = model.sep_conv2(x)
    verification['layer4_conv_out'] = x.numpy()
    x = model.batch_norm3(x)
    verification['layer4_bn_out'] = x.numpy()
    x = model.activation2(x)
    verification['layer4_relu_out'] = x.numpy()
    x = model.pool2(x)
    verification['layer4_pool_out'] = x.numpy()
    x = model.quant5(x)
    verification['layer4_activ'] = x.numpy()
    x = model.flatten(x)
    x = model.fc(x)
    verification['output'] = x.numpy()
    x = model.quant6(x)
    verification['output_quant'] = x.numpy()

np.savez(output_file.format("verification"), **verification)

# copy the configuration file to the export folder
shutil.copyfile(f'{exp_folder}/config.json', output_config_file)
