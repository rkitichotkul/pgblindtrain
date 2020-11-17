"""
Denoise synthetic Poisson-Gaussian noisy images using selected models.
"""

import os, sys
sys.path.append(os.path.dirname(sys.path[0]))

import numpy as np
import torch
import util.denoiser as den
import util.general as gutil

# Parameters
datadir = 'data/test_for_paper'
savedir = 'result/model_demo'
std = 25 / 255
alpha = 0.01
denoiser_type_list = ['bm3d', 'remez', 'remez']  # specify denoiser type
denoiser_name_list = ['bm3d', 'remez_mse', 'remez_transfer']  # denoiser name for saving
modeldir_list = ['', 'data/model/remez_smse_1.pth', 'data/model/remez_spure_transfer_0.pth']  # path to models

if __name__ == '__main__':

    print('Poisson-Gaussian denoising demo')
    print('Gaussian std = {}, Poisson alpha = {}'.format(std, alpha))

    print('Preparing image paths...')
    loadlist, namelist, num_images = gutil.prepare_image_path(datadir)

    psnr_log = torch.zeros(num_images, len(denoiser_type_list) + 1)

    gutil.mkdir_if_not_exists(savedir)

    with torch.no_grad():

        for i, (loadname, name) in enumerate(zip(loadlist, namelist)):

                print('{}. image: {}'.format(i, name))
                image = gutil.read_image(loadname)

                if alpha != 0:
                    noisy_image = alpha * torch.poisson(image / alpha)
                else:
                    noisy_image = image.clone()
                if std != 0:
                    noisy_image += torch.normal(mean=torch.zeros_like(image), std=std)

                psnr_log[i, 0] = gutil.calc_psnr(noisy_image, image)
                print('Image {}, noisy, PSNR = {}'.format(name, psnr_log[i, 0]))
                gutil.save_image(noisy_image, os.path.join(savedir, '{}-noisy.png'.format(name)))

                for j, (denoiser_type, denoiser_name, modeldir) in enumerate(zip(denoiser_type_list, denoiser_name_list, modeldir_list)):

                    print('Setting up denoiser...')

                    if denoiser_type == 'bm3d':
                        print('Denoiser: BM3D')
                        denoiser = den.BM3D_denoiser()
                    elif denoiser_type == 'dncnn':
                        print('Denoiser: DnCNN from saved model: {}'.format(modeldir))
                        model = den.setup_DnCNN(modeldir, num_layers=17)
                        denoiser = den.CNN_denoiser(model)
                    elif denoiser_type == 'remez':
                        print('Denoiser: Remez from saved model: {}'.format(modeldir))
                        model = den.setup_Remez(modeldir, num_layers=20)
                        denoiser = den.CNN_denoiser(model)
                    else:
                        raise RuntimeError('Invalid denoiser type')

                    denoised_image = denoiser(noisy_image, std=std)

                    psnr_log[i, j + 1] = gutil.calc_psnr(denoised_image, image)
                    print('Image {}, PSNR = {}'.format(name, psnr_log[i, j + 1]))
                    gutil.save_image(denoised_image, os.path.join(savedir, '{}-{}.png'.format(name, denoiser_name)))


print('Saving PSNR log...')
torch.save(psnr_log, os.path.join(savedir, 'psnr.pt'))
print('Done!')
