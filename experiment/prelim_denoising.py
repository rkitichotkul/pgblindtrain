import os, sys
sys.path.append(os.path.dirname(sys.path[0]))

import numpy as np
import torch
import algorithm.denoiser as den
import util.general as gutil

# Parameters
datadir = 'data/test'
savedir = 'result/remez_smse_1_nat'
model = 'remez'
oneim = True
std_namelist = np.array([25])
std_list = std_namelist / 255
num_photons = np.array([100])
save_image = True
alpha_list = 1 / num_photons

if __name__ == '__main__':

    print('Setting up denoiser...')
    if model == 'bm3d':
        print('Denoiser: BM3D')
        denoiser = den.BM3D_denoiser()
    elif model == 'dncnn':
        modeldir = 'data/model/std25-0.pth'
        print('Denoiser: DnCNN from saved model: {}'.format(modeldir))
        model = den.setup_DnCNN(modeldir, num_layers=17)
        denoiser = den.DnCNN_denoiser(model)
    else:
        raise RuntimeError('Invalid denoiser type')

    print('Preparing image paths...')
    loadlist, namelist, num_images = gutil.prepare_image_path(datadir, oneim=oneim)

    psnr_log = torch.zeros(num_images, len(std_list), len(alpha_list))

    gutil.mkdir_if_not_exists(savedir)

    with torch.no_grad():
        for i, (loadname, name) in enumerate(zip(loadlist, namelist)):

            print('{}. image: {}'.format(i, name))
            image = gutil.read_image(loadname)

            for j, std in enumerate(std_list):
                for k, alpha in enumerate(alpha_list):

                    if alpha != 0:
                        noisy_image = alpha * torch.poisson(image / alpha)
                    else:
                        noisy_image = image.clone()
                    if std != 0:
                        noisy_image += torch.normal(mean=torch.zeros_like(image), std=std)

                    denoised_image = denoiser(noisy_image, std=std)

                    psnr_log[i, j, k] = gutil.calc_psnr(denoised_image, image)
                    print('Image {}, std {}, alpha {}, PSNR = {}'.format(name, std_namelist[j], alpha, psnr_log[i, j, k]))
                    if save_image:
                        gutil.save_image(noisy_image, os.path.join(savedir, '{}-noisy-std{}-photon{}.png'.format(name, std_namelist[j], num_photons[k])))
                        gutil.save_image(denoised_image, os.path.join(savedir, '{}-denoised-std{}-photon{}.png'.format(name, std_namelist[j], num_photons[k])))


print('Saving PSNR log...')
torch.save(psnr_log, os.path.join(savedir, 'psnr.pt'))
print('Done!')
