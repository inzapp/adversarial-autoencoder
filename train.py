"""
Authors : inzapp

Github url : https://github.com/inzapp/adversarial-autoencoder

Copyright (c) 2024 Inzapp

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import argparse

from aae import AdversarialAutoencoder, TrainingConfig

if __name__ == '__main__':
    config = TrainingConfig(
        train_image_path=r'/train_data/mnist/train/',
        model_name='model',
        generate_shape=(32, 32, 1),
        lr=0.0002,
        batch_size=32,
        latent_dim=32,
        view_grid_size=5,
        save_interval=2000,
        iterations=50000,
        training_view=False)

    parser = argparse.ArgumentParser()
    parser.add_argument('--decoder', type=str, default='', help='pretrained decoder model path')
    parser.add_argument('--generate', action='store_true', help='generate image using pretrained generator model')
    parser.add_argument('--grid', action='store_true', help='generate image grid using pretrained generator model')
    parser.add_argument('--save-count', type=int, default=0, help='count for save images')
    parser.add_argument('--grid-size', type=str, default='auto', help='square grid size for grid image saving')
    args = parser.parse_args()
    if args.decoder != '':
        config.pretrained_ae_d_path = args.decoder
    aae = AdversarialAutoencoder(config=config)
    if args.generate:
        aae.generate(save_count=args.save_count, save_grid_image=args.grid, grid_size=args.grid_size)
    else:
        aae.train()

