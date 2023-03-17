from flask import Flask, request, send_file
import traceback
import os
from test_options import TestOptions
from aligned_dataset import AlignedDataset
from pix2pix_model import Pix2PixModel
import util
import torch
from PIL import Image


app = Flask(__name__)

opt = TestOptions().parse()  # get test options
opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
opt.preprocess = 'none'
opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
model = Pix2PixModel(opt)      # create a model given opt.model and other options
model.setup(opt)               # regular setup: load and print networks; create schedulers
model.eval()


@app.before_first_request
def initialize():
    app.logger.debug('Initializing code')


@app.route('/get_embed', methods=['POST'])
def get_embed():
    image_data = request.data
    with open('/app/inputImage/image_server.tif', 'wb') as f:
        f.write(image_data)
    dataset_class = AlignedDataset(opt)
    dataset = torch.utils.data.DataLoader(
        dataset_class,
        batch_size=opt.batch_size,
        shuffle=not opt.serial_batches,
        num_workers=int(opt.num_threads))
    for i, data in enumerate(dataset):
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        
        im = util.tensor2im(visuals["fake_B"])
        image_pil = Image.fromarray(im)
        
        save_path = f"/app/outputImage/processed_image.tif"
        image_pil.save(save_path)
    return 'Image received'



@app.route('/get_image', methods=['GET'])
def get_image():
    if os.path.exists('/app/outputImage/processed_image.tif'):
        with open('/app/outputImage/processed_image.tif', 'rb') as f:
            image_data = f.read()
        return send_file('/app/outputImage/processed_image.tif',mimetype='image/tif')
    else:
        # If the image file does not exist, return a 404 error
        return 'Image not found', 404


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=6000)
