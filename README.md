# wgan_tensorflow
A tensorflow implementation of WGAN

This version of wgan can generate square images with 
training images(eg. MNIST, Chinese characters image) feed.
- training images should be prepocessed(resize to square, rescale to 1, etc.) before like that in 
`helper_functions.py`. And a data_loading function should 
be implemented in `helper_functions.py` as well.
- run `main.py` with parameters:

        --batch_size 32
        --image_size 32
        --epoch 50
        --learning_rate 0.0001
        --learning_rate_g 0.001
        --c_dim 1 (3 for RGB image, 1 for gray image)
        --d_iters 1
        --g_iters 1
        
