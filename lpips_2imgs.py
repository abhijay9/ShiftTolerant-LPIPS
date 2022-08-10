import argparse
import stlpips
import time

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-p0','--path0', type=str, default='./imgs/ex_ref.png')
parser.add_argument('-p1','--path1', type=str, default='./imgs/ex_p0.png')
parser.add_argument('-v','--version', type=str, default='0.1')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

opt = parser.parse_args()

## Initializing the model

##### LPIPS (v0.1)
# stlpips_metric = stlpips.LPIPS(net='alex')
## Distance: 0.722
#####

##### LPIPS trained from scratch
# stlpips_metric = stlpips.LPIPS(net='alex', variant="vanilla")
## Distance: 0.818
#####

##### ST-LPIPS (STv0.0)
# stlpips_metric = stlpips.LPIPS(net='alex', variant="antialiased")
## Distance: 0.682

stlpips_metric = stlpips.LPIPS(net="alex", variant="shift_tolerant")
## Distance: 0.778

# stlpips_metric = stlpips.LPIPS(net="vgg", variant="shift_tolerant")
## Distance: 0.652
#####

if(opt.use_gpu):
	stlpips_metric.cuda()

# # Load images
img0 = stlpips.im2tensor(stlpips.load_image(opt.path0)) # RGB image from [-1,1]
img1 = stlpips.im2tensor(stlpips.load_image(opt.path1))

if(opt.use_gpu):
	img0 = img0.cuda()
	img1 = img1.cuda()

# Compute distance
dist01 = stlpips_metric.forward(img0,img1)
print('Distance: %.3f'%dist01)


##### Load images in the same way as used for computing 2AFC score
# import torchvision.transforms as transforms
# from PIL import Image
# transform_list = []
# # transform_list.append(transforms.Scale(load_size)) # deprecated
# transform_list.append(transforms.Resize(64, Image.BICUBIC))
# transform_list += [transforms.ToTensor(),
# 	transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]

# transform = transforms.Compose(transform_list)

# img0 = Image.open(opt.path0).convert('RGB')
# img0 = transform(img0).unsqueeze(0)

# img1 = Image.open(opt.path1).convert('RGB')
# img1 = transform(img1).unsqueeze(0)

# dist01 = stlpips_metric.forward(img0,img1)
# print('Distance: %.3f'%dist01)