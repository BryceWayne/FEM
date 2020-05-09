from PIL import Image
import os, sys

def gif_creator(images, filename='Solution'):
    gifnumber = len([file for file in os.listdir(cwd) if file.endswith('.gif')])
    images[0].save(f'{filename}{gifnumber+1}.gif',
        save_all=True, append_images=images[1:], optimize=False, duration=80, loop=0)

cwd = os.getcwd()
path = os.path.join(cwd, 'gifs')
os.chdir(path)
cwd = os.getcwd()
filelist= [file for file in os.listdir(cwd) if file.endswith('.png')]
images = []
for file in filelist:
    images.append(Image.open(file))

if len(images) > 1:
    gif_creator(images)
else:
    print("No images.")

for file in filelist:
    os.remove(file)