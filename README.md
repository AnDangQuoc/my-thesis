## File Name format

<image_name>_<image_type>_<image_layer>.npz


image_name: BRATS001 

image_type:
- 000: t1
- 001: t2
- 002: t1ce
- 003: flair
- 004: segment mask

image_layer:
- 001 
- 002 
...
- 150

How to dump image:
- Scan each layer
- Dump numpy matrix to npz file

Things need to consider

- Image too small
- Mask too small?