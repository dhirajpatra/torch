import torch
import torchvision
from torchvision import transforms as T
import cv2
import cvzone


# create the model by pretrain one
model = torchvision.models.detection.ssd300_vgg16(weights="SSD300_VGG16_Weights.COCO_V1")
model.eval()

# get the class names as labels
classnames = []
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

# print(classnames[0])
# read the image and copy
image = cv2.imread('man_dog.jpg')
img = image.copy()
print(type(image))

# transform the image as a torchvision object
img_transform = T.ToTensor()
# transform the image
image = img_transform(image)
print(type(image))
print(f"Shape of tensor: {image.shape}")
print(f"Datatype of tensor: {image.dtype}")
print(f"Device tensor is stored on: {image.device}")

# predict the image class by model
with torch.no_grad():
    ypred = model([image])
    print(ypred[0].keys())

    # make the rectangle to show the predicted object's label
    bbox, scores, labels = ypred[0]['boxes'], ypred[0]['scores'], ypred[0]['labels']
    nums = torch.argwhere(scores > 0.80).shape[0]
    for i in range(nums):
        x, y, w, h = bbox[i].numpy().astype('int')
        cv2.rectangle(img, (x, y), (w, h), (0, 0, 255), 5)
        class_name = labels[i].numpy().astype('int')
        class_detected = classnames[class_name-1]
        # print(class_detected)
        cvzone.putTextRect(img, class_detected, [x, y+100], scale=2, border=2)

# stop by pressing any key
cv2.imshow('frame', img)
cv2.waitKey(0)
