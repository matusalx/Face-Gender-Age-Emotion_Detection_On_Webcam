import os
import cv2
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import torch

import torchvision.models as models
from torch import nn
from collections import OrderedDict
from torchvision import transforms
import argparse


data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128, 128)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


base_dir = os.path.dirname(os.path.realpath(__file__))
# Load gender model
gender_model_dir = os.path.join(base_dir, 'resnet_18_gender_model.pt')

gender_model = models.resnet18()
fc = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(512, 100)),
    ('relu', nn.ReLU()),
    ('fc2', nn.Linear(100, 2)),
    ('output', nn.LogSoftmax(dim=1))
]))
gender_model.fc = fc

gender_model.load_state_dict(torch.load(gender_model_dir))
gender_model.eval()


# Load age model
age_model_dir = os.path.join(base_dir, 'resnext50_age_model.pt')
age_model = models.resnext50_32x4d()
fc = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(2048, 500)),
    ('relu', nn.ReLU()),
    ('fc2', nn.Linear(500, 117)),
    ('output', nn.LogSoftmax(dim=1))
]))
age_model.fc = fc

age_model.load_state_dict(torch.load(age_model_dir))
age_model.eval()

# Load emotions model

emotions_model_dir = os.path.join(base_dir, 'resnext50_emotions_model.pt')
emotions_model = models.resnext50_32x4d()
fc = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(2048,500)),
    ('relu', nn.ReLU()),
    ('fc2', nn.Linear(500,7)),
    ('output', nn.LogSoftmax(dim=1))
]))
emotions_model.fc = fc

emotions_model.load_state_dict(torch.load(emotions_model_dir))
emotions_model.eval()


# Pytorch.Dataset by default  converts  greyscales images to RGB , dublicates them

def show_webcam(camera_number):
    mtcnn = MTCNN()
    emotions_map = {'anger': 0, 'contempt': 1, 'disgust': 2, 'fear': 3, 'happy': 4, 'sadness': 5, 'surprise': 6}
    emotions_map = dict([(value, key) for key, value in emotions_map.items()])
    cam = cv2.VideoCapture(camera_number)
    while True:
        ret_val, img = cam.read()
        if not ret_val:
            break
        img_cv2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_cv2)
        try:
            # detect face
            boxes, probs, landmarks = mtcnn.detect(img_pil, landmarks=True)
            x, y, x2, y2 = [int(x) for x in boxes[0]]
            margin_x = int((x2-x)*0.1)
            margin_y = int((y2 -y) * 0.2)
            x -= margin_x;  x2 += margin_x; y -= margin_y;  y2 += margin_y;
            cropped_face = img_cv2[x:x2, y:y2]
            # gender model
            tr_img = data_transform(cropped_face).float()
            tr_img = tr_img.unsqueeze(0)
            gender = gender_model(tr_img)
            gender = gender.argmax().item()
            gender = 'Female' if gender == 1 else 'Male'
            # detect age
            age = age_model(tr_img)
            age = age.argmax().item()
            # detect emotions
            emotion = emotions_model(tr_img)
            emotion = emotion.argmax().item()
            emotion = emotions_map[emotion]
            #draw results
            print(gender)
            #img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            img = draw(img, boxes, probs, landmarks, gender, age, emotion)
        except Exception as e:
            print(e)

        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cam.release()
    cv2.destroyAllWindows()


def draw(frame, boxes, probs, landmarks, gender, age, emotion):
    try:
        for box, prob, ld in zip(boxes, probs, landmarks):
            # Draw rectangle on frame
            x, y, x2, y2 = [int(x) for x in box]
            margin_x = int((x2-x)*0.1)
            margin_y = int((y2 -y) * 0.2)
            x -= margin_x; x2 += margin_x; y -= margin_y; y2 += margin_y;
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), thickness=2)
            # Show gender
            frame = cv2.putText(frame, str(gender), (x, y-20), cv2.FONT_HERSHEY_SIMPLEX,  1, (255, 0, 0), 1, cv2.LINE_AA)
            # Show age
            frame = cv2.putText(frame, 'Age: ' + str(age), (x, y2+30), cv2.FONT_HERSHEY_SIMPLEX,  1, (255, 0, 0), 1, cv2.LINE_AA)
            # emotion
            cv2.putText(frame, emotion, (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

    except Exception as e:
        print(e)
        pass

    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="web cam")

    parser.add_argument('-camera_number', '--camera_number', help="camera_number", type=int,
                        required=True, default=0)
    args = parser.parse_args()
    show_webcam(args.camera_number)





def junk_test_to_delete():
    mtcnn = MTCNN(select_largest=False, post_process=False, margin=40)
    mtcnn = MTCNN(margin=30)
    image = Image.open('42_1_0_20170111182452884.JPG')
    frame = cv2.imread('42_1_0_20170111182452884.JPG')
    frame = cv2.imread('49_1_0_20170109220611995.JPG')
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    plt.imshow(frame)

    data = np.asarray(image)

    face = mtcnn(frame)
    plt.imshow(face.permute(1, 2, 0).int().numpy())
    plt.axis('off');

    plt.imshow(img)

    plt.imshow(frame)




    img = Image.open(test_img_dir)
    data = np.asarray(img)
    #data = np.expand_dims(data, axis=0)
    tr_img = data_transform(data).float()
    tr_img = tr_img.unsqueeze(0)
    res = age_model(tr_img)
    print(res.argmax().item())



    img = Image.open(test_img_dir)
    data = np.asarray(img)
    tr_img = data_transform(data).float()
    tr_img = tr_img.repeat(3, 1, 1)
    tr_img = tr_img.unsqueeze(0)
    res = emotions_model(tr_img)
    print(res.argmax().item())
