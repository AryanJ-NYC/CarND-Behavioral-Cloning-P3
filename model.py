import csv
import cv2
from keras.layers import Conv2D, Cropping2D, Dense, Dropout, Flatten, Lambda
from keras.models import load_model, Sequential
from keras.utils import plot_model
import numpy as np
import os.path
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# read the CSV file
csv_filename = './data/driving_log.csv'
samples = []
with open(csv_filename) as csvfile:
    # learn if header is present
    has_header = csv.Sniffer().has_header(csvfile.read(1024))
    csvfile.seek(0)
    reader = csv.reader(csvfile)
    if has_header:
        next(reader)
    for line in reader:
        samples.append(line)

num_samples = len(samples)
images_per_sample = 3
print('Training on {} images total.'.format(num_samples*images_per_sample))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
print('{} are training images.'.format(len(train_samples)*images_per_sample))
print('{} are validation images.'.format(len(validation_samples)*images_per_sample))


def generator(samples, batch_size=32):
    num_samples = len(samples)
    # read the next {batch_size} lines
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            # get images and angles corresponding to next {batch_size} lines
            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    source_path = batch_sample[i]
                    filename = source_path.split('/')[-1]
                    relative_path = './data/IMG/' + filename
                    image = cv2.imread(relative_path)
                    images.append(image)

                    angle = float(batch_sample[3])

                    # if image from left camera, add 0.2 to angle
                    if i == 1:
                        angle += 0.2
                    # if image from right camera, subtract 0.2 to angle
                    if i == 2:
                        angle -= 0.2
                    angles.append(angle)

            # yield images and angles
            X = np.array(images)
            y = np.array(angles)
            assert len(X) == len(y)
            yield shuffle(X, y)


model_file_path = './models/model.h5'

if not os.path.exists(model_file_path):
    # architecture inspired by NVIDIA's DAVE-2:
    # http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    model = Sequential()
    model.add(Cropping2D(input_shape=(160, 320, 3), cropping=((70, 25), (0, 0))))
    model.add(Lambda(lambda x: x / 255.0 - 0.5))
    model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='relu'))
    model.add(Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='relu'))
    model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.add(Dropout(0.3))
    num_epochs = 6
else:
    model = load_model(model_file_path)
    num_epochs = 2
plot_model(model, show_shapes=True, show_layer_names=False)

batch_size = 32
train_generator = generator(train_samples)
validation_generator = generator(validation_samples)

model.compile(loss='mse', optimizer='adagrad')

model.fit_generator(train_generator,
                    steps_per_epoch=len(train_samples)/batch_size,
                    epochs=num_epochs,
                    verbose=2,
                    validation_data=validation_generator,
                    validation_steps=len(validation_samples)/batch_size
                    )

model.save(model_file_path)
print('Model saved.')
exit()
