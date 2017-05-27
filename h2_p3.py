from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from sklearn.decomposition import TruncatedSVD
from scipy.misc import imread, imresize
from matplotlib.colors import rgb_to_hsv
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import heapq, math, os

IMAGE_NET_DIR = 'ImageNet' # name of directory containing all image net images
CATEGORIES = ['animal', 'fungus', 'geological', 'person', 'plant', 'sport'] # as defined by assignment
IMAGE_RESIZE = (100, 100) # uniformly resize all images to 100 x 100
K = 5 # as defined by assignment
np.random.seed(0) # set seed so random results are reproducible

# function to generate confusion matrix
def generate_confusion_matrix(X_train, X_test, y_train, y_test, title, metric='euclidean'):
    if metric == 'pearson':
        y_pred = np.zeros(len(y_test))
        for i in range(X_test.shape[0]):
            row_i = X_test[i]
            similarity_heap = [] # use heap to track greatest similarity values in order to find closest images
            for j in range(X_train.shape[0]):
                # compute pearson correlation coefficient between each test/train image pair
                row_j = X_train[j]
                diff_i, diff_j = row_i - np.mean(row_i), row_j - np.mean(row_j)
                top = np.sum(diff_i * diff_j)
                bot_i = np.sum(diff_i**2)
                bot_j = np.sum(diff_j**2)

                # pearson correlation formula based on: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
                similarity = top / (math.sqrt(bot_i) * math.sqrt(bot_j))
                heapq.heappush(similarity_heap, (-similarity, y_train[j])) # save similarity and corresponding image label
            # use similarity matrix to predict test set images
            k_similar = heapq.nsmallest(K, similarity_heap) # get smallest k since similarity values saved as negative
            k_labels = [label for _, label in k_similar] # get labels associated with each k similarity
            _, label_counts = np.unique(k_labels, return_counts=True) # count the knn voting
            max_label_count = max(label_counts) # find the label with highest vote
            label_counts_index = [index for index in range(len(label_counts)) if (label_counts[index] == max_label_count)]
            if len(label_counts_index) == 1: # if only one label with most votes, use it
                index = label_counts_index[0]
            else: # otherwise if there are ties, randomly select a label
                index = np.random.choice(label_counts_index)
            y_pred[i] = k_labels[index]
    else: # 'euclidean' metric
        knn_classifier = KNeighborsClassifier(n_neighbors=K, n_jobs=-1) # defaults to euclidean distance metric
        knn_classifier.fit(X_train, y_train)
        y_pred = knn_classifier.predict(X_test)

    print('Labels: ' + ', '.join(['{:d}='.format(i) + CATEGORIES[i] for i in range(len(CATEGORIES))]))
    accuracy = np.sum(y_test == y_pred) / len(y_test) # sum up number of matches between test and prediction
    print('{:s} (\'{:s}\' metric) overall accuracy = {:f}'.format(title, metric, accuracy))

    # plot confusion matrix as matplotlib table
    conf_matrix = confusion_matrix(y_test, y_pred)
    labels = list(range(len(CATEGORIES)))
    row_labels = ['Predicted {:d}:'.format(l) for l in labels]
    col_labels = ['True {:d}:'.format(l) for l in labels]
    plt.close()
    plt.title('{:s} (\'{:s}\' metric) Confusion Matrix'.format(title, metric))
    plt.table(cellText=conf_matrix, cellLoc='center', rowLabels=row_labels,
                           colLabels=col_labels, loc='center', bbox=[0, 0, 1, 1])
    plt.xticks([])
    plt.yticks([])
    plt.show()

# extract image info
autoencoder_image_array, svd_image_array, rgb_image_array, hsv_image_array, labels = [], [], [], [], []
label = 0
for filename in sorted(os.listdir(IMAGE_NET_DIR)):
    path = IMAGE_NET_DIR + '/' + filename
    if (filename in CATEGORIES) and os.path.isdir(path):
        for f in os.listdir(path):
            if 'jpeg' in f.lower() or 'jpg' in f.lower(): # only read jpeg/jpg image files
                labels.append(label) # keep track of the image label
                image_rgb = imread(path + '/' + f, mode='RGB') # read image as rgb

                # get histogram for each rgb color channel and concatenate
                rgb_image = []
                for i in range(image_rgb.shape[2]):
                    hist, _ = np.histogram(image_rgb[:, :, i], bins=256)
                    rgb_image.extend(hist)
                rgb_image_array.append(rgb_image)

                # get histogram for each hsv color channel and concatenate
                image_hsv = rgb_to_hsv(image_rgb)
                hsv_image = []
                hist, _ = np.histogram(image_hsv[:, :, 0], bins=180) # hue range [0,179]
                hsv_image.extend(hist)
                for i in range(1, image_hsv.shape[2]):
                    hist, _ = np.histogram(image_hsv[:, :, i], bins=256) # saturation/value range [0,255]
                    rgb_image.extend(hist)
                hsv_image_array.append(hsv_image)

                # resize image if not already the desired down-sample dimensions
                if image_rgb.shape[0] != IMAGE_RESIZE[0] or image_rgb.shape[1] != IMAGE_RESIZE[1]:
                    image_rgb = imresize(image_rgb, IMAGE_RESIZE)

                # keep image array without flattening
                autoencoder_image_array.append(image_rgb)

                # form 1-d vector by flattening each color channel and concatenating
                svd_image = []
                for i in range(image_rgb.shape[2]):
                    svd_image.extend(image_rgb[:, :, i].flatten(order='F')) # flatten in column-major order
                svd_image_array.append(svd_image)
        label += 1 # increment label after each image directory
labels = np.array(labels)

def get_encoded_representations(X_train, X_test):
    # set up convolutional autoencoder with 3 layers
    input_img = Input(shape=(IMAGE_RESIZE[0], IMAGE_RESIZE[1], 3))

    x = Convolution2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Convolution2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Convolution2D(16, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    x = Convolution2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(32, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Convolution2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    autoencoder.fit(X_train, X_train, epochs=10, verbose=0, batch_size=128, shuffle=True)

    encoder = Model(input_img, encoded) # after training autoencoder, just need to user encoder part
    return (encoder.predict(X_train), encoder.predict(X_test)) # return encoded representations of train/test sets

autoencoder_image_array = np.array(autoencoder_image_array).astype('float32') / 255. # normalize image array

# split data 80 train/20 test
X_train, X_test, y_train, y_test = train_test_split(autoencoder_image_array, labels, test_size=0.2, random_state=0)

# get encoded representations (reduced dimensionality) of train and test sets
X_train, X_test = get_encoded_representations(X_train, X_test)

# reshape to flatten each image in order to calculate similarity
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2] * X_train.shape[3])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2] * X_test.shape[3])

# euclidean metric
generate_confusion_matrix(X_train, X_test, y_train, y_test, 'Autoencoder', metric='euclidean')

# pearson metric
generate_confusion_matrix(X_train, X_test, y_train, y_test, 'Autoencoder', metric='pearson')

SVD_K = 40 # reference: https://www.math.cuhk.edu.hk/~lmlui/CaoSVDintro.pdf

# reduce image data with svd and split 80 train/20 test
svd_image_array = np.array(svd_image_array)
svd_image_array = TruncatedSVD(n_components=SVD_K, algorithm='arpack').fit_transform(svd_image_array)
X_train, X_test, y_train, y_test = train_test_split(svd_image_array, labels, test_size=0.2, random_state=0)

# euclidean metric
generate_confusion_matrix(X_train, X_test, y_train, y_test, 'SVD', metric='euclidean')

# pearson metric
generate_confusion_matrix(X_train, X_test, y_train, y_test, 'SVD', metric='pearson')

# split data 80 train/20 test
rgb_image_array = np.array(rgb_image_array)
X_train, X_test, y_train, y_test = train_test_split(rgb_image_array, labels, test_size=0.2, random_state=0)

# euclidean metric
generate_confusion_matrix(X_train, X_test, y_train, y_test, 'RGB Histograms', metric='euclidean')

# pearson metric
generate_confusion_matrix(X_train, X_test, y_train, y_test, 'RGB Histograms', metric='pearson')

# split data 80 train/20 test
hsv_image_array = np.array(hsv_image_array)
X_train, X_test, y_train, y_test = train_test_split(hsv_image_array, labels, test_size=0.2, random_state=0)

# euclidean metric
generate_confusion_matrix(X_train, X_test, y_train, y_test, 'HSV Histograms', metric='euclidean')

# pearson metric
generate_confusion_matrix(X_train, X_test, y_train, y_test, 'HSV Histograms', metric='pearson')
