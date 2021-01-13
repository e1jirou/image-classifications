import matplotlib.pyplot as plt
import numpy as np


def show_images(X_train, y_train, classes):
    plt.figure(figsize=(20,5))
    for i in range(20):
        plt.subplot(2,10,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        try:
            plt.imshow(X_train[i], cmap=plt.cm.binary)
        except:
            plt.imshow(np.reshape(X_train[i],(X_train.shape[1],X_train.shape[2])), cmap=plt.cm.binary)
        try:
            plt.xlabel(classes[y_train[i]])
        except:
            plt.xlabel(classes[np.argmax(y_train[i])])
    plt.show()


def random_erasing(img, p=0.5, s=(0.1,0.4), r=(0.3,3.3)):
    if np.random.rand() > p:
        return img

    mask_value = np.random.rand()
    h, w, _ = img.shape

    mask_area = np.random.uniform(s[0],s[1]) * h*w
    mask_aspect_ratio = np.random.uniform(r[0], r[1])

    mask_h = int(np.sqrt(mask_area / mask_aspect_ratio / 2))
    mask_w = int(mask_h * mask_aspect_ratio)

    x_center = np.random.randint(0,h)
    y_center = np.random.randint(0,w)

    img[max(0,x_center-mask_h):min(h,x_center+mask_h), max(0,y_center-mask_w):min(w,y_center+mask_w), :].fill(mask_value)
    return img


def change_brightness(img, p=0.5, shift_range=(0.75,1.25)):
    if np.random.rand() > p:
        return img
    else:
        return img ** np.random.uniform(shift_range[0], shift_range[1])


def flow_datagen(datagen, X_train, y_train, batch_size=32, shuffle=True, functions=[]):
    for X, y in datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=shuffle):
        for function in functions:
            X = [function(img) for img in X]
        yield X, y


def show_augmented_images(datagen, X_train, y_train, classes):
    print('After Augmentation')
    for X, y in flow_datagen(datagen, X_train, y_train, batch_size=20, shuffle=False, functions=[random_erasing]):
        show_images(np.array(X), y, classes)
        break


def plot_history(history):
    # plot accuracy
    plt.plot(history.history['accuracy'], marker='.', label='train_acc')
    plt.plot(history.history['val_accuracy'], marker='.', label='val_acc')
    plt.title('model accuracy')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc='best')
    plt.show()

    # plot loss
    plt.plot(history.history['loss'], marker='.', label='train_loss')
    plt.plot(history.history['val_loss'], marker='.', label='val_loss')
    plt.title('model loss')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='best')
    plt.show()


def test_time_augmentation(model, datagen, X_test, epochs=20, len_classes=10):
    test_size = X_test.shape[0]
    pred = np.zeros(shape=(test_size,len_classes), dtype=float)
    for _ in range(epochs):
        pred += model.predict(datagen.flow(X_test, shuffle=False))
    return pred / epochs


def calculate_acc(pred, ans):
    cnt = 0
    for i in range(len(pred)):
        if np.argmax(pred[i]) == np.argmax(ans[i]):
            cnt += 1
    return cnt / len(pred)


def show_mislead_images(X_test, y_test, pred, classes):
    cnt = 0
    plt.figure(figsize=(25,5))
    for i in range(y_test.shape[0]):
        if cnt == 20:
            break
        if y_test[i][np.argmax(pred[i])] == 0:
            cnt += 1
            plt.subplot(2,10,cnt)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            try:
                plt.imshow(X_test[i], cmap=plt.cm.binary)
            except:
                plt.imshow(np.reshape(X_test[i], (X_test.shape[1],X_test.shape[2])), cmap=plt.cm.binary)
            plt.xlabel('ans:' + classes[np.argmax(y_test[i])] + ' pred:' + classes[np.argmax(pred[i])])
    plt.show()

