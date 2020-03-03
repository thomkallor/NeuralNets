import numpy as np

from sklearn.metrics import accuracy_score

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def loadbatch(batchname):
    folder = 'cifar-10-batches-py'
    batch = unpickle(folder+"/"+batchname)
    return batch


def loadlabelnames():
    folder = 'cifar-10-batches-py'
    meta = unpickle(folder+"/"+'batches.meta')
    return meta[b'label_names']



import matplotlib.pyplot as plt

def visualise(data, index):
    # MM Jan 2019: Given a CIFAR data nparray and the index of an image, display the image.
    # Note that the images will be quite fuzzy looking, because they are low res (32x32).

    picture = data[index]
    # Initially, the data is a 1D array of 3072 pixels; reshape it to a 3D array of 3x32x32 pixels
    # Note: after reshaping like this, you could select one colour channel or average them.
    picture.shape = (3,32,32) 
    
    # Plot.imshow requires the RGB to be the third dimension, not the first, so need to rearrange
    picture = picture.transpose([1, 2, 0])
    plt.imshow(picture)
    plt.show()


batch1 = loadbatch('data_batch_1')
print("Number of items in the batch is", len(batch1))


print('All keys in the batch:', batch1.keys())



data = batch1[b'data']
labels = batch1[b'labels']
print ("size of data in this batch:", len(data), ", size of labels:", len(labels))
# print (type(data))
# print(data.shape)

names = loadlabelnames()
# labels = [4]
X = list()
Y = list()
y_labs = { 'cat': 0, 'bird': 1}

# data = np.load('t.npy')
# names = ["cat","d"]

for i , image in enumerate(data):
	#    visualise(data, i)
	# print(names[labels[i]])
	# np.save('t',image)
	_label = names[labels[i]].decode("utf-8")
	if  _label in ['cat' ,'bird']:
		# print("Image", i,": Class is ", names[labels[i]])
		# print(isage.shape)
		image.shape = (3,32,32)
		image = image.transpose([1, 2, 0])
		# print(image.shape)
		# plt.imshow(image)
		B, G, R    = image[:, :, 0], image[:, :, 1], image[:, :, 2] # For RGB image
		# [B , G, R] = np.dsplit(image,image.shape[-1])
		# print(R.shape)
		# plt.imshow(R , cmap='gray', vmin=0, vmax=255)
		grey_scaled = 0.2125 * R + 0.7154 * G + 0.0721 * B
		
		# print(grey_scaled.shape)
		grey_scaled = grey_scaled //255
		grey_scaled = list(grey_scaled.flatten())
		feature=[]
		# for i in range(8):
		# 	feature.append(grey_scaled[i*4:(i+1)*4,i*4:(i+1)*4])
		# print(feature)
		# print(Y.shape)
		# plt.imshow(Y,cmap='gray')
		# plt.show()
		# bsreak
		X.append(grey_scaled)
		Y.append(y_labs[_label])


X = np.array(X)
Y = np.array(Y)

print(X.shape)
# for image in X:
# 	break



from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(X , Y , test_size=0.1)

# print(y_train.shape)

y_train = y_train.reshape(-1,1)


# # print(len(X))

from  g_v2 import NeuralNet

n = NeuralNet(x_train , y_train , 1024 ,1024)

for _ in range(1000):
	n.forward()
	n.backprop()
n.inputs = x_test
n.forward()

preds = n.get_predictions()
print(accuracy_score(y_test, preds))



