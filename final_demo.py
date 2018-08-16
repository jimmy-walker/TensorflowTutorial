import sys
import argparse
import cv2
from keras.models import load_model
import os
import dlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import age_inception_resnet_v1
import gender_inception_resnet_v1

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

def eval(emotion_aligned_image, ga_aligned_image, emotion_model_path, age_model_path, gender_model_path):

	#emotion
	emotion_classifier = load_model(emotion_model_path, compile=False)
	emotion_all = emotion_classifier.predict(gray_face).flatten().tolist()
	emotion_dict = dict(zip(emotion_keys, emotion_all))
	emotion_result = sorted(emotion_dict.items(), key=lambda d: d[1], reverse=True)

	#age
    with tf.Graph().as_default():
        sess = tf.Session()
        images_pl = tf.placeholder(tf.float32, shape=[None, 160, 160, 3], name='input_image')
        images = tf.map_fn(lambda frame: tf.reverse_v2(frame, [-1]), images_pl) #BGR TO RGB
        images_norm = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), images)
        train_mode = tf.placeholder(tf.bool)
        age_logits = age_inception_resnet_v1.inference(images_norm, keep_probability=0.8,
                                                                     phase_train=train_mode,
                                                                     weight_decay=1e-5)
        age_ = tf.cast(tf.constant([i for i in range(0, 101)]), tf.float32)
        age = tf.reduce_sum(tf.multiply(tf.nn.softmax(age_logits), age_), axis=1)
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(age_model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("restore and continue training!")
        else:
            pass
        age_result = sess.run([age], feed_dict={images_pl: aligned_images, train_mode: False})

    #gender
    with tf.Graph().as_default():
        sess = tf.Session()
        images_pl = tf.placeholder(tf.float32, shape=[None, 160, 160, 3], name='input_image')
        images = tf.map_fn(lambda frame: tf.reverse_v2(frame, [-1]), images_pl) #BGR TO RGB
        images_norm = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), images)
        train_mode = tf.placeholder(tf.bool)
        gender_logits = gender_inception_resnet_v1.inference(images_norm, keep_probability=0.8,
                                                                     phase_train=train_mode,
                                                                     weight_decay=1e-5)
        gender = tf.argmax(tf.nn.softmax(gender_logits), 1)
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(gender_model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("restore and continue training!")
        else:
            pass
        gender_result = sess.run([gender], feed_dict={images_pl: aligned_images, train_mode: False})

    return emotion_result, age_result, gender_result

def load_image(image_path, shape_predictor, detector_para, face_width, is_gray):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor)
    fa = FaceAligner(predictor, desiredLeftEye=(detector_para, detector_para), desiredFaceWidth=face_width)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 2)
    rect_nums = len(rects)
    XY, aligned_images = [], []
    if rect_nums == 0:
        return aligned_images, image, rect_nums, XY
    else:
    	if is_gray:
		    for i in range(rect_nums):
		        aligned_image = fa.align(image, gray, rects[i])     
		        aligned_gray = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2GRAY) #if para is 64, it will return the shape (64, 64)
		        gray_face = preprocess_input(aligned_gray, True)
				gray_face = np.expand_dims(gray_face, 0) #it will return the shape (1, 64, 64)
				gray_face = np.expand_dims(gray_face, -1) #it will return the shape (1, 64, 64, 1)
				aligned_images.append(gray_face)
	            (x, y, w, h) = rect_to_bb(rects[i])
	            image = cv2.rectangle(image, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=2) #draw a rectangle in the original image
	            XY.append((x, y))	
	        return np.array(aligned_images), image, rect_nums, XY			
        else:
	        for i in range(rect_nums):
	            aligned_image = fa.align(image, gray, rects[i]) #if para is 160, it will return the shape (160, 160, 3)
	            aligned_images.append(aligned_image)
	            (x, y, w, h) = rect_to_bb(rects[i])
	            image = cv2.rectangle(image, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=2) #draw a rectangle in the original image
	            XY.append((x, y))
	        return np.array(aligned_images), image, rect_nums, XY

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", "--I", required=True, type=str, help="Image Path")
    parser.add_argument("--emotion_model_path", "--M", default="./emotion_models/fer2013_mini_XCEPTION.112-0.65.hdf5", type=str, help="Emotion Model Path")
    parser.add_argument("--gender_model_path", "--M", default="./gender_models", type=str, help="Gender Model Path")
    parser.add_argument("--age_model_path", "--M", default="./age_models", type=str, help="Age Model Path")
    parser.add_argument("--shape_detector", "--S", default="./models/shape_predictor_68_face_landmarks.dat", type=str,
                        help="Shape Detector Path")
    parser.add_argument("--cuda", default=False, action="store_true",
                        help="Set this flag will use cuda when testing.")
    parser.add_argument("--font_scale", type=int, default=1, help="Control font size of text on picture.")
    parser.add_argument("--thickness", type=int, default=1, help="Control thickness of texton picture.")
    parser.add_argument("--emotion_para", type=float, default=0.3, help="set the desiredLeftEye within emotion classification.")
    parser.add_argument("--ga_para", type=float, default=0.4, help="set the desiredLeftEye within gender and age classification.")
    parser.add_argument("--emotion_width", type=int, default=64, help="set the desiredFaceWidth within emotion classification.")
    parser.add_argument("--ga_width", type=int, default=160, help="set the desiredFaceWidth within gender and age classification.")

    args = parser.parse_args()
    if not args.cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
    	os.environ['CUDA_VISIBLE_DEVICES'] = '6'


    emotion_aligned_image, emotion_image, emotion_rect_nums, emotion_XY = load_image(args.image_path, args.shape_detector, args.emotion_para, args.emotion_width, True)
    ga_aligned_image, image, rect_nums, XY = load_image(args.image_path, args.shape_detector, args.ga_para, args.ga_width, False)

	if (not emotion_aligned_image) or (not ga_aligned_image):
		print("Face could not be found, Please confirm the frontal face.")
	else:
		emotions_result, age, gender = eval(emotion_aligned_image, ga_aligned_image, args.emotion_model_path, args.age_model_path, args.age_model_path)
	    print("age:", age)
	    print("genders:", gender)
	    print("emotions_all", emotions_result)
	    print("emotion_max", emotions_result[0])





# import sys

# import cv2
# from keras.models import load_model
# import numpy as np
# import os
# import dlib
# import numpy as np
# from imutils.face_utils import FaceAligner
# from imutils.face_utils import rect_to_bb
# import age_inception_resnet_v1
# import gender_inception_resnet_v1
# os.environ["CUDA_VISIBLE_DEVICES"] = ' '

# def preprocess_input(x, v2=True):
#     x = x.astype('float32')
#     x = x / 255.0
#     if v2:
#         x = x - 0.5
#         x = x * 2.0
#     return x

# # parameters for loading data and images
# # image_path = sys.argv[1]
# image_path = 'picture/test.jpg'

# shape_predictor='shape_predictor_68_face_landmarks.dat'

# emotion_para = (0.3, 0.3)
# emotion_width = 64
# emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.112-0.65.hdf5'
# emotion_labels = {0:'angry',1:'disgust',2:'fear',3:'happy', 4:'sad',5:'surprise',6:'neutral'}
# emotion_keys = emotion_labels.values()
# emotion_classifier = load_model(emotion_model_path, compile=False) 

# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(shape_predictor)
# fa_emotion = FaceAligner(predictor, desiredLeftEye=emotion_para, desiredFaceWidth=emotion_width) #change the hype-parameter
# image = cv2.imread(image_path, cv2.IMREAD_COLOR)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# rects = detector(gray, 2)
# rect_nums = len(rects)
# XY, aligned_images = [], []
# if rect_nums == 0:
#     print ("face not found")
# else:
#     for i in range(rect_nums):
#         aligned_image = fa_emotion.align(image, gray, rects[i])
#         aligned_images.append(aligned_image)        
#         aligned_gray = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2GRAY)
#         gray_face = preprocess_input(aligned_gray, True)
# 		gray_face = np.expand_dims(gray_face, 0)
# 		gray_face = np.expand_dims(gray_face, -1)
# 		emotion_all = emotion_classifier.predict(gray_face).flatten().tolist()
# 		emotion_dict = dict(zip(emotion_keys, emotion_all))
#         emotion_sorted = sorted(emotion_dict.items(), key=lambda d: d[1], reverse=True)
# 		print ("All emotions:")
# 		print (emotion_sorted)
# 		print ("Main emotion")
# 		print (emotion_sorted[0])

#         (x, y, w, h) = rect_to_bb(rects[i])
#         #image = cv2.rectangle(image, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=2)
#         XY.append((x, y))