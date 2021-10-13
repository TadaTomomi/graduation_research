import cv2

def shrink(input_img, output_filename, width=96, height=96):    
    #画像の読み込み
    # img = cv2.imread(input_filename) 
    #指定のサイズに画像をリサイズし、書き出す
    img2 = cv2.resize(input_img , (width, height))
    cv2.imwrite(output_filename, img2)

img = cv2.imread('/home/student/datasets/CTJPEG/00022299.28y0m.f/Head_12_3D_Api_(Adult)__1/Dental_10_H60s_3/IM-0002-0180.jpg')

threshold = 90

ret2, img_th = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

shrink(img_th, 'output.jpg', 256, 256)