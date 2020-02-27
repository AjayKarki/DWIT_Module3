import cv2
import os

digits = [0,1,2,3,4,5,6,7,8,9]

for digit in digits:
    os.makedirs('dataset_preprocessed_train_test\\'+ str(digit) +'\\', exist_ok=True)

if __name__ == "__main__":
    for digit in digits:
        for filename in os.listdir('dataset_train\\'+ str(digit) +'\\'):
            img = cv2.imread('dataset_train\\'+ str(digit) +'\\'+filename)
            #print(img.shape)
            img = 255 - img # invert gareko
            # cv2.imshow("image",img) 
            # cv2.waitKey()
            # exit()
            # print(img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            gray_scale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #print("gray",list(gray_scale_img))
            #print("gray",list(gray_scale_img.shape))
            # cv2.imshow("image", gray_scale_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            ret, mask = cv2.threshold(gray_scale_img, 180, 255, cv2.THRESH_BINARY) # image lai threshold gareko ie dark dareko
            #print("mask",list(mask))
            # cv2.imshow("image", mask)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            final_img = cv2.bitwise_and(gray_scale_img, gray_scale_img, mask=mask) # multiply gareko
            #print("final" ,list(final_img))
            #cv2.imshow("image", final_img)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            ret, new_img = cv2.threshold(final_img, 180, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (1,
                                                                 1))  # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
            dilated_img = cv2.dilate(new_img, kernel, iterations=1)  # dilate , more the iteration more the dilation # dilate gareko vaneko pen le lekhera pani ma halda jasto fulxa testai ho
            #cv2.imshow("image", final_img)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            contours, hierarchy = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours_img = []
            for contour in contours:
                [x, y, w, h] = cv2.boundingRect(contour)
                contours_img.append([x, y, w, h])

            contours_img.sort()

            i = 0
            digit_segment_img = []

            for contour in contours_img:
                # get rectangle bounding contour
                [x, y, w, h] = contour
                # eliminating false positive from our contour
                if w < 20 and h < 20:
                    continue

                # drawing rectangle around contour
                rec_img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)

                # crop each contour and save individually
                cropped_img = final_img[y:y + h, x:x + w]
            final_img = cv2.resize(cropped_img, (32, 32))
            cv2.imwrite('dataset_preprocessed_train_test\\'+str(digit)+'\\'+filename, final_img)
#