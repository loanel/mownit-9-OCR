import sys
import cv2
import numpy as np
import os


def threshold_and_deskew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # grab the (x, y) coordinates of all pixel values that
    # are greater than zero, then use these coordinates to
    # compute a rotated bounding box that contains all
    # coordinates
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    # otherwise, just take the inverse of the angle to make
    # it positive
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # rotate the image to deskew it
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    #again threshold from grayscale
    image = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return image




def get_line_with_white_signs(text, output):
    text = sorted(text, key=lambda element: element[0])
    previous_vertical = sys.maxsize
    for letter in text:
        #the size of the font used is 44, above 50 is enough to correctly measure spacing
        if letter[0] - previous_vertical > 50:
            output += " "
        previous_vertical = letter[0]
        output += letter[1]
    return output


def restore_text_with_newline(array):
    text = ""
    previous_horizontal = sys.maxsize
    word = []
    for i in range(len(array)):
        for b in range(len(array[i])):
            if array[i][b] != 0:
                #size is enough with 44 font, this allows to perfectly newline
                if i - previous_horizontal > 20:
                    text = get_line_with_white_signs(word, text)
                    text += "\n"
                    word = []
                previous_horizontal = i
                word += [[b, array[i][b][1]]]
    text = get_line_with_white_signs(word, text)
    return text


def clear_other_occurences(array):
    for p in range(len(array)):
        for q in range(len(array[p])):
            if array[p][q] != 0:
                #font size 44, 16:44 is a good ratio to check
                for k in range(p - 8, p + 8):
                    for l in range(q - 22, q + 22):
                        if (k != p or l != q) and 0 < k < len(array) and len(array[p]) > l > 0 != array[k][l]:
                            if array[k][l][0] > array[p][q][0]:
                                array[p][q][0] = array[k][l][0]
                                array[p][q][1] = array[k][l][1]
                            array[k][l] = 0
    return array


def find_letters_in_image(image, result, letter_folders, correlation_factor):
    image = 255 - image
    # doing corelations for all letters
    for directory in letter_folders:
        for root, dirs, files in os.walk(directory):
            for file in files:
                letter_file = directory + "/" + file
                letter = cv2.imread(letter_file)
                letter = cv2.cvtColor(letter, cv2.COLOR_BGR2GRAY)
                # algorithm from lab9
                letter = 255 - letter

                f_i = np.fft.fft2(image)
                f_p = np.fft.fft2(np.rot90(letter, 2), f_i.shape)
                helper = np.multiply(f_i, f_p)
                corr = np.fft.ifft2(helper)
                corr = np.abs(corr)
                corr = corr.astype(float)

                # we have to calculate the maximum correlation for each letter, in order to properly check parse the correlation matrix
                f_i = np.fft.fft2(letter)
                f_p = np.fft.fft2(np.rot90(letter, 2), f_i.shape)
                helper = np.multiply(f_i, f_p)
                maximum_letter_correlation = np.fft.ifft2(helper)
                maximum_letter_correlation = np.abs(maximum_letter_correlation)
                maximum_letter_correlation = maximum_letter_correlation.astype(float)
                maximum_letter_correlation = np.amax(maximum_letter_correlation)

                corr[corr < correlation_factor * maximum_letter_correlation] = 0

                corr_row, corr_column = corr.shape
                for row in range(corr_row):
                    for column in range(corr_column):
                        if corr[row, column] != 0:
                            if result[row][column] == 0 or result[row][column][0] < corr[row, column]:
                                result[row][column] = [corr[row, column], file[0]]

                result = clear_other_occurences(result)
    return result


def statistics(result, letters):
    letter_count = len(letters) * [0]
    for i in range(len(result)):
        for j in range(len(letters)):
            if result[i] == letters[j]:
                letter_count[j] += 1
    return letter_count


def main():
    english_signs = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                     'U', 'V', 'W', 'X', 'Y', 'Z',
                     'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                     'u', 'v', 'w', 'x', 'y', 'z',
                     '.', ',', '!', '?']


    ## test1.jpg - consolas, test2.jpg - calibri, test3.jpg - cambria, test_longer.jpg - consolas
    print("Enter name of the image you want to retrieve text from")
    filename = sys.stdin.readline().strip("\n")
    print("Enter name of the font used in the image")
    font = sys.stdin.readline().strip("\n")
    ## pick font according to image choice, but can also try seeing what will happen for wrong font choice
    if font == "consolas":
        letter_folders = ["consolas", "consolas2"]
    elif font == "calibri":
        letter_folders = ["calibri", "calibri2"]
    elif font == "cambira":
        letter_folders = ["cambira", "cambira2"]


    correlation_factor = 0.91

    # rotating and denoising image
    image = cv2.imread(filename)
    image = threshold_and_deskew(image)

    image_height, image_width = image.shape

    #setting up results table, it has to account for the letters on the edge
    result = image_height * [0]
    for x in range(len(result)):
        result[x] = image_width * [0]

    result = find_letters_in_image(image, result, letter_folders, correlation_factor)
    result = restore_text_with_newline(result)
    print(result)
    occurences = statistics(result, english_signs)
    for i in range(len(english_signs)):
        print(english_signs[i] + " occured " + str(occurences[i]) + " times\n")

main()
