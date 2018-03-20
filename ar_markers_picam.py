import cv2
import os
#import system
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
from numpy import matrix, array
from numpy import mean, binary_repr, zeros
from numpy.random import randint
from scipy.ndimage import zoom
from numpy import array, rot90
#========================================= Marker =========================================
MARKER_SIZE = 7
flag = 0
start_flag=0
class HammingMarker(object):
    def __init__(self, id, contours=None):
        self.id = id
        self.contours = contours

    def __repr__(self):
        return '<Marker id={} center={}>'.format(self.id, self.center)

    @property
    def center(self):
        if self.contours is None:
            return None
        center_array = mean(self.contours, axis=0).flatten()
        return (int(center_array[0]), int(center_array[1]))

    def generate_image(self):
        img = zeros((MARKER_SIZE, MARKER_SIZE))
        img[1, 1] = 255  # set the orientation marker
        for index, val in enumerate(self.hamming_code):
            coords = HAMMINGCODE_MARKER_POSITIONS[index]
            if val == '1':
                val = 255
            img[coords[0], coords[1]] = int(val)
        return zoom(img, zoom=50, order=0)

    def draw_contour(self, img, color=(255, 255, 0), linewidth=5):
        cv2.drawContours(img, [self.contours], -1, color, linewidth)

    def highlite_marker(self, img, contour_color=(255, 255, 0), text_color=(255, 0, 0), linewidth=5):
        self.draw_contour(img, color=contour_color, linewidth=linewidth)
        cv2.putText(img, str(self.id), self.center, cv2.FONT_HERSHEY_SIMPLEX, 2, text_color)

    @classmethod
    def generate(cls):
        return HammingMarker(id=randint(4096))

    @property
    def id_as_binary(self):
        return binary_repr(self.id, width=12)

    @property
    def hamming_code(self):
        return encode(self.id_as_binary)

#=================================== Coding ================================================
GENERATOR_MATRIX = matrix([
    [1, 1, 0, 1],
    [1, 0, 1, 1],
    [1, 0, 0, 0],
    [0, 1, 1, 1],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
])

REGENERATOR_MATRIX = matrix([
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1],
])

PARITY_CHECK_MATRIX = matrix([
    [1, 0, 1, 0, 1, 0, 1],
    [0, 1, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1],
])

HAMMINGCODE_MARKER_POSITIONS = [
    [1, 2], [1, 3], [1, 4],
    [2, 1], [2, 2], [2, 3], [2, 4], [2, 5],
    [3, 1], [3, 2], [3, 3], [3, 4], [3, 5],
    [4, 1], [4, 2], [4, 3], [4, 4], [4, 5],
    [5, 2], [5, 3], [5, 4],
]

def encode(bits):
    encoded_code = ''
    if len(bits) % 4 != 0:
        raise ValueError('Only a multiple of 4 as bits are allowed.')
    while len(bits) >= 4:
        four_bits = bits[:4]
        bit_array = generate_bit_array(four_bits)
        hamming_code = matrix_array_multiply_and_format(GENERATOR_MATRIX, bit_array)
        encoded_code += ''.join(hamming_code)
        bits = bits[4:]
    return encoded_code


def decode(bits):
    decoded_code = ''
    if len(bits) % 7 != 0:
        raise ValueError('Only a multiple of 7 as bits are allowed.')
    for bit in bits:
        if int(bit) not in [0, 1]:
            raise ValueError('The provided bits contain other values that 0 or 1: %s' % bits)
    while len(bits) >= 7:
        seven_bits = bits[:7]
        uncorrected_bit_array = generate_bit_array(seven_bits)
        corrected_bit_array = parity_correct(uncorrected_bit_array)
        decoded_bits = matrix_array_multiply_and_format(REGENERATOR_MATRIX, corrected_bit_array)
        decoded_code += ''.join(decoded_bits)
        bits = bits[7:]
    return decoded_code


def parity_correct(bit_array):
    # Check the parity using the PARITY_CHECK_MATRIX
    checked_parity = matrix_array_multiply_and_format(PARITY_CHECK_MATRIX, bit_array)
    parity_bits_correct = True
    # every value as to be 0, so no error accoured:
    for bit in checked_parity:
        if int(bit) != 0:
            parity_bits_correct = False
    if not parity_bits_correct:
        error_bit = int(''.join(checked_parity), 2)
        for index, bit in enumerate(bit_array):
            if error_bit == index + 1:
                if bit == 0:
                    bit_array[index] = 1
                else:
                    bit_array[index] = 0
    return bit_array


def matrix_array_multiply_and_format(matrix, array):
    unformated = matrix.dot(array).tolist()[0]
    return [str(bit % 2) for bit in unformated]


def generate_bit_array(bits):
    return array([int(bit) for bit in bits])


def extract_hamming_code(mat):
    hamming_code = ''
    for pos in HAMMINGCODE_MARKER_POSITIONS:
        hamming_code += str(int(mat[pos[0], pos[1]]))
    return hamming_code

#===================================== Detect =============================================================
BORDER_COORDINATES = [
    [0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [1, 0], [1, 6], [2, 0], [2, 6], [3, 0],
    [3, 6], [4, 0], [4, 6], [5, 0], [5, 6], [6, 0], [6, 1], [6, 2], [6, 3], [6, 4], [6, 5], [6, 6],
]

ORIENTATION_MARKER_COORDINATES = [[1, 1], [1, 5], [5, 1], [5, 5]]

def validate_and_turn(marker):
    # first, lets make sure that the border contains only zeros
    for crd in BORDER_COORDINATES:
        if marker[crd[0], crd[1]] != 0.0:
            raise ValueError('Border contians not entirely black parts.')
    # search for the corner marker for orientation and make sure, there is only 1
    orientation_marker = None
    for crd in ORIENTATION_MARKER_COORDINATES:
        marker_found = False
        if marker[crd[0], crd[1]] == 1.0:
            marker_found = True
        if marker_found and orientation_marker:
            raise ValueError('More than 1 orientation_marker found.')
        elif marker_found:
            orientation_marker = crd
    if not orientation_marker:
        raise ValueError('No orientation marker found.')
    rotation = 0
    if orientation_marker == [1, 5]:
        rotation = 1
    elif orientation_marker == [5, 5]:
        rotation = 2
    elif orientation_marker == [5, 1]:
        rotation = 3
    marker = rot90(marker, k=rotation)
    return marker


def detect_markers(img):
    width, height, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 40, 150)#old 10,100
    #cv2.imshow('edges', edges)
    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]

    # We only keep the long enough contours
    min_contour_length = min(width, height) / 50
    contours = [contour for contour in contours if len(contour) > min_contour_length]
    warped_size = 49
    canonical_marker_coords = array(((0, 0),
                                     (warped_size - 1, 0),
                                     (warped_size - 1, warped_size - 1),
                                     (0, warped_size - 1)),
                                    dtype='float32')

    markers_list = []
    for contour in contours:
        approx_curve = cv2.approxPolyDP(contour, len(contour) * 0.01, True)
        if not (len(approx_curve) == 4 and cv2.isContourConvex(approx_curve)):
            continue

        sorted_curve = array(cv2.convexHull(approx_curve, clockwise=False),dtype='float32')
        persp_transf = cv2.getPerspectiveTransform(sorted_curve, canonical_marker_coords)
        warped_img = cv2.warpPerspective(img, persp_transf, (warped_size, warped_size))
        warped_gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)

        _, warped_bin = cv2.threshold(warped_gray, 127, 255, cv2.THRESH_BINARY)
        marker = warped_bin.reshape(
            [MARKER_SIZE, warped_size / MARKER_SIZE, MARKER_SIZE, warped_size / MARKER_SIZE]
        )
        marker = marker.mean(axis=3).mean(axis=1)
        marker[marker < 127] = 0
        marker[marker >= 127] = 1

        try:
            marker = validate_and_turn(marker)
            hamming_code = extract_hamming_code(marker)
            marker_id = int(decode(hamming_code), 2)
            markers_list.append(HammingMarker(id=marker_id, contours=approx_curve))
        except ValueError:
            continue
    return markers_list
#====================================================================================================================================
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
time.sleep(0.1)
cv2.namedWindow('TT_TT', cv2.WINDOW_NORMAL)
cv2.resizeWindow('TT_TT', 320,195)
bcimg = cv2.imread('pibackground.png',1)
sdimg = cv2.imread('sacandone.png',1)
os.system('mplayer -fs -vf mirror splash.mp4')
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array
    markers = detect_markers(frame.array)
    for marker in markers:
        marker.highlite_marker(frame.array, contour_color=(255, 255, 255))
        cv2.putText(frame.array,'Hello World!',(10,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
        print 'marker.center'
        print marker.center
        print 'marker.contours'
        print cv2.contourArea(marker.contours)
        print 'marker.id'
        print marker.id
        print flag
        if flag >= 1:
            flag = 0
        elif marker.id == 22 and flag==0:
            cv2.putText(sdimg,'Marker 22 ready to play..',(10,26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,120),1)
            sdimg1 = cv2.flip(sdimg,1)
            cv2.imshow('TT_TT', sdimg1)
            cv2.waitKey(1)
            os.system('aplay alert.wav')
            time.sleep(2)
            os.system('mplayer -fs -vf mirror 1.3gpp')
            flag = flag+1
        elif marker.id == 1725 and flag==0:
            cv2.putText(sdimg,'Marker 1725 ready to play..',(10,26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,120),1)
            sdimg1 = cv2.flip(sdimg,1)
            cv2.imshow('TT_TT', sdimg1)
            cv2.waitKey(1)
            os.system('aplay alert.wav')
            time.sleep(2)
            os.system('mplayer -fs -vf mirror 2.3gpp')
            flag = flag+1
        else: 
            flag = flag+1
        
        
        #print marker.contours[0][0][1]
        #print marker.contours[0][0][1]
            
    #cv2.imshow('TT_TT', frame.array)
    
    cv2.imshow('TT_TT', bcimg)
    key = cv2.waitKey(1) & 0xFF
    rawCapture.truncate(0)
    if key == ord("c"):
		break
    #frame_captured, frame = capture.read()

    # When everything done, release the capture
#capture.release()
#cv2.destroyAllWindows()
