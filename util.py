import string
import easyocr
import pytesseract

# Initialize the OCR reader
reader = easyocr.Reader(['en'])

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}



def format_license(licence):
    #TNO2BL8713
    print("inside format licence")
    char_index = [0,1,4,5]
    num_index = [2,3,6,7,8,9]
    if len(licence)!=10:
        return licence
    for i in range(10):
        if i in char_index:
            licence[i] = dict_int_to_char.get(licence[i], licence[i])
        else:
            licence[i] = dict_char_to_int.get(licence[i], licence[i])
    return licence

def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """

    # detections = reader.readtext(license_plate_crop)
    string = pytesseract.image_to_string(license_plate_crop)
    string_list=[]
    string_list.append(format_license(string))
    score = 0
    # for detection in detections:
    #     bbox, text, score = detection

    #     string = text.upper().replace(' ', '')
    #     string_list.append(format_license(string))
    # string = pytesseract.image_to_string(license_plate_crop)
    return "_".join(string_list), score

    # return None, None


def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1
