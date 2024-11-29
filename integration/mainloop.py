from capturecropgrey import capture_image_from_webcam, isolate_hand
from infer import make_inference

lst = ['A', 'B', 'Blank', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
while(True):
    result = isolate_hand(capture_image_from_webcam())
    if (result):
        predicted_label = make_inference('isolated_hand.png')
        print(predicted_label)
