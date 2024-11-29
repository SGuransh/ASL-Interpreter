import time
from capturecropgrey import capture_image_from_webcam, isolate_hand
from infer import make_inference

lst = ['A', 'B', 'Blank', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
       'V', 'W', 'X', 'Y', 'Z']
output = ''
false_counter = 0

while True:
    time.sleep(1)  # Sleep for 3 seconds before each loop
    result = isolate_hand(capture_image_from_webcam())

    if result:
        predicted_label = make_inference('isolated_hand.png')
        letter = lst[predicted_label]
        print(letter)
        output += letter
        print(output)
        false_counter = 0  # Reset false counter when a hand is successfully isolated
    else:
        false_counter += 1
        if false_counter == 3:
            output += ' '
            print(output)
            false_counter = 0
