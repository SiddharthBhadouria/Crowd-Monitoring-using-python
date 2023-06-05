import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns


def count_people(link, threshold=5, min_box_area=10000, max_box_area=50000, max_frames_to_skip=10, max_distance=100):
    num_people = 0
    times = []
    num_of_people = []
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    video = cv2.VideoCapture(link)

    prev_boxes = []
    skipped_frames = 0

    while True:
        ret, frame = video.read()

        if not ret:
            break

        frame = cv2.resize(frame, (640, 360))  # resize for faster processing

        # detect humans in the frame
        boxes, weights = hog.detectMultiScale(frame, winStride=(8,8), padding=(16, 16), scale=1.05)

        # filter out small and large boxes
        filtered_boxes = []
        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            box_area = w * h
            if min_box_area <= box_area <= max_box_area:
                filtered_boxes.append((x, y, w, h, weights[i]))

        # match boxes to previous boxes
        
        current_boxes = []
        for box in filtered_boxes:
            x, y, w, h, weight = box
            box_center = (x + w // 2, y + h // 2)

            matched_box = None
            min_distance = float('inf')
            for prev_box in prev_boxes:
                prev_center = prev_box[0]
                prev_num_frames_skipped = prev_box[1]
                
                # calculates the distance between two boxes

                distance = np.linalg.norm(np.array(box_center) - np.array(prev_center))
                if distance < min_distance and prev_num_frames_skipped < max_frames_to_skip:
                    matched_box = prev_box
                    min_distance = distance

            if matched_box is not None:
                current_boxes.append((box_center, matched_box[1] + 1))
                prev_boxes.remove(matched_box)
            else:
                current_boxes.append((box_center, 0))

        # add new boxes to the list of current boxes
        current_boxes += [(box[0], box[1] + 1) for box in prev_boxes]

        # draw bounding boxes around the detected humans
        for box in current_boxes:
            center, num_frames_skipped = box
            x, y = center[0] - max_distance // 2, center[1] - max_distance // 2
            w, h = max_distance, max_distance

            if num_frames_skipped <= max_frames_to_skip:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                num_people += 1

            num_of_people.append(num_people)
            curr_time = time.time()
            times.append(curr_time)

        # update the list of previous boxes
        prev_boxes = current_boxes

        # display the resulting frame
        cv2.imshow('frame', frame)

        # exit if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

    # release the video capture object and close the window
    video.release()
    cv2.destroyAllWindows()

    plt.plot(times, num_of_people)
    plt.xlabel("Time")
    plt.ylabel("Number of people")
    plt.show()

    sns.set(style = "whitegrid")
    sns.kdeplot(num_of_people, fill = True)
    plt.title("Density Plot")
    plt.xlabel("Values")
    plt.ylabel("Density")
    plt.show()
    return num_people   



num_people = count_people(r"C:\Users\asus.pc\Shopping People Commerce Mall Many Crowd Walking   Free Stock video footage   YouTube.mp4")
print(f'Total number of people in the video: {num_people}')

