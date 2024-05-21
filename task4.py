import cv2
import os
import numpy as np

def find_best_contours(gray):
    low_thresholds = [50, 100, 150, 200]
    high_thresholds = [100, 150, 200, 250]

    best_contours = []
    max_contour_area = 0

    for low_thresh in low_thresholds:
        for high_thresh in high_thresholds:
            edges = cv2.Canny(gray, low_thresh, high_thresh)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            filtered_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h

                min_area = 100
                max_area = 5000
                min_aspect_ratio = 0.5
                max_aspect_ratio = 2.0

                if min_area < area < max_area and min_aspect_ratio < aspect_ratio < max_aspect_ratio:
                    filtered_contours.append(contour)

                    if area > max_contour_area:
                        max_contour_area = area
                        best_contours = filtered_contours.copy()
    return best_contours

def calculate_dimensions(image_path, ref_length, ref_height, output_image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Couldn't load image at " + image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    best_contours = find_best_contours(gray)

    if len(best_contours) < 2:
        raise ValueError("Not able to find enough contours in this image")

    pothole_contour = sorted(best_contours, key=cv2.contourArea, reverse=True)[0]
    reference_contour = sorted(best_contours, key=cv2.contourArea, reverse=True)[1]

    pothole_rect = cv2.boundingRect(pothole_contour)
    reference_rect = cv2.boundingRect(reference_contour)

    ref_width_pixels = reference_rect[2]
    ref_height_pixels = reference_rect[3]

    scaling_factor_width = ref_length / ref_width_pixels
    scaling_factor_height = ref_height / ref_height_pixels

    pothole_width_pixels = pothole_rect[2]
    pothole_height_pixels = pothole_rect[3]

    actual_pothole_width = pothole_width_pixels * scaling_factor_width
    actual_pothole_height = pothole_height_pixels * scaling_factor_height

    cv2.drawContours(image, [pothole_contour], -1, (0, 255, 0), 2)
    cv2.drawContours(image, [reference_contour], -1, (255, 0, 0), 2)
    cv2.rectangle(image, (pothole_rect[0], pothole_rect[1]),
                  (pothole_rect[0] + pothole_rect[2], pothole_rect[1] + pothole_rect[3]),
                  (0, 255, 0), 2)
    cv2.rectangle(image, (reference_rect[0], reference_rect[1]),
                  (reference_rect[0] + reference_rect[2], reference_rect[1] + reference_rect[3]),
                  (255, 0, 0), 2)

    cv2.imwrite(output_image_path, image)
    print("Processed image saved as " + output_image_path)

    return [actual_pothole_width, actual_pothole_height, ref_height]

def process_images(folder_path, ref_length, ref_height, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    results = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            output_image_path = os.path.join(output_folder, "processed_" + filename)

            try:
                actual_dimensions = calculate_dimensions(image_path, ref_length, ref_height, output_image_path)
                results.append([filename, actual_dimensions])
            except ValueError as e:
                print(e)
                continue

    return results

# Example usage
folder_path = "D:\\work\\jal\\Desktop\\internship\\Task_4"
output_folder = "D:\\work\\jal\\Desktop\\internship\\Task_4\\Label_image"
ref_length = 1.0
ref_height = 1.0

results = process_images(folder_path, ref_length, ref_height, output_folder)
print("Pothole dimensions for each image:", results)
