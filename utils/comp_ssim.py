import cv2
from pathlib import Path
from skimage import metrics
import re
import matplotlib.pyplot as plt


def find_matching_test_file(test_dir, number):
    test_dir = Path(test_dir)
    pattern = rf'^{number}(_NOX)?\.jpg$'
    for file in test_dir.iterdir():
        if file.is_file() and re.search(pattern, file.name):
            return file
    return None


def calculate_ssim(image1_path, image2_path):
    image1 = cv2.imread(str(image1_path))
    # plt.imshow(image1)
    image2 = cv2.imread(str(image2_path))
    if image1 is None or image2 is None:
        return None
    
    # 이미지 크기 조정 검토
    if image1.shape[1] > image2.shape[1] or image1.shape[0] > image2.shape[0]:
        image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]), interpolation=cv2.INTER_AREA)

    ssim_score, _ = metrics.structural_similarity(image1, image2, multichannel=True, full=True, gaussian_weights=True)
    return round(ssim_score, 4)


def compare_images_ssim(output_path, test_dir):
    print(f"output_path : {output_path}")
    print(f"test_dir : {test_dir}")
    output_file = Path(output_path)
    output_name = output_file.name
    match = re.search(r'(\d+)_', output_name)
    if match:
        number = match.group(1)
        test_file = find_matching_test_file(test_dir, number)
        if test_file:
            ssim_score = calculate_ssim(output_file, test_file)
            print(f"SSIM score between {output_file.name} and {test_file.name}: {ssim_score}")
            return {'output_name': output_file.name, 'test_name': test_file.name, 'ssim_score': ssim_score}
        else:
            print("Matching test file not found.")
    else:
        print("No matching number found in the file name.")

    return {'output_name': output_name, 'test_name': None, 'ssim_score': None}
