import cv2
import numpy as np
import os
import csv

def rgb_to_hsl(image):
    height, width, _ = image.shape
    hsl = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            r, g, b = image[y, x] / 255.0 
            vmax = max(r, g, b)
            vmin = min(r, g, b)
            l = (vmax + vmin) / 2
            c = vmax - vmin
            if c == 0:
                s = 0
            else:
                s = c / (1 - abs(2 * l - 1))
            if c == 0:
                h = 0
            elif vmax == r:
                h = (60 * ((g - b) / c)) % 360
            elif vmax == g:
                h = (60 * ((b - r) / c) + 120) % 360
            else:
                h = (60 * ((r - g) / c) + 240) % 360
            
            hsl[y, x] = [int(h / 2), int(s * 255), int(l * 255)]  # Convert for OpenCV
    return hsl

def convert_to_grayscale(image):
    height, width, _ = image.shape
    gray = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            r, g, b = image[y, x]
            gray[y, x] = int(0.299 * r + 0.587 * g + 0.114 * b)
    return gray

def convolve(image, kernel):
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    padded_img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    output = np.zeros((img_h, img_w), dtype=np.float32)
    for i in range(img_h):
        for j in range(img_w):
            region = padded_img[i:i + k_h, j:j + k_w]
            output[i, j] = np.sum(region * kernel)
    return output

def compute_gradients(image):
    Gx_kernel = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])

    Gy_kernel = np.array([[-1, -2, -1],
                           [0,  0,  0],
                           [1,  2,  1]])

    Gx = convolve(image, Gx_kernel)
    Gy = convolve(image, Gy_kernel)
    Gmag = np.sqrt(Gx.astype(np.float32)**2 + Gy.astype(np.float32)**2)
    Gmag = np.clip(Gmag, 0, 255).astype(np.uint8)
    return Gx, Gy, Gmag

def non_maximal_suppression(Gmag, Gx, Gy):
    H, W = Gmag.shape
    suppressed = np.zeros((H, W), dtype=np.uint8)
    theta = np.arctan2(Gy, Gx) * (180.0 / np.pi)  
    theta = theta % 180 
    for y in range(1, H - 1):
        for x in range(1, W - 1):
            q, r = 255, 255 
            if (0 <= theta[y, x] < 22.5) or (157.5 <= theta[y, x] <= 180):  # Horizontal (0°)
                q = Gmag[y, x + 1]  # Right neighbor
                r = Gmag[y, x - 1]  # Left neighbor

            elif 22.5 <= theta[y, x] < 67.5:  # Diagonal (45°)
                q = Gmag[y + 1, x + 1]  # Bottom-right
                r = Gmag[y - 1, x - 1]  # Top-left

            elif 67.5 <= theta[y, x] < 112.5:  # Vertical (90°)
                q = Gmag[y + 1, x]  # Bottom
                r = Gmag[y - 1, x]  # Top

            elif 112.5 <= theta[y, x] < 157.5:  # Diagonal (135°)
                q = Gmag[y - 1, x + 1]  # Top-right
                r = Gmag[y + 1, x - 1]  # Bottom-left

            if Gmag[y, x] >= q and Gmag[y, x] >= r:
                suppressed[y, x] = Gmag[y, x]
            else:
                suppressed[y, x] = 0

    return suppressed
def double_threshold(gmag, low_ratio, high_ratio):
    strong_edge = 255
    weak_edge = 75
    non_edge = 0
    high_thresh = high_ratio * gmag.max()
    low_thresh = low_ratio*high_thresh
    # Initialize edge map
    edges = np.zeros_like(gmag, dtype=np.uint8)

    # Strong edges
    edges[gmag >= high_thresh] = strong_edge

    # Weak edges
    edges[(gmag >= low_thresh) & (gmag < high_thresh)] = weak_edge

    return edges

def hysteresis(edges):
    strong_edge = 255
    weak_edge = 75
    non_edge = 0
    rows, cols = edges.shape
    final_edges = edges.copy()
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if final_edges[i, j] == weak_edge:
                if (strong_edge in final_edges[i-1:i+2, j-1:j+2]):
                    final_edges[i, j] = strong_edge 
                else:
                    final_edges[i, j] = non_edge 

    return final_edges

def merge_lines(lines, angle_threshold=25, distance_threshold=40):
    if lines is None:
        return []
    
    merged_lines = []
    used = set()

    for i, line1 in enumerate(lines):
        if i in used:
            continue
        x1, y1, x2, y2 = line1
        slope1 = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi  # Convert to degrees
        
        merged = [line1]
        for j, line2 in enumerate(lines):
            if j in used or i == j:
                continue
            x3, y3, x4, y4 = line2
            slope2 = np.arctan2(y4 - y3, x4 - x3) * 180 / np.pi  

            if abs(slope1 - slope2) < angle_threshold:
                # Check if lines are close
                dist = np.sqrt((x3 - x1)**2 + (y3 - y1)**2)
                if dist < distance_threshold:
                    merged.append(line2)
                    used.add(j)

        # Find average line
        merged_x1 = int(np.mean([l[0] for l in merged]))
        merged_y1 = int(np.mean([l[1] for l in merged]))
        merged_x2 = int(np.mean([l[2] for l in merged]))
        merged_y2 = int(np.mean([l[3] for l in merged]))

        merged_lines.append([[merged_x1, merged_y1, merged_x2, merged_y2]])
        used.add(i)

    return np.array(merged_lines)

def extend_lines(lines, img_shape,min_length=300):
    height, width = img_shape[:2]
    extended_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        print(length)
        if length > min_length:
            # Compute slope and intercept
            if x2 - x1 != 0:  # Avoid division by zero
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1

                # Extend to the left and right edges of the image
                x_start = 0
                y_start = int(slope * x_start + intercept)
                x_end = width
                y_end = int(slope * x_end + intercept)

                extended_lines.append([[x_start, y_start, x_end, y_end]])

    return np.array(extended_lines)

def filter_short_lines(lines, min_length=10):
    filtered = []
    for line in lines:
        x1, y1, x2, y2 = line
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if length > min_length:
            filtered.append(line)
    return np.array(filtered)

def detect_lines_from_edges(grad_img: np.ndarray, original,threshold, min_points_on_line: int = 50) -> np.ndarray:
    height, width = grad_img.shape
    step=5
    edge_pairs = [
        ([(0, x) for x in range(0,width,step)], [(height-1, x) for x in range(0,width,step)]),  # Top → Bottom
        ([(y, 0) for y in range(0,height,step)], [(y, width-1) for y in range(0,height,step)]),  # Left → Right
        ([(0, x) for x in range(0,width,step)], [(y, 0) for y in range(0,height,step)]),        # Top → Left
        ([(0, x) for x in range(0,width,step)], [(y, width-1) for y in range(0,height,step)]),  # Top → Right
        ([(height-1, x) for x in range(0,width,step)], [(y, 0) for y in range(0,height,step)]), # Bottom → Left
        ([(height-1, x) for x in range(0,width,step)], [(y, width-1) for y in range(0,height,step)])  # Bottom → Right
    ]
    final_lines=[]
    for start_points, end_points in edge_pairs:
        for y1, x1 in start_points:
                for y2, x2 in end_points:
                    line_points = []
                    num_steps = max(abs(y2 - y1), abs(x2 - x1))
                    for i in range(1, num_steps):
                        y_interp = int(y1 + (y2 - y1) * (i / num_steps))
                        x_interp = int(x1 + (x2 - x1) * (i / num_steps))
                        if grad_img[y_interp, x_interp] > threshold:
                            line_points.append((x_interp, y_interp))
                    if len(line_points) > min_points_on_line:
                        pt1, pt2 = line_points[0], line_points[-1]
                        final_lines.append([pt1[0], pt1[1], pt2[0], pt2[1]])

    return final_lines

def detect_lanes(lines, original):
    lane_image = original.copy()
    if lines is not None:
        lines = filter_short_lines(lines)
        lines = merge_lines(lines)
        lines_1 = extend_lines(lines, original.shape)
        for i in lines:
            x1, y1, x2, y2 = i[0]
            cv2.line(lane_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
        for line in lines_1:
            print(line)
            x1, y1, x2, y2 = line[0]
            cv2.line(lane_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
        lines_reshaped = lines.reshape(-1, 4)
        lines_1_reshaped = lines_1.reshape(-1, 4) 

        all_lines = np.concatenate((lines_reshaped, lines_1_reshaped), axis=0)
    return lane_image,all_lines

def process_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(input_dir, filename)
            base_name = os.path.splitext(filename)[0]
            image = cv2.imread(img_path)
            if image is None:
                print(f"Could not read {filename}, skipping...")
                continue
            kernel = (1/16) * np.array([[1, 2, 1],
                                        [2, 4, 2],
                                        [1, 2, 1]])
            height, width, channels = image.shape

            image=cv2.resize(image, (width//2, height//2))
            
            original_image=image.copy()
            image=rgb_to_hsl(image)
            gray_image = convert_to_grayscale(image)
            blur=convolve(gray_image,kernel)
            blur=convolve(blur,kernel)
            blur=convolve(blur,kernel)
            Gx, Gy, Gmag = compute_gradients(blur)
            supressed=non_maximal_suppression(Gmag,Gx,Gy)
            low_thresh = 0.02
            high_thresh = 0.4 
            dt = double_threshold(supressed, low_thresh, high_thresh)
            final_edge_map = hysteresis(dt)
            Hough=detect_lines_from_edges(final_edge_map,original_image, threshold=50)    
            merged,lines = detect_lanes(Hough, original_image)

            cv2.imwrite(os.path.join(output_dir, f"{base_name}.png"), merged)
            print(f"Processed: {filename}")
           
def line_intersection(p1, p2, p3, p4):
    """Finds intersection of two line segments (p1->p2) and (p3->p4)."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    # Compute determinants
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:  # Parallel lines
        return None
    px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / denom

    # Check if intersection is within the segment bounds
    if (min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2) and
        min(x3, x4) <= px <= max(x3, x4) and min(y3, y4) <= py <= max(y3, y4)):
        return (px, py)
    
    return None 

def find_intersections(endpoints, image_shape):
    intersections = []
    height, width = image_shape[:2]

    for i in range(len(endpoints)):
        p1, p2 = (endpoints[i][0:2]),(endpoints[i][2:4])
        for j in range(i + 1, len(endpoints)):
            p3, p4 = (endpoints[j][0:2]),(endpoints[j][2:4])
            intersection = line_intersection(p1, p2, p3, p4)
            if intersection:
                x, y = intersection
                if 0 <= x < width and 0 <= y < height:
                    intersections.append(intersection)
    
    return np.array(intersections)

def Centroid(points):
    if len(points) == 0:
        return None
    return np.mean(points, axis=0)

def sum_of_distances(centroid, points):
    if centroid is None or len(points) == 0:
        return None
    distances = np.linalg.norm(points - centroid, axis=1)
    return np.sum(distances)

def process_image_lines(endpoints, image_shape, image_name, csv_filename="output.csv"):
    intersections = find_intersections(endpoints, image_shape)
    if len(intersections) == 0:
        print("No intersections")
        return
    centroid = Centroid(intersections)
    total_distance = sum_of_distances(centroid, intersections)
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([image_name, total_distance])
    print(f"Processed {image_name}: Sum of distances = {total_distance}")
def part2():
    input_dir = input_directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(input_dir, filename)
            base_name = os.path.splitext(filename)[0]
            image = cv2.imread(img_path)
            if image is None:
                print(f"Could not read {filename}, skipping...")
                continue
            kernel = (1/16) * np.array([[1, 2, 1],
                                        [2, 4, 2],
                                        [1, 2, 1]])
            height, width, channels = image.shape

            image=cv2.resize(image, (width//2, height//2))  
            original_image=image.copy()
            image=rgb_to_hsl(image)
            gray_image = convert_to_grayscale(image)
            blur=convolve(gray_image,kernel)
            blur=convolve(blur,kernel)
            blur=convolve(blur,kernel)
            Gx, Gy, Gmag = compute_gradients(blur)
            supressed=non_maximal_suppression(Gmag,Gx,Gy)
            low_thresh = 0.02
            high_thresh = 0.3  
            dt = double_threshold(supressed, low_thresh, high_thresh)
            final_edge_map = hysteresis(dt)
            Hough=detect_lines_from_edges(final_edge_map,original_image, threshold=50)   
            merged,lines = detect_lanes(Hough, original_image)
            print(f"Processed: {filename}")
            process_image_lines(lines, original_image.shape, filename, csv_filename=output_directory)
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python3 script.py part <input_directory> <output_directory>")
        sys.exit(1)
    input_directory = sys.argv[2]
    output_directory = sys.argv[3]
    if sys.argv[1] == "2":
        part2()
    elif sys.argv[1] == "1":
        process_images(input_directory, output_directory)
    else:
        print("not correct format")
        