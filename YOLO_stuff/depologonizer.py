import os

def polygon_to_bbox(annotation_line: str) -> str:
    """
    Convert a YOLOv11 polygon segmentation annotation line
    into a YOLO bounding box annotation line.

    Input format:
        class_id x1 y1 x2 y2 x3 y3 ... xn yn

    Output format:
        class_id x_center y_center width height
    """

    parts = annotation_line.strip().split()
    
    # First value is class ID
    class_id = parts[0]
    
    # Remaining values are polygon coordinates
    coords = list(map(float, parts[1:]))

    # Separate x and y values
    xs = coords[0::2]
    ys = coords[1::2]

    # Bounding rectangle extremes
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    # YOLO format requires center + width/height
    x_center = (min_x + max_x) / 2
    y_center = (min_y + max_y) / 2
    width = max_x - min_x
    height = max_y - min_y

    return f"{class_id} {x_center} {y_center} {width} {height}"

def readfile(path):
    lines = []
    with open(path, 'r') as file:
        for line in file:
            line = line.strip()
            lines.append(line)
    return(lines)

def get_all_files_os_walk(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Construct the full file path
            full_path = os.path.join(root, file)
            file_list.append(full_path)
    return file_list
        

#print(polygon_to_bbox("2 0.0654953125 0.9028562499999999 0.0962625 0.9129208333333333 0.07861875 0.9444250000000001 0.0098453125 0.9709729166666666 0.00789375 0.9979166666666667 0.082696875 0.9979166666666667 0.08768125 0.98686875 0.1480296875 0.9205083333333334 0.1408421875 0.8636541666666666 0.10556406250000001 0.8552812500000001 0.0298296875 0.8463729166666667 0.022678125 0.9081604166666666 0.0654953125 0.9028562499999999"))
#print(os.listdir("test/labels"))
#print(readfile("test/labels/d5_0001_AI_Albedo_png.rf.a1b48da2abd2ebddf747d8a1b0b88157.txt"))

source_directories = ["test/labels"]
output_directory = "rectangulafied"

for source_directory in source_directories:
    files = os.listdir(source_directory)
    for file in files:
        labels = readfile(source_directory+"/"+file)
        converted = []
        for label in labels:
            converted.append(polygon_to_bbox(label))

        with open(output_directory+"/"+file, 'w') as f:
            for new_label in converted:
                f.write(f"{new_label}\n")

        print(converted)