# ================================================
# Skeleton codes for HW5
# Read the skeleton codes carefully and put all your
# codes into function "reconstruct_from_binary_patterns"
# ================================================

import sys
import cv2
import numpy as np
from math import log, ceil, floor
import matplotlib.pyplot as plt
import pickle

def help_message():
    # Note: it is assumed that "binary_codes_ids_codebook.pckl", "stereo_calibration.pckl",
    # and images folder are in the same root folder as your "generate_data.py" source file.
    # Same folder structure will be used when we test your program

    print("Usage: [Output_Directory]")
    print("[Output_Directory]")
    print("Where to put your output.xyz")
    print("Example usages:")
    print(sys.argv[0] + " ./")

def reconstruct_from_binary_patterns():
    scale_factor = 1.0
    ref_white = cv2.resize(cv2.imread("images/pattern000.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)
    ref_black = cv2.resize(cv2.imread("images/pattern001.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)
    ref_avg   = (ref_white + ref_black) / 2.0
    ref_on    = ref_avg + 0.05 # a threshold for ON pixels
    ref_off   = ref_avg - 0.05 # add a small buffer region

    colour_image = cv2.resize(cv2.imread("images/pattern001.jpg", cv2.IMREAD_COLOR), (0,0), fx=scale_factor,fy=scale_factor)
    # print("Size of colour_image is: ", colour_image.shape)
    
    # cv2.namedWindow("InitialImages", cv2.WINDOW_NORMAL)
    # cv2.imshow("InitialImages", ref_white)
    # cv2.waitKey(0)
    # cv2.imshow("InitialImages", ref_black)
    # cv2.waitKey(0)
    # cv2.imshow("InitialImages", ref_avg)
    # cv2.waitKey(0)
    # cv2.imshow("InitialImages", ref_on)
    # cv2.waitKey(0)
    # cv2.imshow("InitialImages", ref_off)
    # cv2.waitKey(0)

    h,w = ref_white.shape

    # mask of pixels where there is projection
    proj_mask = (ref_white > (ref_black + 0.05))

    # cv2.imshow("InitialImages", np.array(proj_mask, dtype=np.uint8) * 255)
    # cv2.waitKey(0)

    scan_bits = np.zeros((h,w), dtype=np.uint16)
    red_channel = np.zeros((h,w), dtype=np.float32)
    green_channel = np.zeros((h,w), dtype=np.float32)
    blue_channel = np.zeros((h,w), dtype=np.float32)

    # analyze the binary patterns from the camera
    for i in range(0,15):
        # read the file
        patt_gray = cv2.resize(cv2.imread("images/pattern%03d.jpg"%(i+2), cv2.IMREAD_GRAYSCALE) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)

        #cv2.imshow("InitialImages", patt_gray)
        #cv2.waitKey(0)

        # mask where the pixels are ON
        on_mask = (patt_gray > ref_on) & proj_mask

        #cv2.imshow("InitialImages", np.array(on_mask, dtype=np.uint8) * 255)
        #cv2.waitKey(0)

        # this code corresponds with the binary pattern code
        bit_code = np.uint16(1 << i)

        # TODO: populate scan_bits by putting the bit_code according to on_mask
        scan_bits = scan_bits + on_mask * bit_code


    # plt.imshow(scan_bits)
    # plt.show()

    print("load codebook")
    # the codebook translates from <binary code> to (x,y) in projector screen space
    with open("binary_codes_ids_codebook.pckl","r") as f:
        binary_codes_ids_codebook = pickle.load(f)

    # print("Going to print what is in the binary codebook.")
    # print(binary_codes_ids_codebook)
    # print("End of printing the binary codebook.")

    camera_points = []
    projector_points = []
    rgb_points = []
    for x in range(w):
        for y in range(h):
            if not proj_mask[y,x]:
                continue # no projection here
            if scan_bits[y,x] not in binary_codes_ids_codebook:
                continue # bad binary code

            # TODO: use binary_codes_ids_codebook[...] and scan_bits[y,x] to
            # TODO: find for the camera (x,y) the projector (p_x, p_y).
            # TODO: store your points in camera_points and projector_points

            # IMPORTANT!!! : due to differences in calibration and acquisition - divide the camera points by 2

            # Write code here.
            x_p, y_p = binary_codes_ids_codebook[scan_bits[y, x]]
            b = colour_image[y, x, 0]
            g = colour_image[y, x, 1]
            r = colour_image[y, x, 2] 

            # ...
            # ... obtain x_p, y_p - the projector coordinates from the codebook
            if x_p >= 1279 or y_p >= 799: # filter
                continue

            camera_points.append([x / 2.0, y / 2.0])
            projector_points.append([x_p, y_p])
            rgb_points.append([r, g, b])

            red_channel[y,x] = (x_p / 1280.0)
            green_channel[y,x] = (y_p / 800.0)



    # print("Before convertint to nparray.")
    # print("camera_points shape is:", len(camera_points), len(camera_points[0]))
    # print("projector_points shape is:", len(projector_points), len(projector_points[0]))
    camera_points = np.array([camera_points], dtype = np.float32)
    projector_points = np.array([projector_points], dtype = np.float32)
    rgb_points = np.array(rgb_points)
    # camera_points = camera_points.transpose(1,0,2)
    # projector_points = projector_points.transpose(1,0,2)
    # print("After convertint to nparray.")
    # print("camera_points shape is:", camera_points.shape)
    # print("projector_points shape is:", projector_points.shape)

    Image = cv2.merge((blue_channel, green_channel, red_channel))
    CorrespondenceImageName = sys.argv[1] + "correspondence.jpg"
    cv2.imwrite(CorrespondenceImageName, Image * 255)
    #cv2.namedWindow("CorrespondenceImage", cv2.WINDOW_NORMAL)
    #cv2.imshow("CorrespondenceImage",Image)
    #cv2.waitKey(0)
    # TODO: write code here to save the correspondence image as correspondence.jpg in the Results folder as mentioned in HW5 document.

    # now that we have 2D-2D correspondances, we can triangulate 3D points!

    # load the prepared stereo calibration between projector and camera
    with open("stereo_calibration.pckl","r") as f:
        d = pickle.load(f)
        camera_K    = d['camera_K']
        camera_d    = d['camera_d']
        projector_K = d['projector_K']
        projector_d = d['projector_d']
        projector_R = d['projector_R']
        projector_t = d['projector_t']

    # print("camera_K is: ", camera_K)
    # print("camers_d is: ", camera_d)
    # print("projector_K is: ", projector_K)
    # print("projector_d is: ", projector_d)
    # print("projector_R is: ", projector_R)
    # print("projector_t is: ", projector_t)

    # TODO: use cv2.undistortPoints to get normalized points for camera, use camera_K and camera_d
    camera_normalizedPoints = cv2.undistortPoints(camera_points, camera_K, camera_d)

    # TODO: use cv2.undistortPoints to get normalized points for projector, use projector_K and projector_d
    projector_normalizedPoints = cv2.undistortPoints(projector_points, projector_K, projector_d)

    # TODO: use cv2.triangulatePoints to triangulate the normalized points
    CameraProjectionMatrix = np.hstack((np.identity(3), np.zeros((3, 1)))).astype(np.float32)
    # print("Type of CameraProjectionMatrix is: ", type(CameraProjectionMatrix)) 
    # print("CameraProjectionMatrix is: ", CameraProjectionMatrix)
    ProjectorProjectionMatrix = np.hstack((projector_R, projector_t)).astype(np.float32)
    # print("Type of ProjectorProjectionMatrix is: ", type(ProjectorProjectionMatrix))
    # print("ProjectorProjectionMatrix is: ", ProjectorProjectionMatrix)
    TriangulatedPoints = cv2.triangulatePoints(CameraProjectionMatrix, ProjectorProjectionMatrix, camera_normalizedPoints, projector_normalizedPoints)
    # print("TriangulatedPoints shape is: ", TriangulatedPoints.shape)
 
    # TODO: use cv2.convertPointsFromHomogeneous to get real 3D points
    # TODO: name the resulted 3D points as "points_3d"
    points_3d = cv2.convertPointsFromHomogeneous(np.transpose(TriangulatedPoints))
    # print("points_3d shape is: ", points_3d.shape)
    # print("rgb_points shape is: ", rgb_points.shape)
    points_3d = np.hstack((points_3d.reshape(-1, 3), rgb_points))
    # print("points_3d shape after merging colours is: ", points_3d.shape)
    mask = (points_3d[:,2] > 200) & (points_3d[:,2] < 1400)
    points_3d = points_3d[mask]
    # print("points_3d shape after final masking is: ", points_3d.shape)
    return points_3d
	
def write_3d_points(points_3d):
	
    # ===== DO NOT CHANGE THIS FUNCTION =====
	
    print("write output point cloud")
    print("points_3d shape is: ", points_3d.shape)
    output_name = sys.argv[1] + "output.xyz"
    with open(output_name,"w") as f:
        for p in points_3d:
            f.write("%d %d %d\n"%(p[0], p[1], p[2]))

    output_name = sys.argv[1] + "output_color.xyz"
    with open(output_name,"w") as f:
        for p in points_3d:
            f.write("%d %d %d %d %d %d\n"%(p[0], p[1], p[2], p[3], p[4], p[5]))

    return points_3d

    
if __name__ == '__main__':

    # ===== DO NOT CHANGE THIS FUNCTION =====
	
    # validate the input arguments
    if (len(sys.argv) != 2):
        help_message()
        sys.exit()

    points_3d = reconstruct_from_binary_patterns()
    write_3d_points(points_3d)
	
