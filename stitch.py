import cv2
import numpy as np
import random


def harris(img, mainimg):
    operatedImage = np.float32(img)
    corners = cv2.cornerHarris(operatedImage, 2, 5, 0.07)
    cor = cv2.dilate(corners, None)
    mainimg[cor > 0.05 * cor.max()] = [0, 0, 255]
    kps = np.argwhere(corners > 0.01 * corners.max())
    keypoints = [cv2.KeyPoint(kp[1], kp[0], 1) for kp in kps]
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.compute(img, keypoints)
    #cv2.imshow("harris", mainimg)
    return keypoints, descriptors


def shift(img):
    sift = cv2.xfeatures2d.SIFT_create()
    (keypoints, descriptors) = sift.detectAndCompute(img, None)
    img = cv2.drawKeypoints(img, keypoints, None)
    cv2.imshow("sift", img)
    return keypoints, descriptors


def match(des1, des2):
    match_instance = cv2.DescriptorMatcher_create("BruteForce")
    matches = match_instance.knnMatch(des1, des2, 2)
    good_match = []
    count_match = 0
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
            count_match += 1
            good_match.append([m[0]])
    return matches, good_match, len(matches), count_match


def draw(img1, img2, key1, key2, matches):
    img3 = cv2.drawMatchesKnn(img1, key1, img2, key2, matches, None, flags=2)
    #cv2.imshow('matchingg', img3)


def projection(x1, y1, H):
    coord = np.array([x1, y1, 1])
    trans = np.dot(H, coord)
    w = trans[2]
    xx = trans[0] / w
    yy = trans[1] / w
    return xx, yy


def inliers(H, kp_matches, inlierThreshold):
    inlier_count = 0
    for match in kp_matches:
        dist = distance(match, H)
        if dist < inlierThreshold:
            inlier_count = inlier_count + 1

    return inlier_count


def distance(keymatch, affine):
    x1 = keymatch[0].pt[0]
    y1 = keymatch[0].pt[1]
    x2 = keymatch[1].pt[0]
    y2 = keymatch[1].pt[1]

    xp, yp = projection(x1, y1, affine)
    distance = np.sqrt((xp - x2) ** 2 + (yp - y2) ** 2)
    return distance


def get_points(keypoints1, keypoints2, matches):
    points = []
    for i, match in enumerate(matches):
        p1 = keypoints1[match[0].queryIdx]
        p2 = keypoints2[match[0].trainIdx]
        points.append([p1, p2])

    return points


def AffineTransform(Coordinates_1, Coordinates_2):

    if len(Coordinates_1[0]) != 2 or len(Coordinates_2[1]) != 2:
        raise ValueError("Incorrect dimensions")
    elif len(Coordinates_1) != len(Coordinates_2):
        raise ValueError("mismatch coordinates")

    P = np.zeros(shape=(len(Coordinates_1[0]) + 1, len(Coordinates_1)), dtype=np.float)
    Q = np.zeros(shape=(len(Coordinates_2[0]) + 1, len(Coordinates_2)), dtype=np.float)

    P[0, :] = Coordinates_1.transpose()[0, :]
    P[1, :] = Coordinates_1.transpose()[1, :]
    P[2, :] = np.ones(shape=(len(Coordinates_1)), dtype=int)

    Q[0, :] = Coordinates_2.transpose()[0, :]
    Q[1, :] = Coordinates_2.transpose()[1, :]
    Q[2, :] = np.ones(shape=(len(Coordinates_2)), dtype=int)
    if np.linalg.det(np.matmul(P, np.transpose(P)))!=0:
        Affine = np.matmul(Q, np.matmul(np.transpose(P), np.linalg.inv(np.matmul(P, np.transpose(P)))))
        return Affine
    else:
        return 0


def ransac(matches, kp1, kp2, kp_matches, numIterations, inlierThreshold, image1, image2):
    max_inliers = 0
    highest_score_H = np.zeros((3, 3))
    for i in range(0, numIterations):
        inlier_src_pts = np.zeros((4, 2))
        inlier_dst_pts = np.zeros((4, 2))
        for j in range(4):
            randomInt = random.randint(0, len(kp_matches) - 1)
            inlier_src_pts[j] = kp_matches[randomInt][0].pt
            inlier_dst_pts[j] = kp_matches[randomInt][1].pt
        transform = AffineTransform(inlier_src_pts, inlier_dst_pts)
        if np.size(transform)!=1:
            total_inliers = inliers(transform, kp_matches, inlierThreshold)
            if total_inliers > max_inliers:
                max_inliers = total_inliers
                print('\tInliers updated', max_inliers)
                highest_score_H = transform
        else:
            print('error occurred skipping')
    inlier_matches = []
    kp_1 = []
    kp_2 = []
    inlier_src_pts = []
    inlier_dst_pts = []
    for i, match in enumerate(kp_matches):
        dist = distance(match, highest_score_H)
        if dist < inlierThreshold:
            inlier_src_pts.append(match[0].pt)
            inlier_dst_pts.append(match[1].pt)
            kp_1.append(match[0])
            kp_2.append(match[1])
            inlier_matches.append(matches[i])
    inlier_src_pts = np.asarray(inlier_src_pts)
    inlier_dst_pts = np.asarray(inlier_dst_pts)
    A1 = AffineTransform(inlier_src_pts, inlier_dst_pts)
    homInv = np.linalg.inv(A1)
    #matching_image = cv2.vconcat(image1, image2)
    #matching_image = cv2.drawMatchesKnn(image1, kp1, image2, kp2, matches1to2=inlier_matches, outImg=matching_image,
                                        #flags=2)
    # cv2.imwrite( "ransac_image.png",matching_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return A1,homInv


def wrap(i1, i2, homo):
    val = i1.shape[1] + i2.shape[1]
    result_image = cv2.warpPerspective(i1, homo, (val, i1.shape[0]))
    result_image[0:i2.shape[0], 0:i2.shape[1]] = i2
    #cv2.imwrite("result.png", result_image)
    #cv2.imshow("final", result_image)
    return result_image

def stitch(img1,img2):
    img1=cv2.resize(img1,(400,400))
    img2=cv2.resize(img2,(400,400))
    file1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    file2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    k1, d1 = harris(file1, np.copy(img1))
    k2, d2 = harris(file2, np.copy(img2))
    m, gm, count, count1 = match(d1, d2)
    key_points = get_points(k1, k2, gm)
    I1,I2 = ransac(gm, k1, k2, key_points, 100, 10, img1, img2)
    return wrap(img1, img2, I1)
    
"""
if __name__ == '__main__':
    path1 = "rightImage.png"
    path2 = "leftImage.png"
    stitch(path1, path2, 100, 5)
"""
