import cv2
import numpy as np
from typing import List, Tuple

def help():
    print("\nSigh: This program is not complete/will be replaced. \n"
          "So:   Use this just to see hints of how to use things like Rodrigues\n"
          "      conversions, finding the fundamental matrix, using descriptor\n"
          "      finding and matching in features2d and using camera parameters\n"
          "Usage: build3dmodel -i <intrinsics_filename>\n"
          "\t[-d <detector>] [-de <descriptor_extractor>] -m <model_name>\n\n")
    return

def readCameraMatrix(filename: str):
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
    cameraMatrix = fs.getNode("camera_matrix").mat()
    distCoeffs = fs.getNode("distortion_coefficients").mat()
    width = int(fs.getNode("image_width").real()) 
    height = int(fs.getNode("image_height").real())
    fs.release()

    calibratedImageSize = (width, height)
    if distCoeffs.dtype != np.float64:
        distCoeffs = distCoeffs.astype(np.float64)
    if cameraMatrix.dtype != np.float64:
        cameraMatrix = cameraMatrix.astype(np.float64)

    return cameraMatrix, distCoeffs, calibratedImageSize

def readModelViews(filename: str):
    imagelist = []
    roiList = []
    poseList = []
    box = [] #que es box? 3dbounding box?

    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        return False
    box = fs.getNode("box").mat()

    views = fs.getNode("views")
    for i in range(views.size()):
        node = views.at(i)
        imagelist.append(node.getNode("image").string())
        roi = node.getNode("roi").mat()
        roiList.append((int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])))
        pose = node.getNode("pose").mat()
        poseList.append(tuple(pose))

    fs.release()
    return box, imagelist, roiList, poseList

class PointModel:
    def __init__(self):
        self.points = []
        self.didx = []
        self.descriptors = None
        self.name = ""

def writeModel(modelFileName: str, modelname: str, model: PointModel):
    fs = cv2.FileStorage(modelFileName, cv2.FILE_STORAGE_WRITE)
    fs.write(modelname, {
        "points": np.array(model.points),
        "idx": np.array([np.array(i) for i in model.didx]),
        "descriptors": model.descriptors
    })
    fs.release()

def unpackPose(pose: Tuple[float, float, float, float, float, float]):
    rvec = np.array(pose[:3], dtype=np.float64).reshape((3, 1))
    t = np.array(pose[3:], dtype=np.float64).reshape((3, 1))
    R = cv2.Rodrigues(rvec)[0]
    return R, t

def getFundamentalMat(R1, t1, R2, t2, cameraMatrix): # xq el vector es en 2d?
    R = R2 @ R1.T
    t = t2 - R @ t1
    tx, ty, tz = t[0, 0], t[1, 0], t[2, 0]
    E = np.array([[0, -tz, ty],
                  [tz, 0, -tx],
                  [-ty, tx, 0]]) @ R
    iK = np.linalg.inv(cameraMatrix)
    F = iK.T @ E @ iK

    return F

def findConstrainedCorrespondences(F, keypoints1, keypoints2, descriptors1, descriptors2, matches, eps, ratio):
    F = F.astype(np.float32)
    matches.clear()
    dsize = descriptors1.shape[1]

    for i, kp1 in enumerate(keypoints1):
        p1 = kp1.pt
        bestDist1 = float('inf')
        bestDist2 = float('inf')
        bestIdx1 = -1
        d1 = descriptors1[i]

        for j, kp2 in enumerate(keypoints2):
            p2 = kp2.pt
            e = (p2[0] * (F[0, 0] * p1[0] + F[0, 1] * p1[1] + F[0, 2]) +
                 p2[1] * (F[1, 0] * p1[0] + F[1, 1] * p1[1] + F[1, 2]) +
                 (F[2, 0] * p1[0] + F[2, 1] * p1[1] + F[2, 2]))
            if abs(e) > eps:
                continue
            d2 = descriptors2[j]
            dist = np.sum((d1 - d2) ** 2)

            if dist < bestDist2:
                if dist < bestDist1:
                    bestDist2 = bestDist1
                    bestDist1 = dist
                    bestIdx1 = j
                else:
                    bestDist2 = dist

        if bestIdx1 >= 0 and bestDist1 < bestDist2 * ratio:
            matches.append([i, bestIdx1])

def findRayIntersection(k1, b1, k2, b2):
    A = np.array([[np.dot(k1, k1), -np.dot(k1, k2)],
                  [-np.dot(k1, k2), np.dot(k2, k2)]])
    B = np.array([np.dot(k1, b2 - b1), np.dot(k2, b1 - b2)])
    s1, s2 = np.linalg.solve(A, B)
    return (k1 * s1 + b1 + k2 * s2 + b2) * 0.5

def triangulatePoint(ps, Rs, ts, cameraMatrix):
    K = np.array(cameraMatrix, dtype=np.float64)
    iK = np.linalg.inv(K)
    R1t = np.array(Rs[0], dtype=np.float32).T
    R2t = np.array(Rs[1], dtype=np.float32).T
    m1 = np.array([ps[0][0], ps[0][1], 1], dtype=np.float32).reshape((3, 1))
    m2 = np.array([ps[1][0], ps[1][1], 1], dtype=np.float32).reshape((3, 1))
    K1 = R1t @ (iK @ m1)
    K2 = R2t @ (iK @ m2)
    B1 = -R1t @ np.array(ts[0], dtype=np.float32)
    B2 = -R2t @ np.array(ts[1], dtype=np.float32)
    return findRayIntersection(K1.ravel(), B1.ravel(), K2.ravel(), B2.ravel())

def triangulatePoint_test():
    n = 100
    objpt = np.random.uniform(-10, 10, (n, 3)).astype(np.float32)
    delta1 = np.random.uniform(-1e-2, 1e-2, (n, 3)).astype(np.float32)
    delta2 = np.random.uniform(-1e-2, 1e-2, (n, 3)).astype(np.float32)

    rvec1 = np.random.uniform(-10, 10, (3, 1)).astype(np.float32)
    tvec1 = np.random.uniform(-10, 10, (3, 1)).astype(np.float64)
    rvec2 = np.random.uniform(-10, 10, (3, 1)).astype(np.float32)
    tvec2 = np.random.uniform(-10, 10, (3, 1)).astype(np.float64)

    cameraMatrix = np.array([[1000., 0., 400.5],
                             [0., 1010., 300.5],
                             [0., 0., 1.]], dtype=np.float32)

    imgpt1, _ = cv2.projectPoints(objpt + delta1, rvec1, tvec1, cameraMatrix, None)
    imgpt2, _ = cv2.projectPoints(objpt + delta2, rvec2, tvec2, cameraMatrix, None)

    objptt = []
    Rv, tv = [], []
    Rv.append(cv2.Rodrigues(rvec1)[0])
    Rv.append(cv2.Rodrigues(rvec2)[0])
    tv.append(tvec1)
    tv.append(tvec2)
    for i in range(n):
        pts = [imgpt1[i].ravel(), imgpt2[i].ravel()]
        objptt.append(triangulatePoint(pts, Rv, tv, cameraMatrix))

    objptt = np.array(objptt)
    err = np.linalg.norm(objpt - objptt)
    assert err < 1e-1

# Example usage
if __name__ == "__main__":
    triangulatePoint_test()
