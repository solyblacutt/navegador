import cv2
import numpy as np
import sys

def print_help():
    print("""
    Given a list of chessboard images, the number of corners (nx, ny)
    on the chessboards, and a flag: useCalibrated for 
      calibrated (0) or
      uncalibrated 
        (1: use cv2.stereoCalibrate(), 2: compute fundamental
            matrix separately) stereo. 
    Calibrate the cameras and display the
    rectified results along with the computed disparity images.
    """)
    print("Usage:\n python stereo_calib.py -w board_width -h board_height [-nr /*do not view results*/] <image list XML/YML file>\n")
    return 0

def StereoCalib(imagelist, boardSize, useCalibrated=True, showRectified=True):
    if len(imagelist) % 2 != 0:
        print("Error: the image list contains an odd number of elements")
        return

    displayCorners = False
    maxScale = 2
    squareSize = 1.0  # Set this to your actual square size

    imagePoints = [[], []]
    objectPoints = []
    imageSize = None

    nimages = len(imagelist) // 2
    goodImageList = []

    for i in range(nimages):
        for k in range(2):
            filename = imagelist[i*2 + k]
            img = cv2.imread(filename, 0)
            if img is None:
                break
            if imageSize is None:
                imageSize = img.shape[::-1]
            elif img.shape[::-1] != imageSize:
                print(f"The image {filename} has a size different from the first image size. Skipping the pair")
                break
            found = False
            corners = None
            for scale in range(1, maxScale + 1):
                timg = img if scale == 1 else cv2.resize(img, None, fx=scale, fy=scale)
                found, corners = cv2.findChessboardCorners(timg, boardSize,
                                                           cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
                if found:
                    if scale > 1:
                        corners = corners / scale
                    break

            if displayCorners:
                print(filename)
                cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                cv2.drawChessboardCorners(cimg, boardSize, corners, found)
                sf = 640.0 / max(img.shape)
                cimg1 = cv2.resize(cimg, None, fx=sf, fy=sf)
                cv2.imshow("corners", cimg1)
                c = cv2.waitKey(500)
                if c in [27, ord('q'), ord('Q')]:  # Allow ESC to quit
                    sys.exit(-1)

            if not found:
                break
            corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
            imagePoints[k].append(corners)
        if k == 1:
            goodImageList.append(imagelist[i*2])
            goodImageList.append(imagelist[i*2 + 1])

    print(f"{len(goodImageList)//2} pairs have been successfully detected.")
    nimages = len(goodImageList) // 2
    if nimages < 2:
        print("Error: too few pairs to run the calibration")
        return

    objectPoints = np.zeros((nimages, boardSize.height * boardSize.width, 3), np.float32)
    objectPoints[:, :, :2] = np.mgrid[0:boardSize.width, 0:boardSize.height].T.reshape(-1, 2)
    objectPoints *= squareSize
    objectPoints = [objectPoints] * nimages

    print("Running stereo calibration ...")

    cameraMatrix = [np.eye(3, 3, dtype=np.float64) for _ in range(2)]
    distCoeffs = [None, None]
    flags = cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_ZERO_TANGENT_DIST + \
            cv2.CALIB_SAME_FOCAL_LENGTH + cv2.CALIB_RATIONAL_MODEL + \
            cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5

    rms, cameraMatrix[0], distCoeffs[0], cameraMatrix[1], distCoeffs[1], R, T, E, F = cv2.stereoCalibrate(
        objectPoints, imagePoints[0], imagePoints[1],
        cameraMatrix[0], distCoeffs[0],
        cameraMatrix[1], distCoeffs[1],
        imageSize, flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5))
    print(f"done with RMS error={rms}")

    # Calibration quality check
    # because the output fundamental matrix implicitly
    # includes all the output information,
    # we can check the quality of calibration using the
    # epipolar geometry constraint: m2^t*F*m1=0
    err = 0
    npoints = 0
    lines = [[], []]
    for i in range(nimages):
        npt = len(imagePoints[0][i])
        imgpt = [None, None]
        for k in range(2):
            imgpt[k] = cv2.undistortPoints(imagePoints[k][i], cameraMatrix[k], distCoeffs[k], None, cameraMatrix[k])
            lines[k] = cv2.computeCorrespondEpilines(imgpt[k], k + 1, F)

        for j in range(npt):
            errij = abs(imagePoints[0][i][j][0][0] * lines[1][j][0] +
                        imagePoints[0][i][j][0][1] * lines[1][j][1] + lines[1][j][2]) + \
                    abs(imagePoints[1][i][j][0][0] * lines[0][j][0] +
                        imagePoints[1][i][j][0][1] * lines[0][j][1] + lines[0][j][2])
            err += errij
        npoints += npt

    print(f"average reprojection err = {err/npoints}")

    # save intrinsic parameters
    cv_file = cv2.FileStorage("intrinsics.yml", cv2.FILE_STORAGE_WRITE)
    cv_file.write("M1", cameraMatrix[0])
    cv_file.write("D1", distCoeffs[0])
    cv_file.write("M2", cameraMatrix[1])
    cv_file.write("D2", distCoeffs[1])
    cv_file.release()

    R1, R2, P1, P2, Q, validRoi = cv2.stereoRectify(cameraMatrix[0], distCoeffs[0],
                                                    cameraMatrix[1], distCoeffs[1],
                                                    imageSize, R, T, flags=cv2.CALIB_ZERO_DISPARITY,
                                                    alpha=1, newImageSize=imageSize)
    cv_file = cv2.FileStorage("extrinsics.yml", cv2.FILE_STORAGE_WRITE)
    cv_file.write("R", R)
    cv_file.write("T", T)
    cv_file.write("R1", R1)
    cv_file.write("R2", R2)
    cv_file.write("P1", P1)
    cv_file.write("P2", P2)
    cv_file.write("Q", Q)
    cv_file.release()

    isVerticalStereo = abs(P2[1, 3]) > abs(P2[0, 3])

    # Compute and display rectification
    if not showRectified:
        return

    rmap = [[], []]
    rmap[0] = cv2.initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, cv2.CV_16SC2)
    rmap[1] = cv2.initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, cv2.CV_16SC2)

    canvas = None
    sf = 0
    w, h = 0, 0
    if not isVerticalStereo:
        sf = 600.0 / max(imageSize)
        w = int(round(imageSize[0] * sf))
        h = int(round(imageSize[1] * sf))
        canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)
    else:
        sf = 300.0 / max(imageSize)
        w = int(round(imageSize[0] * sf))
        h = int(round(imageSize[1] * sf))
        canvas = np.zeros((h * 2, w, 3), dtype=np.uint8)

    for i in range(nimages):
        for k in range(2):
            img = cv2.imread(goodImageList[i * 2 + k], 0)
            rimg = cv2.remap(img, rmap[k][0], rmap[k][1], cv2.INTER_LINEAR)
            rimg = cv2.cvtColor(rimg, cv2.COLOR_GRAY2BGR)
            if not isVerticalStereo:
                canvas[:, k * w:(k + 1) * w] = cv2.resize(rimg, (w, h))
            else:
                canvas[k * h:(k + 1) * h, :] = cv2.resize(rimg, (w, h))

        if not isVerticalStereo:
            for j in range(0, canvas.shape[0], 16):
                cv2.line(canvas, (0, j), (canvas.shape[1], j), (0, 255, 0), 1)
        else:
            for j in range(0, canvas.shape[1], 16):
                cv2.line(canvas, (j, 0), (j, canvas.shape[0]), (0, 255, 0), 1)

        cv2.imshow("rectified", canvas)
        if cv2.waitKey() == 27:
            break

if __name__ == '__main__':
    import getopt

    if len(sys.argv) < 2:
        print_help()
        sys.exit(-1)

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'w:h:nrs')
    except getopt.GetoptError as err:
        print_help()
        sys.exit(-1)

    board_w = 0
    board_h = 0
    useUncalibrated = False
    showRectified = True

    for opt, arg in opts:
        if opt == '-w':
            board_w = int(arg)
        elif opt == '-h':
            board_h = int(arg)
        elif opt == '-n':
            showRectified = False
        elif opt == '-r':
            useUncalibrated = True
        elif opt == '-s':
            # Extra flag handling can be added here
            pass

    if board_w <= 0 or board_h <= 0:
        print_help()
        sys.exit(-1)

    board_size = (board_w, board_h)

    if len(args) < 1:
        print_help()
        sys.exit(-1)

    imagelist_fn = args[0]

    fs = cv2.FileStorage(imagelist_fn, cv2.FILE_STORAGE_READ)
    imagelist = fs.getNode("images").mat().ravel()
    fs.release()

    if len(imagelist) == 0:
        print(f"Could not load image list: {imagelist_fn}")
        sys.exit(-1)

    StereoCalib(imagelist, board_size, not useUncalibrated, showRectified)
    cv2.destroyAllWindows()
