import cv2 as cv
import numpy as np

def select_img(video_file, board_pattern, select_all=False, wait_msec=10):
    video = cv.VideoCapture(video_file)
    img_list = []

    if not video.isOpened():
        print("비디오 파일을 열 수 없습니다.")
        return img_list

    # 1. 영상의 고유 FPS (초당 프레임 수) 가져오기
    fps = video.get(cv.CAP_PROP_FPS)
    
    # 2. wait_msec 시간 간격을 프레임 개수로 변환하기
    skip_frames = int(fps * (wait_msec / 1000.0))
    
    # 만약 입력한 대기 시간이 너무 짧아서 건너뛸 프레임이 0이 되면, 최소 1프레임씩(매 프레임) 추출
    if skip_frames < 1:
        skip_frames = 1

    frame_count = 0

    while True:
        # 영상은 무조건 순서대로 정상 속도로 읽어옴
        ret, frame = video.read()
        
        if not ret:
            break
            
        # 3. 현재 프레임 번호가 skip_frames의 배수일 때만 리스트에 담기
        if frame_count % skip_frames == 0:
            img_list.append(frame)
            
        frame_count += 1

    video.release()
    print("""총 {}개의 프레임이 추출되었습니다.""".format(len(img_list)))
    return img_list

def calib_camera_from_chessboard(image_list, board_pattern, board_cellsize, K=None, dist_coeffs=None, calib_flags=None):
    print("체스보드 패턴이 감지된 이미지에서 카메라 보정 수행 중...")
    img_points =[]
    for img in image_list:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        complete, corners = cv.findChessboardCorners(gray, board_pattern)

        if complete:
            img_points.append(corners)
    assert  len(img_points) > 0, "체스보드 패턴이 감지된 이미지가 없습니다." #조건이 참이 아니면 AssertionError 예외 발생

    obj_pts = [[c,r,0] for r in range(board_pattern[1]) for c in range(board_pattern[0])]
    obj_points = [np.array(obj_pts, dtype=np.float32) * board_cellsize] * len(img_points)
    print("calib return sequence")
    return cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], K, dist_coeffs, flags=calib_flags)

def undistort_video(video_file, output_video, K, dist_coeffs):
    video = cv.VideoCapture(video_file)
    if not video.isOpened():
        print("비디오 파일을 열 수 없습니다.")
        return

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    fps = video.get(cv.CAP_PROP_FPS)
    width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    out = cv.VideoWriter(output_video, fourcc, fps, (width, height))

    while True:
        ret, frame = video.read()
        if not ret:
            break
        undistorted_frame = cv.undistort(frame, K, dist_coeffs)
        out.write(undistorted_frame)

    video.release()
    out.release()
    print("보정된 비디오가 {}에 저장되었습니다.".format(output_video))


if __name__ == "__main__":
    video_file = "distorted.mp4"
    output_video = "undistorted.mp4"
    board_pattern = (10, 7) # 체스보드 패턴의 가로, 세로 코너 수
    board_cellsize = 0.025 # 체스보드 셀의 실제 크기 (단위: 미터)

    # 비디오에서 체스보드 패턴이 감지된 프레임 추출
    img_list = select_img(video_file, board_pattern, select_all=True, wait_msec=500)

    # 카메라 보정 수행
    RMSE, K, dist_coeffs, _, _ = calib_camera_from_chessboard(img_list, board_pattern, board_cellsize)

    # K, dist_coeffs, RMSE 출력
    print("Camera Matrix (K):\n", K)
    print("Distortion Coefficients:\n", dist_coeffs)
    print("Reprojection Error:\n", RMSE)

    # 보정된 비디오 저장
    undistort_video(video_file, output_video, K, dist_coeffs)
