import cv2

is_drawing = False      
x0, y0 = -1, -1         
roi = None             
img = None
temp_img = None

def mouse_callback(event, x, y, flags, param):
    """
    마우스 이벤트에 따라 사각형 그리기 및 ROI 선택
    """
    global x0, y0, is_drawing, img, temp_img, roi

    if event == cv2.EVENT_LBUTTONDOWN:

        is_drawing = True
        x0, y0 = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if is_drawing:
            temp_img = img.copy()
            cv2.rectangle(temp_img, (x0, y0), (x, y), (0, 255, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        is_drawing = False
        cv2.rectangle(img, (x0, y0), (x, y), (0, 255, 0), 2)
        temp_img = img.copy()

        x1, y1 = sorted([x0, x]), sorted([y0, y])
        roi = img[y1[0]:y1[1], x1[0]:x1[1]]

        if roi.size > 0:
            cv2.imshow('ROI', roi)

def main():
    global img, temp_img


    img = cv2.imread('./image/soccer.jpg')
    if img is None:
        print("이미지를 불러올 수 없습니다. 경로를 확인하세요.")
        return
    

    temp_img = img.copy()


    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', mouse_callback)

    while True:
        cv2.imshow('Image', temp_img)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            if roi is not None and roi.size > 0:
                cv2.imwrite('./output/roi.jpg', roi)
                print("ROI 영역을 roi.jpg로 저장했습니다.")
            else:
                print("ROI가 설정되지 않았습니다.")
        
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
