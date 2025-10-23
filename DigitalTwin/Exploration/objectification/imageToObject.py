from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any

class ImageDetect:
    def __init__(self, path: str, model_path: str = "yolov8n.pt") -> None:
        """
        path: 이미지 파일 경로
        model_path: YOLO 가중치 (기본은 빠른 경량 모델 yolov8n.pt)
        """
        self.path = path
        self.image = cv2.imread(path)
        if self.image is None:
            raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {path}")
        self.model = YOLO(model_path)  # 처음 생성 시 한 번만 로드

    def return_ObjectResult(self) -> List[Tuple[Tuple[int, int], str]]:
        """
        [( (center_x, center_y), "label" ), ... ] 형태로 반환
        """
        # YOLO 추론
        results = self.model(self.image, verbose=False)

        out: List[Tuple[Tuple[int, int], str]] = []
        for r in results:
            boxes = r.boxes  # 바운딩 박스들
            names = r.names  # {class_id: class_name}
            for b in boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
                cls_id = int(b.cls.item())
                label = names.get(cls_id, str(cls_id))
                # 중심 좌표 계산 (정수로 반올림)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                out.append(((cx, cy), label))
        return out


    def draw_and_save(self, save_path: str = "result.jpg") -> str:
        """
        탐지 결과를 이미지 위에 그려서 저장. 저장 경로를 반환.
        """
        results = self.model(self.image, verbose=False)
        img = self.image.copy()
        for r in results:
            boxes = r.boxes
            names = r.names
            for b in boxes:
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].cpu().numpy().tolist()]
                cls_id = int(b.cls.item())
                conf = float(b.conf.item())
                label = names.get(cls_id, str(cls_id))
                # 박스 & 라벨
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                caption = f"{label} {conf:.2f}"
                cv2.putText(img, caption, (x1, max(y1 - 5, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(save_path, img)
        return save_path


def image_loading(path: str):
    imageObjectDetect = ImageDetect(path)
    return imageObjectDetect.return_ObjectResult()


if __name__ == "__main__":
    path = "test.png"
    result = image_loading(path)
    print(result)          

