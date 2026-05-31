import numpy as np
import onnxruntime as ort
import cv2


class CatDetector:
    def __init__(self, onnx_path, imgsz=640, conf=0.25):
        self.session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"]
        )

        self.imgsz = imgsz
        self.conf = conf

        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]

        img = cv2.resize(img, (self.imgsz, self.imgsz))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        return img, (w0, h0)

    def nms(self, boxes, scores, iou_thres=0.45):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)

            iou = (w * h) / (areas[i] + areas[order[1:]] - w * h + 1e-6)

            order = order[np.where(iou <= iou_thres)[0] + 1]

        return keep

    def predict(self, image_path):
        img = cv2.imread(image_path)

        if img is None:
            return []

        input_tensor, (w0, h0) = self.preprocess(img)

        preds = self.session.run(None, {self.input_name: input_tensor})[0]
        preds = np.squeeze(preds, axis=0)

        # YOLO format: x1 y1 x2 y2 conf class
        preds = preds[preds[:, 4] >= self.conf]

        if len(preds) == 0:
            return []

        boxes = preds[:, :4]
        scores = preds[:, 4]

        keep = self.nms(boxes, scores)

        preds = preds[keep]

        # scale back to original image size
        scale_x = w0 / self.imgsz
        scale_y = h0 / self.imgsz

        results = []

        for p in preds:
            x1, y1, x2, y2, conf, cls = p

            results.append({
                "x1": float(x1 * scale_x),
                "y1": float(y1 * scale_y),
                "x2": float(x2 * scale_x),
                "y2": float(y2 * scale_y),
                "confidence": float(conf),
                "class": float(cls)
            })

        return results