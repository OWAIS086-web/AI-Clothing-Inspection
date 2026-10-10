import json

import paddle
from paddleocr import PaddleOCR


class PaddleOCRCompat:
    def __init__(self, **settings):
        use_gpu = settings.get("use_gpu", False) and paddle.is_compiled_with_cuda()
        device = f"gpu:{settings.get('gpu_id', 0)}" if use_gpu else "cpu"

        self.engine = PaddleOCR(
            lang=settings.get("lang", "en"),
            ocr_version=settings.get("ocr_version", "PP-OCRv4"),
            device=device,
            use_textline_orientation=settings.get(
                "use_angle_cls", settings.get("se_angle_cls", False)
            ),
            text_recognition_batch_size=settings.get("rec_batch_num", 6),
            text_det_limit_side_len=settings.get("det_limit_side_len", 960),
            text_det_limit_type=settings.get("det_limit_type", "max"),
            text_det_thresh=settings.get("det_db_thresh", 0.3),
            text_det_box_thresh=settings.get("det_db_box_thresh", 0.6),
            text_det_unclip_ratio=settings.get("det_db_unclip_ratio", 1.5),
            text_rec_score_thresh=settings.get("drop_score", 0.5),
        )

    @staticmethod
    def _convert_result(result):
        data = getattr(result, "json", result)
        if callable(data):
            data = data()
        if isinstance(data, str):
            data = json.loads(data)
        if isinstance(data, dict) and "res" in data:
            data = data["res"]

        polygons = data.get("rec_polys")
        if polygons is None:
            polygons = data.get("dt_polys", [])
        texts = data.get("rec_texts", [])
        scores = data.get("rec_scores", [])

        lines = []
        for polygon, text, score in zip(polygons, texts, scores):
            if hasattr(polygon, "tolist"):
                polygon = polygon.tolist()
            if hasattr(score, "item"):
                score = score.item()
            lines.append([polygon, (text, float(score))])
        return lines

    def ocr(self, image, det=True, rec=True, cls=False, **kwargs):
        results = self.engine.predict(image)
        return [self._convert_result(result) for result in results]