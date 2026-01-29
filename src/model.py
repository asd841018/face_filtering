from insightface.app import FaceAnalysis

class FaceModel:
    def __init__(self, providers=None):
        # STEP 2: Initialize InsightFace
        print("Initializing InsightFace...")
        self.model = FaceAnalysis(
            name='buffalo_l',
            allowed_modules=["detection", "landmark_2d_106"],
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.model.prepare(ctx_id=0, det_size=(640, 640))

    def detect_faces(self, img):
        faces = self.model.get(img)
        if len(faces) == 0:
            return None
        return faces[0]