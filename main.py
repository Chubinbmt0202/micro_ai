from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from deepface import DeepFace
from contextlib import asynccontextmanager
import urllib.request
import ssl
import numpy as np
import cv2
import time
import threading

# Tắt xác thực SSL để sửa lỗi CERTIFICATE_VERIFY_FAILED khi tải ảnh/tải model
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass


# =======================================================
# KHÓA BẢO VỆ AI (Ngăn sập server khi có nhiều người điểm danh cùng lúc)
# =======================================================
ai_lock = threading.Lock()

# =======================================================
# HÀM WARM-UP: KHỞI ĐỘNG AI NGAY KHI BẬT SERVER
# =======================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("⏳ Đang nạp mô hình AI vào RAM (Warm-up)...")
    start_time = time.time()
    try:
        # Tạo một bức ảnh đen giả (100x100 pixel) để mồi cho AI chạy thử
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        DeepFace.represent(img_path=dummy_img, model_name="Facenet", enforce_detection=False)
        print(f"✅ Nạp AI thành công! Mất {round(time.time() - start_time, 2)} giây.")
        print("🚀 Server Python AI đã sẵn sàng nhận ảnh tốc độ cao!")
    except Exception as e:
        print(f"⚠️ Lỗi lúc nạp AI: {e}")
    yield # Cho phép server chạy tiếp

app = FastAPI(title="MindCheck AI Service", lifespan=lifespan)

class ImageRequest(BaseModel):
    url: str

# API kiểm tra sức khỏe server
@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "Python Microservice is running!"}

# =======================================================
# API TRÍCH XUẤT KHUÔN MẶT (CHẠY ĐA LUỒNG - KHÔNG CÓ async)
# =======================================================
@app.post("/api/extract")
def extract_face(request: ImageRequest):
    try:
        # 1. TẢI ẢNH TỪ URL (Các luồng chạy song song, tải rất nhanh)
        req = urllib.request.urlopen(request.url)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Không thể đọc được ảnh từ URL này.")

        # 2. ÉP CÂN ẢNH BẰNG OPENCV (Thu nhỏ ảnh 4K về 500px siêu nhẹ)
        height, width = img.shape[:2]
        new_width = 500
        new_height = int((new_width / width) * height)
        img_resized = cv2.resize(img, (new_width, new_height))

        # 3. QUÉT MẶT & TRÍCH XUẤT VECTOR (Xếp hàng tuần tự qua Lock)
        with ai_lock:
            # Tham số enforce_detection=True bắt buộc ảnh phải có khuôn mặt
            faces = DeepFace.represent(img_path=img_resized, model_name="Facenet", enforce_detection=True)
            embedding = faces[0]["embedding"]

        return {
            "success": True,
            "embedding": embedding
        }

    except ValueError:
        # Lỗi này văng ra khi DeepFace không tìm thấy mặt trong ảnh
        raise HTTPException(status_code=400, detail="Không tìm thấy khuôn mặt trong ảnh.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý AI: {str(e)}")