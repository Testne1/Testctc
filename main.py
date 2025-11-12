import sys
from ultralytics import YOLO

# --- ĐỊNH NGHĨA CÁC ĐƯỜNG DẪN ---

# Đường dẫn đến mô hình của bạn
MODEL_PATH = 'best.pt'

# Đường dẫn đến ảnh bạn muốn dự đoán
# Lấy từ tham số dòng lệnh, ví dụ: python run.py test.jpg
if len(sys.argv) < 2:
    print("Lỗi: Vui lòng cung cấp đường dẫn đến ảnh.")
    print("Cách dùng: python run.py path/to/image.jpg")
    sys.exit(1)

IMAGE_PATH = sys.argv[1]

# ------------------------------------

try:
    # 1. Tải mô hình
    print(f"Đang tải mô hình từ '{MODEL_PATH}'...")
    model = YOLO(MODEL_PATH)

    # 2. Chạy dự đoán
    # save=True sẽ lưu ảnh kết quả vào thư mục 'runs/detect/predict/'
    print(f"Đang dự đoán trên ảnh '{IMAGE_PATH}'...")
    results = model.predict(source=IMAGE_PATH, conf=0.5, save=True)
    
    # 3. In kết quả ra terminal
    print("\n--- KẾT QUẢ DỰ ĐOÁN ---")
    
    # results là một danh sách, lấy phần tử đầu tiên
    result = results[0]

    if len(result.boxes) == 0:
        print("Không phát hiện được đối tượng nào.")
    else:
        print(f"Phát hiện được {len(result.boxes)} đối tượng:")
        
        # Lặp qua từng đối tượng
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            class_name = model.names[class_id]
            confidence = round(box.conf[0].item(), 2)
            coords = [round(c) for c in box.xyxy[0].tolist()] # [x1, y1, x2, y2]

            print(f"  + Lớp (Class): {class_name}")
            print(f"    Độ tự tin (Confidence): {confidence}")
            print(f"    Tọa độ (Coordinates): {coords}")
            print("-" * 20)

    # In đường dẫn ảnh đã lưu
    print(f"\nẢnh kết quả (đã vẽ hộp) được lưu tại: {result.save_dir}")

except FileNotFoundError as e:
    print(f"LỖI: Không tìm thấy file. {e}")
except Exception as e:
    print(f"Đã xảy ra lỗi: {e}")

