import requests
import io
from PIL import Image

BASE_URL = "http://localhost:5000"

def create_test_image(size=(640, 480)):
    """Create a test image"""
    img = Image.new('RGB', size, color=(128, 128, 128))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes

def test_predict_animal():
    """Test /predict-animal"""
    print("\n[TEST] /predict-animal")
    
    # Test 1: Send a normal image
    files = {'image': ('test.jpg', create_test_image(), 'image/jpeg')}
    response = requests.post(f"{BASE_URL}/predict-animal", files=files)
    print(f"  Status: {response.status_code}")
    print(f"  Result: {response.json()}")
    
    # Test 2: Do not send any image
    response = requests.post(f"{BASE_URL}/predict-animal")
    print(f"  No image: {response.json()}")

def test_predict_view():
    """Test /predict-view"""
    print("\n[TEST] /predict-view")
    
    # Test 1: Send a normal image
    files = {'image': ('test.jpg', create_test_image(), 'image/jpeg')}
    response = requests.post(f"{BASE_URL}/predict-view", files=files)
    print(f"  Status: {response.status_code}")
    print(f"  Result: {response.json()}")
    
    # Test 2: Do not send any image
    response = requests.post(f"{BASE_URL}/predict-view")
    print(f"  No image: {response.json()}")

def test_predict_bcs():
    """ทดสอบ /predict-bcs"""
    print("\n[TEST] /predict-bcs")
    
    # Test 1: Send all views
    files = {
        'top': ('top.jpg', create_test_image(), 'image/jpeg'),
        'back': ('back.jpg', create_test_image(), 'image/jpeg'),
        'left': ('left.jpg', create_test_image(), 'image/jpeg'),
        'right': ('right.jpg', create_test_image(), 'image/jpeg')
    }
    response = requests.post(f"{BASE_URL}/predict-bcs", files=files)
    print(f"  Status: {response.status_code}")
    print(f"  Result: {response.json()}")
    
    # Test 2: Send partial views
    files = {
        'top': ('top.jpg', create_test_image(), 'image/jpeg'),
        'left': ('left.jpg', create_test_image(), 'image/jpeg')
    }
    response = requests.post(f"{BASE_URL}/predict-bcs", files=files)
    print(f"  Partial views: {response.json()}")
    
    # Test 3: Do not send any images
    response = requests.post(f"{BASE_URL}/predict-bcs")
    print(f"  No images: {response.json()}")

if __name__ == '__main__':
    print("="*50)
    print("API Testing")
    print("="*50)
    
    test_predict_animal()
    test_predict_view()
    test_predict_bcs()
    
    print("\n" + "="*50)
    print("Done")
    print("="*50)