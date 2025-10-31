"""
API Testing Script
Test the FastAPI endpoints with sample requests
"""
import requests
import json
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:8000"
TEST_IMAGE_PATH = "test_images/sample.jpg"  # Replace with your test image


def test_health_check():
    """Test the health check endpoint"""
    print("\n" + "=" * 70)
    print("Testing Health Check Endpoint")
    print("=" * 70)
    
    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    return response.status_code == 200


def test_detect_hijab(image_path):
    """Test the hijab detection endpoint"""
    print("\n" + "=" * 70)
    print("Testing Hijab Detection Endpoint")
    print("=" * 70)
    
    if not Path(image_path).exists():
        print(f"✗ Test image not found: {image_path}")
        print("Please create a 'test_images' folder and add a sample image")
        return False
    
    # Prepare the file
    files = {
        'file': ('test_image.jpg', open(image_path, 'rb'), 'image/jpeg')
    }
    
    # Send POST request
    response = requests.post(f"{API_BASE_URL}/api/detect-hijab", files=files)
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Detection successful!")
        print(f"  Image Name: {result['image_name']}")
        print(f"  Hijab Count: {result['hijab_count']}")
        print(f"  Timestamp: {result['timestamp']}")
        print(f"  Message: {result['message']}")
        return True
    else:
        print(f"✗ Detection failed: {response.text}")
        return False


def test_get_all_results():
    """Test getting all detection results"""
    print("\n" + "=" * 70)
    print("Testing Get All Results Endpoint")
    print("=" * 70)
    
    response = requests.get(f"{API_BASE_URL}/api/results")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        results = response.json()
        print(f"✓ Retrieved {len(results)} detection records")
        
        if results:
            print("\nLatest 5 records:")
            for i, record in enumerate(results[:5], 1):
                print(f"  {i}. {record['image_name']} - {record['hijab_count']} hijabs - {record['timestamp']}")
        else:
            print("  No records found in database")
        
        return True
    else:
        print(f"✗ Failed to retrieve results: {response.text}")
        return False


def test_get_specific_result(image_name):
    """Test getting a specific detection result"""
    print("\n" + "=" * 70)
    print("Testing Get Specific Result Endpoint")
    print("=" * 70)
    
    response = requests.get(f"{API_BASE_URL}/api/results/{image_name}")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Record found!")
        print(f"  Image Name: {result['image_name']}")
        print(f"  Hijab Count: {result['hijab_count']}")
        print(f"  Timestamp: {result['timestamp']}")
        return True
    elif response.status_code == 404:
        print(f"✗ Record not found")
        return False
    else:
        print(f"✗ Error: {response.text}")
        return False


def test_delete_result(image_name):
    """Test deleting a detection result"""
    print("\n" + "=" * 70)
    print("Testing Delete Result Endpoint")
    print("=" * 70)
    
    response = requests.delete(f"{API_BASE_URL}/api/results/{image_name}")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ {result['message']}")
        return True
    elif response.status_code == 404:
        print(f"✗ Record not found")
        return False
    else:
        print(f"✗ Error: {response.text}")
        return False


def main():
    """Run all API tests"""
    print("\n" + "=" * 70)
    print("HIJAB DETECTION API - TEST SUITE")
    print("=" * 70)
    print(f"API Base URL: {API_BASE_URL}")
    print("=" * 70)
    
    # Test 1: Health Check
    if not test_health_check():
        print("\n✗ Server is not running or not accessible")
        print("Please start the server with: python run_server.py")
        return
    
    # Test 2: Detect Hijab
    detection_success = test_detect_hijab(TEST_IMAGE_PATH)
    
    # Test 3: Get All Results
    test_get_all_results()
    
    # Test 4: Get Specific Result (if detection was successful)
    if detection_success:
        # Get the latest result
        response = requests.get(f"{API_BASE_URL}/api/results")
        if response.status_code == 200:
            results = response.json()
            if results:
                latest_image = results[0]['image_name']
                test_get_specific_result(latest_image)
                
                # Test 5: Delete Result (optional)
                delete_choice = input("\nDelete the test record? (y/n): ").strip().lower()
                if delete_choice == 'y':
                    test_delete_result(latest_image)
    
    print("\n" + "=" * 70)
    print("TEST SUITE COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("\n✗ Cannot connect to the API server")
        print("Please ensure the server is running: python run_server.py")
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
