import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def test_api():
    # Wait for server to start
    print("Waiting for server...")
    time.sleep(5)
    
    # 1. Create Restaurant
    print("Creating Restaurant...")
    restaurant_data = {"name": "Test Bistro", "location": "Downtown"}
    response = requests.post(f"{BASE_URL}/restaurants", json=restaurant_data)
    if response.status_code == 200:
        print("✅ Restaurant created:", response.json())
        restaurant_id = response.json()['id']
    else:
        print("❌ Failed to create restaurant:", response.text)
        return

    # 2. Create Menu Item
    print("Creating Menu Item...")
    item_data = {
        "restaurant_id": restaurant_id,
        "name": "Truffle Fries",
        "price": 45.0,
        "category": "Star",
        "description": "Crispy fries with truffle oil"
    }
    response = requests.post(f"{BASE_URL}/menu-items", json=item_data)
    if response.status_code == 200:
        print("✅ Menu Item created:", response.json())
        item_id = response.json()['id']
    else:
        print("❌ Failed to create menu item:", response.text)
        return

    # 3. Create Order
    print("Creating Order...")
    order_data = {
        "restaurant_id": restaurant_id,
        "total_amount": 90.0,
        "items": [
            {"menu_item_id": item_id, "quantity": 2, "price": 45.0}
        ]
    }
    response = requests.post(f"{BASE_URL}/orders", json=order_data)
    if response.status_code == 200:
        print("✅ Order created:", response.json())
    else:
        print("❌ Failed to create order:", response.text)
        return

    # 4. Initialize Analysis (triggers data loading from DB)
    print("Initializing Analysis...")
    response = requests.post(f"{BASE_URL}/initialize?data_dir=data")
    if response.status_code == 200:
        print("✅ Analysis initialized:", response.json())
    else:
        print("❌ Failed to initialize analysis:", response.text)

    # 5. Check if item appears in analysis
    print("Verifying Item in Analysis...")
    response = requests.get(f"{BASE_URL}/items?limit=1000")
    if response.status_code == 200:
        items = response.json()
        found = False
        for item in items:
            if item['name'] == "Truffle Fries":
                print("✅ Found 'Truffle Fries' in analysis results!", item)
                found = True
                break
        if not found:
            print("❌ 'Truffle Fries' NOT found in analysis results.")
    else:
        print("❌ Failed to get items:", response.text)

if __name__ == "__main__":
    test_api()
