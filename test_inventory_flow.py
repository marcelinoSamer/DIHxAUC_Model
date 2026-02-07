import requests
import time
import pandas as pd
import io

BASE_URL = "http://127.0.0.1:8000"

def test_inventory_flow():
    print("🚀 Starting Inventory Data Flow Test...")
    
    # 1. Create a Restaurant and Menu Item to link inventory to
    print("\n1. Setting up metadata (Restaurant & Item)...")
    try:
        # Create Restaurant
        r = requests.post(f"{BASE_URL}/restaurants", json={"name": "Inventory Bistro", "location": "Uptown"})
        r_id = r.json()['id']
        
        # Create Item
        i = requests.post(f"{BASE_URL}/menu-items", json={
            "restaurant_id": r_id,
            "name": "Wagyu Burger",
            "price": 25.0,
            "category": "Main",
            "description": "Premium burger"
        })
        item_id = i.json()['id']
        print(f"   ✅ Created Item: Wagyu Burger (ID: {item_id})")
    except Exception as e:
        print(f"   ❌ Setup failed: {e}")
        return

    # 2. Prepare Inventory CSV
    print("\n2. Preparing Inventory CSV...")
    # Simulation: Current stock 5, Reorder Point 10 -> Should trigger LOW STOCK alert
    csv_data = f"""item_id,current_stock,reorder_point,safety_stock
{item_id},5,10,2
"""
    files = {'file': ('inventory.csv', csv_data, 'text/csv')}

    # 3. Ingest Data
    print("3. Ingesting Inventory Data...")
    response = requests.post(f"{BASE_URL}/inventory/ingest", files=files)
    if response.status_code == 200:
        print("   ✅ Ingestion successful:", response.json())
    else:
        print("   ❌ Ingestion failed:", response.text)
        return

    # 4. Fetch Recommendations
    print("\n4. Fetching Weekly Recommendations...")
    response = requests.get(f"{BASE_URL}/recommendations/weekly")
    if response.status_code == 200:
        recs = response.json()
        print(f"   ✅ Received {len(recs)} recommendations")
        for rec in recs:
            print(f"      - [{rec['priority']}] {rec['type']}: {rec['item']} -> {rec['message']}")
            
        # Verify specific logic
        found = any(r['item'] == "Wagyu Burger" and r['type'] == "Restock Alert" for r in recs)
        if found:
            print("   🎉 VERIFICATION PASSED: Wagyu Burger needs restocking.")
        else:
            print("   ⚠️ VERIFICATION FAILED: Wagyu Burger alert not found.")
    else:
        print("   ❌ Failed to fetch recommendations:", response.text)

if __name__ == "__main__":
    test_inventory_flow()
