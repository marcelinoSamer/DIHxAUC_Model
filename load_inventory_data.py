import pandas as pd
import requests
import os
from src.database import SessionLocal
from src.models.db_models import InventoryReport, MenuItem
from datetime import datetime

# Path to the dataset
DATA_FILE = "Inventory Management/dim_skus.csv"
API_URL = "http://127.0.0.1:8000/inventory/ingest"

def load_data():
    print(f"🚀 Loading inventory data from {DATA_FILE}...")
    
    if not os.path.exists(DATA_FILE):
        print(f"❌ File not found: {DATA_FILE}")
        return

    # Read CSV
    df = pd.read_csv(DATA_FILE)
    print(f"   Total rows found: {len(df)}")
    
    # Map columns to expected format
    # dim_skus: item_id, quantity, low_stock_threshold
    # Expected: item_id, current_stock, reorder_point, safety_stock
    
    # Note: dim_skus.item_id might be null for some rows (ingredients vs menu items)
    # Filter rows with valid item_id if we only care about menu items, 
    # OR import all and let the DB handle it (though InventoryReport links to MenuItem foreign key).
    
    # Let's inspect data first.
    # If item_id is NaN, it might be an ingredient not directly sold. 
    # For this exercise, we'll focus on rows where item_id is present to link to MenuItem.
    
    # Rename columns for the API/DB
    df_clean = df.rename(columns={
        'quantity': 'current_stock',
        'low_stock_threshold': 'reorder_point'
    })
    
    # Fill missing values
    df_clean['safety_stock'] = df_clean['reorder_point'] * 0.5 # Heuristic
    df_clean['item_id'] = pd.to_numeric(df_clean['item_id'], errors='coerce')
    
    # Drop rows without item_id (ingredients not linked to menu items yet)
    # adjust based on user requirement. User said "add the data set to the database".
    # But our InventoryReport schema enforces item_id as ForeignKey to MenuItem.
    # We will only load items that map to menu items for now.
    
    df_ready = df_clean.dropna(subset=['item_id'])
    df_ready = df_ready[['item_id', 'current_stock', 'reorder_point', 'safety_stock']]
    
    print(f"   Rows to import (linked to Menu Items): {len(df_ready)}")
    
    # Convert to CSV string/buffer
    csv_buffer = df_ready.to_csv(index=False)
    
    # Upload via API
    try:
        files = {'file': ('inventory_data.csv', csv_buffer, 'text/csv')}
        response = requests.post(API_URL, files=files)
        
        if response.status_code == 200:
            print("   ✅ Data loaded successfully!")
            print("   Response:", response.json())
        else:
            print(f"   ❌ Failed to load data. Status: {response.status_code}")
            print(f"   Error: {response.text}")
            
    except Exception as e:
        print(f"   ❌ connection failed: {e}")

if __name__ == "__main__":
    load_data()
