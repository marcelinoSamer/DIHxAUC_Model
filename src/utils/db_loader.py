import pandas as pd
from sqlalchemy.orm import Session
from src.models.db_models import InventoryReport, MenuItem
from src.database import SessionLocal
from datetime import datetime

def load_inventory_from_csv(file_path: str, db: Session):
    """
    Load inventory data from a CSV file into the database.
    
    Expected CSV columns:
    - item_id (maps to menu_item_id)
    - current_stock
    - reorder_point
    - safety_stock
    """
    try:
        df = pd.read_csv(file_path)
        
        # Validate columns
        required_cols = ['item_id', 'current_stock']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV must contain columns: {required_cols}")
        
        reports = []
        for _, row in df.iterrows():
            report = InventoryReport(
                date=datetime.utcnow(),
                item_id=int(row['item_id']),
                current_stock=float(row['current_stock']),
                reorder_point=float(row.get('reorder_point', 0)),
                safety_stock=float(row.get('safety_stock', 0))
            )
            reports.append(report)
        
        db.add_all(reports)
        db.commit()
        return len(reports)
        
    except Exception as e:
        db.rollback()
        raise e
