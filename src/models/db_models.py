from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from src.database import Base

class Restaurant(Base):
    __tablename__ = "restaurants"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    location = Column(String)
    
    # Relationships
    orders = relationship("Order", back_populates="restaurant")
    menu_items = relationship("MenuItem", back_populates="restaurant")

class MenuItem(Base):
    __tablename__ = "menu_items"

    id = Column(Integer, primary_key=True, index=True)
    restaurant_id = Column(Integer, ForeignKey("restaurants.id"))
    name = Column(String, index=True)
    price = Column(Float)
    category = Column(String) # Star, Plowhorse, Puzzle, Dog
    description = Column(String, nullable=True)
    
    # Analysis metrics (can be updated by ML model)
    popularity_score = Column(Float, default=0.0)
    profitability_score = Column(Float, default=0.0)
    
    # Relationships
    restaurant = relationship("Restaurant", back_populates="menu_items")
    order_items = relationship("OrderItem", back_populates="menu_item")

class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, index=True)
    restaurant_id = Column(Integer, ForeignKey("restaurants.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    total_amount = Column(Float)
    
    # Relationships
    restaurant = relationship("Restaurant", back_populates="orders")
    items = relationship("OrderItem", back_populates="order")

class OrderItem(Base):
    __tablename__ = "order_items"

    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(Integer, ForeignKey("orders.id"))
    menu_item_id = Column(Integer, ForeignKey("menu_items.id"))
    quantity = Column(Integer)
    price_at_time = Column(Float) # Price might change over time
    
    # Relationships
    order = relationship("Order", back_populates="items")
    menu_item = relationship("MenuItem", back_populates="order_items")

class InventoryReport(Base):
    __tablename__ = "inventory_reports"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, default=datetime.utcnow)
    item_id = Column(Integer, ForeignKey("menu_items.id")) # Linking to menu items for simplicity, assuming 1:1 or managed mapping
    current_stock = Column(Float)
    reorder_point = Column(Float)
    safety_stock = Column(Float)
    
    # Relationship
    item = relationship("MenuItem")

