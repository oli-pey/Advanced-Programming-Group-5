"""
Script to create example data for the prediction database
"""

from DB.database import SessionLocal, PredictionEntry
from datetime import datetime, timedelta
import random

def create_example_data():
    """Create and insert example prediction entries into the database"""
    db = SessionLocal()
    
    try:
        # Create 30 example entries with random images and digit predictions
        digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        
        for i in range(30):
            # Create a random 28x28 image (as bytes)
            # Using random bytes to simulate handwritten digit images
            image_bytes = bytes([random.randint(0, 255) for _ in range(28 * 28)])
            
            # Randomly select a digit
            predicted_digit = digits[i % 10]
            
            # Create entries with different timestamps (spread over the last 7 days)
            days_offset = i // 5
            hours_offset = (i % 5) * 2
            created_at = datetime.utcnow() - timedelta(days=days_offset, hours=hours_offset)
            
            # Create the entry
            entry = PredictionEntry(
                image=image_bytes,
                text=predicted_digit,
                created_at=created_at
            )
            
            db.add(entry)
        
        # Commit all entries
        db.commit()
        print("✓ Successfully created 30 example prediction entries")
        print("Database location: ./mydata.db")
        
        # Display summary
        count = db.query(PredictionEntry).count()
        print(f"Total entries in database: {count}")
        
    except Exception as e:
        db.rollback()
        print(f"✗ Error creating example data: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    create_example_data()
