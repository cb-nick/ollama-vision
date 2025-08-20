import sqlite3
import json
import uuid
import streamlit as st
from typing import Optional

class DatabaseService:
    """Handles all database operations."""
    
    def __init__(self, db_path: str = "/app/logs/streamlit_db.db"):
        self.db_path = db_path
        self._initialize_database()
    
    def _initialize_database(self):
        """Create database tables if they don't exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS logs (
                    request_id TEXT PRIMARY KEY,
                    image BLOB,
                    response TEXT,
                    response_time REAL,
                    feedback TEXT,
                    types TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
    
    def get_connection(self):
        """Get database connection."""
        return sqlite3.connect(self.db_path)
    
    def generate_request_id(self) -> str:
        """Generate a unique request ID."""
        return str(uuid.uuid4())
    
    def log_response(self, request_id: str, image: str, response: dict, response_time: float):
        """Log analysis response to database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM logs WHERE request_id = ?", (request_id,))
            
            if cursor.fetchone() is not None:
                return  # Skip if already logged
            
            cursor.execute('''
                INSERT INTO logs (request_id, image, response, response_time)
                VALUES (?, ?, ?, ?)
            ''', (request_id, image, json.dumps(response), response_time))
            conn.commit()
    
    def log_feedback(self, request_id: str, feedback: Optional[str]):
        """Log user feedback to database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE logs SET feedback = ? WHERE request_id = ?",
                (feedback, request_id)
            )
            conn.commit()
    
    def log_types(self, request_id: str, types: dict):
        """Log saved types to database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE logs SET types = ? WHERE request_id = ?",
                (json.dumps(types), request_id)
            )
            conn.commit()
