"""
Neo4j Connection and Data Models
Handles database operations for detection records
"""
from neo4j import GraphDatabase
from typing import List, Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Neo4jConnection:
    """Neo4j database connection and operations"""
    
    def __init__(self):
        """Initialize Neo4j connection using environment variables"""
        self.uri = os.getenv("NEO4J_URI", "neo4j+s://your-instance.databases.neo4j.io")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "your-password")
        
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # Test connection
            self.driver.verify_connectivity()
            print("Successfully connected to Neo4j Aura")
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            print("Please check your NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD environment variables")
            raise
    
    def close(self):
        """Close the database connection"""
        if self.driver:
            self.driver.close()
    
    def save_detection(self, image_name: str, hijab_count: int, timestamp: str) -> bool:
        """
        Save a detection record to Neo4j
        
        Args:
            image_name: Name of the processed image
            hijab_count: Number of hijabs detected
            timestamp: ISO format timestamp
            
        Returns:
            True if successful
        """
        query = """
        CREATE (d:Detection {
            image_name: $image_name,
            hijab_count: $hijab_count,
            timestamp: $timestamp
        })
        RETURN d
        """
        
        with self.driver.session() as session:
            result = session.run(
                query,
                image_name=image_name,
                hijab_count=hijab_count,
                timestamp=timestamp
            )
            return result.single() is not None
    
    def get_all_detections(self) -> List[dict]:
        """
        Retrieve all detection records from Neo4j
        
        Returns:
            List of detection records
        """
        query = """
        MATCH (d:Detection)
        RETURN d.image_name as image_name, 
               d.hijab_count as hijab_count, 
               d.timestamp as timestamp
        ORDER BY d.timestamp DESC
        """
        
        with self.driver.session() as session:
            result = session.run(query)
            records = []
            for record in result:
                records.append({
                    "image_name": record["image_name"],
                    "hijab_count": record["hijab_count"],
                    "timestamp": record["timestamp"]
                })
            return records
    
    def get_detection_by_image(self, image_name: str) -> Optional[dict]:
        """
        Retrieve a specific detection record by image name
        
        Args:
            image_name: Name of the image
            
        Returns:
            Detection record or None if not found
        """
        query = """
        MATCH (d:Detection {image_name: $image_name})
        RETURN d.image_name as image_name, 
               d.hijab_count as hijab_count, 
               d.timestamp as timestamp
        LIMIT 1
        """
        
        with self.driver.session() as session:
            result = session.run(query, image_name=image_name)
            record = result.single()
            if record:
                return {
                    "image_name": record["image_name"],
                    "hijab_count": record["hijab_count"],
                    "timestamp": record["timestamp"]
                }
            return None
    
    def delete_detection(self, image_name: str) -> bool:
        """
        Delete a detection record from Neo4j
        
        Args:
            image_name: Name of the image
            
        Returns:
            True if deleted, False if not found
        """
        query = """
        MATCH (d:Detection {image_name: $image_name})
        DELETE d
        RETURN count(d) as deleted_count
        """
        
        with self.driver.session() as session:
            result = session.run(query, image_name=image_name)
            record = result.single()
            return record["deleted_count"] > 0 if record else False
    
    def get_statistics(self) -> dict:
        """
        Get statistics about all detections
        
        Returns:
            Dictionary with total detections and total hijabs counted
        """
        query = """
        MATCH (d:Detection)
        RETURN count(d) as total_detections,
               sum(d.hijab_count) as total_hijabs,
               avg(d.hijab_count) as avg_hijabs_per_image
        """
        
        with self.driver.session() as session:
            result = session.run(query)
            record = result.single()
            if record:
                return {
                    "total_detections": record["total_detections"],
                    "total_hijabs": record["total_hijabs"],
                    "avg_hijabs_per_image": float(record["avg_hijabs_per_image"]) if record["avg_hijabs_per_image"] else 0.0
                }
            return {
                "total_detections": 0,
                "total_hijabs": 0,
                "avg_hijabs_per_image": 0.0
            }
