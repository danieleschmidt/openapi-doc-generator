#!/usr/bin/env python3
"""Simple example application for testing documentation generation."""

class SimpleAPI:
    """A simple API class for testing."""
    
    def get_users(self):
        """Get all users.
        
        Returns:
            list: List of user dictionaries
        """
        return [{"id": 1, "name": "John"}, {"id": 2, "name": "Jane"}]
    
    def create_user(self, user_data: dict):
        """Create a new user.
        
        Args:
            user_data (dict): User information including name and email
            
        Returns:
            dict: Created user with assigned ID
        """
        return {"id": 3, "name": user_data.get("name"), "email": user_data.get("email")}

if __name__ == "__main__":
    api = SimpleAPI()
    print("Simple API initialized")
    print("Users:", api.get_users())
    print("Created:", api.create_user({"name": "Test", "email": "test@example.com"}))