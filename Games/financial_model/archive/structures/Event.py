from datetime import datetime


class Event:
    def __init__(self, timestamp: datetime, description: dict):
        self.timestamp = timestamp
        self.description = description