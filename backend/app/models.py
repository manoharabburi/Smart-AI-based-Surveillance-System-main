from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, JSON, Text, Boolean
from .database import Base

class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    camera_id = Column(String(50), default="cam_1", index=True)
    type = Column(String(50), index=True)
    description = Column(Text)
    data = Column(JSON, nullable=True)
    severity = Column(String(20), default="info")
    resolved = Column(Boolean, default=False, index=True)

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "camera_id": self.camera_id,
            "type": self.type,
            "description": self.description,
            "data": self.data or {},
            "severity": self.severity,
            "resolved": self.resolved,
        }
