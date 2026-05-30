from datetime import datetime
from typing import Optional, List, Any
from pydantic import BaseModel

class AlertBase(BaseModel):
    camera_id: str
    type: str
    description: str
    severity: str = "info"
    data: Optional[Any] = None
    resolved: bool = False

class AlertCreate(AlertBase):
    pass

class AlertOut(AlertBase):
    id: int
    timestamp: datetime

    class Config:
        orm_mode = True

class AlertsList(BaseModel):
    items: List[AlertOut]
    total: int
