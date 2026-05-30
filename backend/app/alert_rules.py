import math
import time
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field
from . import config

PERSON_CLASS_NAME = "person"
# Include original plus custom dataset bag labels
BAG_CLASS_NAMES = {"backpack", "handbag", "suitcase", "school bag", "wrong bag"}

@dataclass
class TrackState:
    track_id: str
    cls: str
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    origin_pos: Tuple[float, float] = (0.0, 0.0)
    last_pos: Tuple[float, float] = (0.0, 0.0)
    last_moved_time: float = field(default_factory=time.time)
    loitering_alerted: bool = False
    last_near_person_time: Optional[float] = None  # for bags
    abandoned_alerted: bool = False
    bag_owner_id: Optional[str] = None
    stationary_alerted: bool = False

class RuleEngine:
    def __init__(self, alert_callback):
        self.tracks: Dict[str, TrackState] = {}
        self.alert_callback = alert_callback
        self.last_crowd_alert_time: float = 0.0
        self.crowd_alert_cooldown = 30.0  # seconds

    def update_track(self, track_id: str, cls: str, cx: float, cy: float):
        now = time.time()
        st = self.tracks.get(track_id)
        if st is None:
            st = TrackState(track_id=track_id, cls=cls, origin_pos=(cx, cy), last_pos=(cx, cy))
            if cls in BAG_CLASS_NAMES:
                st.last_near_person_time = now  # assume initially accompanied
            self.tracks[track_id] = st
        # movement detection
        dist = math.hypot(cx - st.last_pos[0], cy - st.last_pos[1])
        if dist > config.STILL_MOVEMENT_PX_RADIUS / 5.0:  # consider small movement resetting still timer
            st.last_moved_time = now
        st.last_pos = (cx, cy)
        st.last_seen = now

    def post_frame(self):
        # cleanup stale tracks not seen for some time
        now = time.time()
        stale_ids = [tid for tid, st in self.tracks.items() if now - st.last_seen > 10]
        for tid in stale_ids:
            del self.tracks[tid]

    def evaluate_loitering(self):
        now = time.time()
        for st in self.tracks.values():
            if st.cls == PERSON_CLASS_NAME and not st.loitering_alerted:
                if (now - st.last_moved_time) > config.LOITERING_SECONDS:
                    st.loitering_alerted = True
                    self.alert_callback(
                        type="Loitering Alert",
                        severity="medium",
                        description=f"Person {st.track_id} loitering for > {config.LOITERING_SECONDS}s",
                        data={"track_id": st.track_id},
                    )

    def evaluate_crowd(self):
        now = time.time()
        person_count = sum(1 for st in self.tracks.values() if st.cls == PERSON_CLASS_NAME)
        if person_count >= config.CROWD_COUNT_THRESHOLD and (now - self.last_crowd_alert_time) > self.crowd_alert_cooldown:
            self.last_crowd_alert_time = now
            self.alert_callback(
                type="Crowd Surge Alert",
                severity="high",
                description=f"Crowd size {person_count} >= threshold {config.CROWD_COUNT_THRESHOLD}",
                data={"count": person_count},
            )

    def evaluate_abandoned_bag(self):
        now = time.time()
        # determine bag proximity to any person each frame in external logic; here just check timer
        for st in self.tracks.values():
            if st.cls in BAG_CLASS_NAMES and not st.abandoned_alerted:
                if st.last_near_person_time is not None:
                    if (now - st.last_near_person_time) > config.ABANDONED_BAG_SECONDS:
                        st.abandoned_alerted = True
                        owner_txt = f" (owner person {st.bag_owner_id})" if st.bag_owner_id else ""
                        self.alert_callback(
                            type="Suspicious: Abandoned Bag",
                            severity="high",
                            description=f"Bag {st.track_id}{owner_txt} unattended > {config.ABANDONED_BAG_SECONDS}s",
                            data={"track_id": st.track_id, "owner_person_id": st.bag_owner_id},
                        )

    def evaluate_stationary_bag(self):
        now = time.time()
        for st in self.tracks.values():
            if st.cls in BAG_CLASS_NAMES and not st.stationary_alerted:
                if (now - st.last_moved_time) > config.BAG_STATIONARY_SECONDS:
                    st.stationary_alerted = True
                    owner_txt = f" (owner person {st.bag_owner_id})" if st.bag_owner_id else ""
                    self.alert_callback(
                        type="Stationary Bag",
                        severity="medium",
                        description=f"Bag {st.track_id}{owner_txt} stationary > {config.BAG_STATIONARY_SECONDS}s",
                        data={"track_id": st.track_id, "owner_person_id": st.bag_owner_id},
                    )

    def mark_bag_near_person(self, bag_track_id: str, person_track_id: str):
        st = self.tracks.get(bag_track_id)
        if st and st.cls in BAG_CLASS_NAMES:
            now = time.time()
            st.last_near_person_time = now
            if st.bag_owner_id is None:
                st.bag_owner_id = person_track_id

    def evaluate_all(self):
        self.evaluate_loitering()
        self.evaluate_crowd()
        self.evaluate_abandoned_bag()
        self.evaluate_stationary_bag()
        self.post_frame()

    def snapshot(self) -> List[dict]:
        return [
            {
                "track_id": st.track_id,
                "cls": st.cls,
                "last_seen": st.last_seen,
                "last_pos": st.last_pos,
                "bag_owner_id": st.bag_owner_id,
            }
            for st in self.tracks.values()
        ]
