"""
perception/temporal_tracker.py
===============================
Temporal Perception Engine — processes video-rate frames and maintains
persistent object tracking with velocity estimation and change detection.

This is what separates a "screenshot tool" from a "world model."
Instead of processing one frame, it processes N frames per second and:
  1. Maintains persistent track IDs across frames
  2. Estimates velocity and acceleration from position history
  3. Detects object state changes (appeared, disappeared, moved, stopped)
  4. Handles occlusion (object hidden for N frames → still remembered)

Usage:
    tracker = TemporalTracker(pipeline, fusion)
    
    # Process frames from a video or live camera
    for rgb, depth in video_stream:
        state = tracker.process_frame(rgb, depth, intrinsics, extrinsics)
        print(state.events)  # ['cup MOVED 5cm', 'ball APPEARED']
"""
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from perception.scene_graph import SceneGraph, SceneObject


@dataclass
class ObjectState:
    """Temporal state of a tracked object."""
    track_id: int
    label: str
    position: np.ndarray
    velocity: np.ndarray         # m/s
    acceleration: np.ndarray     # m/s²
    is_moving: bool
    is_occluded: bool
    frames_tracked: int
    first_seen_frame: int
    last_seen_frame: int


@dataclass
class ChangeEvent:
    """A detected state change in the world."""
    event_type: str              # APPEARED, DISAPPEARED, MOVED, STOPPED, COLLISION
    track_id: int
    label: str
    details: str                 # human-readable description
    frame: int
    position: Optional[np.ndarray] = None
    displacement_m: float = 0.0


@dataclass
class WorldState:
    """Complete temporal world state at a given frame."""
    frame_number: int
    timestamp: float
    objects: List[ObjectState]
    events: List[ChangeEvent]
    scene_graph: SceneGraph
    fps: float


class TemporalTracker:
    """
    Processes camera frames over time and maintains a living world model.
    
    The tracker wraps the perception pipeline and adds temporal reasoning:
    - Position smoothing (Kalman-like exponential filter)
    - Velocity estimation from consecutive positions
    - Change detection with configurable thresholds
    - Occlusion handling with configurable patience
    """

    # Thresholds
    MOVEMENT_THRESHOLD_M = 0.01       # 1cm — below this, object is "stationary"
    SIGNIFICANT_MOVE_M = 0.03         # 3cm — triggers a MOVED event
    VELOCITY_SMOOTHING = 0.3          # exponential smoothing factor
    OCCLUSION_PATIENCE_FRAMES = 30    # keep object in memory for 30 frames after disappearing
    
    def __init__(self, target_fps: float = 2.0):
        """
        Args:
            target_fps: desired processing rate (frames per second)
        """
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        
        self.frame_count = 0
        self.start_time = None
        self.last_frame_time = 0.0
        
        # Temporal state per tracked object
        self._object_states: Dict[int, ObjectState] = {}
        self._previous_positions: Dict[int, np.ndarray] = {}
        self._velocity_buffer: Dict[int, List[np.ndarray]] = {}
        
        # Event log
        self._all_events: List[ChangeEvent] = []
        
        # The living scene graph
        self.scene_graph = SceneGraph()
    
    def process_frame(self, scene_graph: SceneGraph) -> WorldState:
        """
        Process a new perception result and update temporal state.
        
        This should be called after running the perception pipeline
        on each frame. It takes the pipeline's output SceneGraph and
        adds temporal reasoning on top.
        
        Args:
            scene_graph: the SceneGraph output from PerceptionPipeline.perceive()
            
        Returns:
            WorldState with tracked objects, events, and timing info
        """
        now = time.time()
        if self.start_time is None:
            self.start_time = now
        
        self.frame_count += 1
        dt = now - self.last_frame_time if self.last_frame_time > 0 else self.frame_interval
        self.last_frame_time = now
        
        # Update our internal scene graph
        self.scene_graph = scene_graph
        
        # Detect changes and update temporal states
        current_events = []
        current_track_ids = set()
        
        for obj in scene_graph.get_all_objects():
            tid = obj.track_id
            current_track_ids.add(tid)
            pos = np.array(obj.position_world)
            
            if tid in self._object_states:
                # Existing object — update tracking
                state = self._object_states[tid]
                old_pos = state.position
                
                # Velocity estimation
                if dt > 0:
                    raw_velocity = (pos - old_pos) / dt
                    # Exponential smoothing
                    state.velocity = (
                        self.VELOCITY_SMOOTHING * raw_velocity +
                        (1 - self.VELOCITY_SMOOTHING) * state.velocity
                    )
                    # Acceleration
                    if tid in self._velocity_buffer and len(self._velocity_buffer[tid]) > 0:
                        prev_vel = self._velocity_buffer[tid][-1]
                        state.acceleration = (state.velocity - prev_vel) / dt
                    
                    # Store velocity history
                    if tid not in self._velocity_buffer:
                        self._velocity_buffer[tid] = []
                    self._velocity_buffer[tid].append(state.velocity.copy())
                    if len(self._velocity_buffer[tid]) > 10:
                        self._velocity_buffer[tid] = self._velocity_buffer[tid][-10:]
                
                # Movement detection
                displacement = np.linalg.norm(pos - old_pos)
                speed = np.linalg.norm(state.velocity)
                
                was_moving = state.is_moving
                state.is_moving = speed > self.MOVEMENT_THRESHOLD_M
                
                # Change events
                if displacement > self.SIGNIFICANT_MOVE_M:
                    event = ChangeEvent(
                        event_type="MOVED",
                        track_id=tid,
                        label=obj.label,
                        details=f"{obj.label} moved {displacement*100:.1f}cm",
                        frame=self.frame_count,
                        position=pos,
                        displacement_m=displacement,
                    )
                    current_events.append(event)
                
                if was_moving and not state.is_moving:
                    event = ChangeEvent(
                        event_type="STOPPED",
                        track_id=tid,
                        label=obj.label,
                        details=f"{obj.label} stopped moving",
                        frame=self.frame_count,
                        position=pos,
                    )
                    current_events.append(event)
                
                # Update position and frame info
                state.position = pos
                state.label = obj.label
                state.is_occluded = False
                state.frames_tracked += 1
                state.last_seen_frame = self.frame_count
                
            else:
                # New object — first appearance
                state = ObjectState(
                    track_id=tid,
                    label=obj.label,
                    position=pos,
                    velocity=np.zeros(3),
                    acceleration=np.zeros(3),
                    is_moving=False,
                    is_occluded=False,
                    frames_tracked=1,
                    first_seen_frame=self.frame_count,
                    last_seen_frame=self.frame_count,
                )
                self._object_states[tid] = state
                
                event = ChangeEvent(
                    event_type="APPEARED",
                    track_id=tid,
                    label=obj.label,
                    details=f"{obj.label} appeared at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})",
                    frame=self.frame_count,
                    position=pos,
                )
                current_events.append(event)
            
            self._previous_positions[tid] = pos.copy()
        
        # Check for disappeared objects
        for tid, state in list(self._object_states.items()):
            if tid not in current_track_ids:
                frames_missing = self.frame_count - state.last_seen_frame
                
                if not state.is_occluded and frames_missing == 1:
                    # Just went occluded
                    state.is_occluded = True
                    event = ChangeEvent(
                        event_type="OCCLUDED",
                        track_id=tid,
                        label=state.label,
                        details=f"{state.label} became occluded",
                        frame=self.frame_count,
                        position=state.position,
                    )
                    current_events.append(event)
                
                elif frames_missing > self.OCCLUSION_PATIENCE_FRAMES:
                    # Permanently gone
                    event = ChangeEvent(
                        event_type="DISAPPEARED",
                        track_id=tid,
                        label=state.label,
                        details=f"{state.label} disappeared after {frames_missing} frames",
                        frame=self.frame_count,
                        position=state.position,
                    )
                    current_events.append(event)
                    del self._object_states[tid]
        
        # Store events
        self._all_events.extend(current_events)
        
        # Calculate FPS
        elapsed = now - self.start_time if self.start_time else 1.0
        fps = self.frame_count / max(elapsed, 0.001)
        
        # Build world state
        world_state = WorldState(
            frame_number=self.frame_count,
            timestamp=now,
            objects=list(self._object_states.values()),
            events=current_events,
            scene_graph=scene_graph,
            fps=fps,
        )
        
        return world_state
    
    def get_object_trajectory(self, track_id: int) -> Optional[List[np.ndarray]]:
        """Get the full position history of a tracked object."""
        obj = self.scene_graph.get_object_by_track_id(track_id)
        if obj and obj.position_history:
            return [np.array(p) for p in obj.position_history]
        return None
    
    def get_moving_objects(self) -> List[ObjectState]:
        """Return all objects currently in motion."""
        return [s for s in self._object_states.values() if s.is_moving]
    
    def get_stable_objects(self) -> List[ObjectState]:
        """Return all objects that are stationary and fully tracked."""
        return [s for s in self._object_states.values()
                if not s.is_moving and not s.is_occluded]
    
    def get_event_log(self, last_n: int = None) -> List[ChangeEvent]:
        """Get the event log (optionally last N events)."""
        if last_n:
            return self._all_events[-last_n:]
        return self._all_events
    
    def print_state(self):
        """Print current temporal tracking state."""
        print(f"\n{'='*60}")
        print(f"  TEMPORAL TRACKER — Frame {self.frame_count}")
        print(f"{'='*60}")
        
        for state in self._object_states.values():
            pos = state.position
            vel = np.linalg.norm(state.velocity)
            status = "🟢 MOVING" if state.is_moving else ("🔴 OCCLUDED" if state.is_occluded else "⚪ STATIC")
            print(f"\n  [T{state.track_id}] {state.label} — {status}")
            print(f"    Position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
            print(f"    Speed: {vel*100:.1f} cm/s")
            print(f"    Tracked for: {state.frames_tracked} frames")
        
        if self._all_events:
            recent = self._all_events[-5:]
            print(f"\n  Recent Events:")
            for e in recent:
                print(f"    [{e.event_type}] {e.details} (frame {e.frame})")
        
        print(f"\n{'='*60}\n")

