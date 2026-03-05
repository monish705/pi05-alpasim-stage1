"""
perception/scene_graph.py
=========================
3D Scene Graph v2 — the central data structure consumed by all downstream components:
 - Cognitive Core (task decomposition)
 - Sparse Sim (scene building)
 - Trajectory Optimiser (grasp targets)
 - Mission Monitor (success verification)
 - System ID (initial physics estimates)

v2 improvements:
 - Persistent track IDs across frames (object permanence)
 - Temporal tracking: velocity, position history, staleness
 - Richer spatial relationships (LEFT_OF, BEHIND, ABOVE, INSIDE, etc.)
 - Confidence decay and stale object pruning
 - Object state change detection
"""
import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Tuple


@dataclass
class SceneObject:
    """A single object in the scene graph."""
    id: str                                 # detection-level UUID (may change per frame)
    label: str                              # "red ceramic mug"
    position_world: List[float]             # [x, y, z] in world frame
    orientation: List[float] = field(default_factory=lambda: [1, 0, 0, 0])  # quat

    # Geometry
    collision_primitive: str = "box"        # "cylinder"|"box"|"sphere"|"capsule"
    dimensions_m: Dict[str, float] = field(
        default_factory=lambda: {"width": 0.05, "height": 0.05, "depth": 0.05}
    )

    # Physics
    mass_kg: float = 0.2
    friction: float = 0.4
    material: str = "unknown"

    # Semantics
    is_graspable: bool = True
    is_deformable: bool = False

    # Tracking (v2)
    track_id: int = -1                      # persistent ID across frames (-1 = unassigned)
    confidence: float = 0.0                 # unified confidence [0, 1]
    last_seen_frame: int = 0
    frames_since_seen: int = 0              # increments each frame if not re-detected
    velocity_estimate: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    position_history: List[List[float]] = field(default_factory=list)

    # Relationships
    relationships: List[str] = field(default_factory=list)

    # Mesh data stored separately (not serialised to JSON)
    _mesh_vertices: Optional[np.ndarray] = field(default=None, repr=False)
    _mesh_faces: Optional[np.ndarray] = field(default=None, repr=False)


class SceneGraph:
    """
    Persistent 3D scene representation with temporal tracking.

    Objects survive across frames via stable track IDs.
    New detections are matched to existing objects using label similarity + position distance.
    Unmatched detections create new tracks. Stale tracks are pruned.
    """

    # Confidence decay per frame for unseen objects
    CONFIDENCE_DECAY = 0.05
    # Max frames before pruning a stale object
    MAX_STALE_FRAMES = 15
    # Max positions to keep in history
    MAX_HISTORY = 20

    def __init__(self):
        self.objects: Dict[str, SceneObject] = {}  # keyed by track_id (as str)
        self.frame_count: int = 0
        self._next_track_id: int = 1

    def _assign_track_id(self) -> int:
        """Get the next available persistent track ID."""
        tid = self._next_track_id
        self._next_track_id += 1
        return tid

    # ---- Queries ----

    def add_object(self, obj: SceneObject):
        """Add or update an object in the graph."""
        if obj.track_id < 0:
            obj.track_id = self._assign_track_id()
        self.objects[str(obj.track_id)] = obj

    def query(self, label: str) -> Optional[SceneObject]:
        """
        Fuzzy-match query by label.
        Returns the best matching object or None.
        """
        label_lower = label.lower()
        best = None
        best_score = 0

        for obj in self.objects.values():
            obj_label = obj.label.lower()
            # Exact match
            if label_lower == obj_label:
                return obj
            # Substring match
            if label_lower in obj_label or obj_label in label_lower:
                score = len(label_lower) / max(len(obj_label), 1)
                if score > best_score:
                    best_score = score
                    best = obj
            # Keyword match
            for word in label_lower.split():
                if word in obj_label and len(word) > 2:
                    score = 0.5
                    if score > best_score:
                        best_score = score
                        best = obj

        return best

    def query_all(self, label: str) -> List[SceneObject]:
        """Return all objects matching the label."""
        label_lower = label.lower()
        matches = []
        for obj in self.objects.values():
            obj_label = obj.label.lower()
            if label_lower in obj_label or obj_label in label_lower:
                matches.append(obj)
            else:
                for word in label_lower.split():
                    if word in obj_label and len(word) > 2:
                        matches.append(obj)
                        break
        return matches

    def get_object_by_track_id(self, track_id: int) -> Optional[SceneObject]:
        """Query by persistent track ID."""
        return self.objects.get(str(track_id))

    def get_all_objects(self) -> List[SceneObject]:
        """Return all objects in the scene."""
        return list(self.objects.values())

    def get_graspable_objects(self) -> List[SceneObject]:
        """Return only graspable objects."""
        return [o for o in self.objects.values() if o.is_graspable]

    def get_active_objects(self, max_stale: int = 5) -> List[SceneObject]:
        """Return objects that have been seen recently."""
        return [o for o in self.objects.values() if o.frames_since_seen <= max_stale]

    def get_task_relevant_objects(self, target_label: str,
                                  radius_m: float = 0.3) -> List[SceneObject]:
        """
        Get objects relevant to a task involving target_label.
        Returns the target + any objects within radius_m (potential obstacles).
        """
        target = self.query(target_label)
        if target is None:
            return []

        relevant = [target]
        target_pos = np.array(target.position_world)

        for obj in self.objects.values():
            if obj.track_id == target.track_id:
                continue
            dist = np.linalg.norm(np.array(obj.position_world) - target_pos)
            if dist < radius_m:
                relevant.append(obj)

        return relevant

    # ---- Temporal Update ----

    def update(self, new_objects: List[SceneObject],
               merge_distance_m: float = 0.12,
               label_weight: float = 0.4,
               position_weight: float = 0.6):
        """
        Merge new detections into the existing graph with temporal tracking.

        Uses weighted cost matrix (label similarity + position distance)
        for matching. Unmatched new objects get fresh track IDs.
        Unmatched existing objects age and eventually get pruned.
        """
        self.frame_count += 1

        existing = list(self.objects.values())

        if not existing:
            # First frame — assign all as new tracks
            for obj in new_objects:
                obj.track_id = self._assign_track_id()
                obj.last_seen_frame = self.frame_count
                obj.frames_since_seen = 0
                obj.position_history = [list(obj.position_world)]
                self.objects[str(obj.track_id)] = obj
            self._compute_relationships()
            return

        if not new_objects:
            # No detections — age all existing
            self._age_all()
            self._prune_stale()
            return

        # Build cost matrix: existing (rows) x new (cols)
        n_existing = len(existing)
        n_new = len(new_objects)
        cost_matrix = np.full((n_existing, n_new), float('inf'))

        for i, ex in enumerate(existing):
            for j, nw in enumerate(new_objects):
                pos_dist = np.linalg.norm(
                    np.array(ex.position_world) - np.array(nw.position_world)
                )
                if pos_dist > merge_distance_m * 3:
                    # Skip obviously impossible matches
                    continue

                label_sim = self._label_similarity(ex.label, nw.label)
                cost = position_weight * pos_dist + label_weight * (1.0 - label_sim)
                cost_matrix[i, j] = cost

        # Hungarian matching (greedy fallback if scipy not available)
        matches, unmatched_existing, unmatched_new = self._match(
            cost_matrix, max_cost=merge_distance_m * 2
        )

        # Update matched objects
        for ex_idx, new_idx in matches:
            ex = existing[ex_idx]
            nw = new_objects[new_idx]
            self._merge_object(ex, nw)

        # Age unmatched existing objects
        for ex_idx in unmatched_existing:
            ex = existing[ex_idx]
            ex.frames_since_seen += 1
            ex.confidence = max(0.0, ex.confidence - self.CONFIDENCE_DECAY)

        # Create new tracks for unmatched detections
        for new_idx in unmatched_new:
            nw = new_objects[new_idx]
            nw.track_id = self._assign_track_id()
            nw.last_seen_frame = self.frame_count
            nw.frames_since_seen = 0
            nw.position_history = [list(nw.position_world)]
            self.objects[str(nw.track_id)] = nw

        # Prune stale objects
        self._prune_stale()

        # Recompute spatial relationships
        self._compute_relationships()

    def _merge_object(self, existing: SceneObject, new: SceneObject):
        """Merge a new detection into an existing tracked object."""
        old_pos = np.array(existing.position_world)
        new_pos = np.array(new.position_world)

        # Velocity estimate
        if existing.last_seen_frame > 0:
            dt = self.frame_count - existing.last_seen_frame
            if dt > 0:
                existing.velocity_estimate = ((new_pos - old_pos) / dt).tolist()

        # Update position
        existing.position_world = new.position_world

        # Position history
        existing.position_history.append(list(new.position_world))
        if len(existing.position_history) > self.MAX_HISTORY:
            existing.position_history = existing.position_history[-self.MAX_HISTORY:]

        # Update other fields
        existing.label = new.label
        existing.confidence = new.confidence
        existing.last_seen_frame = self.frame_count
        existing.frames_since_seen = 0

        if new.mass_kg > 0:
            existing.mass_kg = new.mass_kg
        if new.friction > 0:
            existing.friction = new.friction
        if new.material != "unknown":
            existing.material = new.material
        if new.collision_primitive != "box" or existing.collision_primitive == "box":
            existing.collision_primitive = new.collision_primitive
        if new.dimensions_m:
            existing.dimensions_m = new.dimensions_m
        existing.is_graspable = new.is_graspable
        existing.is_deformable = new.is_deformable
        existing.orientation = new.orientation

        # Keep mesh if new one is available
        if new._mesh_vertices is not None:
            existing._mesh_vertices = new._mesh_vertices
            existing._mesh_faces = new._mesh_faces

    def _age_all(self):
        """Age all objects by one frame."""
        for obj in self.objects.values():
            obj.frames_since_seen += 1
            obj.confidence = max(0.0, obj.confidence - self.CONFIDENCE_DECAY)

    def _prune_stale(self):
        """Remove objects that haven't been seen for too long."""
        to_remove = []
        for key, obj in self.objects.items():
            if obj.frames_since_seen > self.MAX_STALE_FRAMES:
                to_remove.append(key)
        for key in to_remove:
            obj = self.objects.pop(key)
            print(f"[SceneGraph] Pruned stale object: '{obj.label}' "
                  f"(track={obj.track_id}, unseen for {obj.frames_since_seen} frames)")

    # ---- Matching ----

    def _match(self, cost_matrix: np.ndarray,
               max_cost: float) -> Tuple[list, list, list]:
        """
        Match existing objects to new detections.

        Uses scipy Hungarian if available, otherwise greedy matching.

        Returns:
            (matches, unmatched_existing_indices, unmatched_new_indices)
        """
        n_existing, n_new = cost_matrix.shape

        try:
            from scipy.optimize import linear_sum_assignment
            # Pad cost matrix with max_cost entries so unmatched are possible
            padded = np.full(
                (max(n_existing, n_new), max(n_existing, n_new)),
                max_cost + 1.0
            )
            padded[:n_existing, :n_new] = cost_matrix
            row_ind, col_ind = linear_sum_assignment(padded)

            matches = []
            matched_existing = set()
            matched_new = set()
            for r, c in zip(row_ind, col_ind):
                if r < n_existing and c < n_new and cost_matrix[r, c] < max_cost:
                    matches.append((r, c))
                    matched_existing.add(r)
                    matched_new.add(c)

            unmatched_existing = [i for i in range(n_existing) if i not in matched_existing]
            unmatched_new = [j for j in range(n_new) if j not in matched_new]
            return matches, unmatched_existing, unmatched_new

        except ImportError:
            # Greedy fallback
            return self._greedy_match(cost_matrix, max_cost)

    def _greedy_match(self, cost_matrix: np.ndarray,
                      max_cost: float) -> Tuple[list, list, list]:
        """Greedy nearest-neighbor matching (fallback)."""
        n_existing, n_new = cost_matrix.shape
        matches = []
        matched_existing = set()
        matched_new = set()

        # Sort all pairs by cost
        pairs = []
        for i in range(n_existing):
            for j in range(n_new):
                if cost_matrix[i, j] < max_cost:
                    pairs.append((cost_matrix[i, j], i, j))
        pairs.sort()

        for cost, i, j in pairs:
            if i not in matched_existing and j not in matched_new:
                matches.append((i, j))
                matched_existing.add(i)
                matched_new.add(j)

        unmatched_existing = [i for i in range(n_existing) if i not in matched_existing]
        unmatched_new = [j for j in range(n_new) if j not in matched_new]
        return matches, unmatched_existing, unmatched_new

    @staticmethod
    def _label_similarity(a: str, b: str) -> float:
        """Simple label similarity: word overlap ratio."""
        a_words = set(a.lower().split())
        b_words = set(b.lower().split())
        if not a_words or not b_words:
            return 0.0
        overlap = len(a_words & b_words)
        total = max(len(a_words), len(b_words))
        return overlap / total

    # ---- Spatial Relationships ----

    def _compute_relationships(self):
        """Auto-compute spatial relationships between all objects."""
        objs = list(self.objects.values())
        for obj in objs:
            obj.relationships = []

        for i, a in enumerate(objs):
            for j, b in enumerate(objs):
                if i >= j:
                    continue

                a_pos = np.array(a.position_world)
                b_pos = np.array(b.position_world)

                dx = b_pos[0] - a_pos[0]  # +x = forward / in front
                dy = b_pos[1] - a_pos[1]  # +y = left
                dz = b_pos[2] - a_pos[2]  # +z = up

                horiz_dist = np.linalg.norm(a_pos[:2] - b_pos[:2])
                vert_diff = dz

                # ON: a is above b and horizontally close
                if vert_diff < -0.02 and horiz_dist < 0.2:
                    a.relationships.append(f"ON {b.label}")
                elif vert_diff > 0.02 and horiz_dist < 0.2:
                    b.relationships.append(f"ON {a.label}")

                # ABOVE / BELOW
                if vert_diff > 0.05 and horiz_dist < 0.3:
                    b.relationships.append(f"ABOVE {a.label}")
                    a.relationships.append(f"BELOW {b.label}")
                elif vert_diff < -0.05 and horiz_dist < 0.3:
                    a.relationships.append(f"ABOVE {b.label}")
                    b.relationships.append(f"BELOW {a.label}")

                # NEXT_TO: close horizontally, similar height
                if horiz_dist < 0.15 and abs(vert_diff) < 0.05:
                    a.relationships.append(f"NEXT_TO {b.label}")
                    b.relationships.append(f"NEXT_TO {a.label}")

                # LEFT_OF / RIGHT_OF (relative to world Y axis)
                if horiz_dist < 0.3 and abs(vert_diff) < 0.1:
                    if dy > 0.05:
                        a.relationships.append(f"RIGHT_OF {b.label}")
                        b.relationships.append(f"LEFT_OF {a.label}")
                    elif dy < -0.05:
                        a.relationships.append(f"LEFT_OF {b.label}")
                        b.relationships.append(f"RIGHT_OF {a.label}")

                # IN_FRONT_OF / BEHIND (relative to world X axis)
                if horiz_dist < 0.3 and abs(vert_diff) < 0.1:
                    if dx > 0.05:
                        a.relationships.append(f"IN_FRONT_OF {b.label}")
                        b.relationships.append(f"BEHIND {a.label}")
                    elif dx < -0.05:
                        a.relationships.append(f"BEHIND {b.label}")
                        b.relationships.append(f"IN_FRONT_OF {a.label}")

    # ---- State Change Detection ----

    def get_moved_objects(self, threshold_m: float = 0.03) -> List[SceneObject]:
        """Return objects whose position has changed significantly since last frame."""
        moved = []
        for obj in self.objects.values():
            vel = np.linalg.norm(obj.velocity_estimate)
            if vel > threshold_m:
                moved.append(obj)
        return moved

    def get_object_displacement(self, track_id: int) -> Optional[float]:
        """Get total displacement of an object from its first-seen position."""
        obj = self.get_object_by_track_id(track_id)
        if obj is None or len(obj.position_history) < 2:
            return None
        first = np.array(obj.position_history[0])
        last = np.array(obj.position_history[-1])
        return float(np.linalg.norm(last - first))

    # ---- Display ----

    def print_all(self):
        """Pretty-print the entire scene graph."""
        print(f"\n{'='*60}")
        print(f"  SCENE GRAPH — {len(self.objects)} objects (frame {self.frame_count})")
        print(f"{'='*60}")
        for obj in self.objects.values():
            pos = obj.position_world
            stale = f" [STALE {obj.frames_since_seen}f]" if obj.frames_since_seen > 0 else ""
            print(f"\n  [T{obj.track_id}] {obj.label}{stale}")
            print(f"    Position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
            print(f"    Shape: {obj.collision_primitive} "
                  f"({obj.dimensions_m.get('width',0)*100:.0f}×"
                  f"{obj.dimensions_m.get('height',0)*100:.0f}×"
                  f"{obj.dimensions_m.get('depth',0)*100:.0f} cm)")
            print(f"    Physics: mass={obj.mass_kg:.2f}kg, "
                  f"friction={obj.friction:.2f}, material={obj.material}")
            print(f"    Graspable: {obj.is_graspable}, "
                  f"Confidence: {obj.confidence:.2f}")
            vel = np.linalg.norm(obj.velocity_estimate)
            if vel > 0.001:
                print(f"    Velocity: {vel:.3f} m/frame")
            if obj.relationships:
                print(f"    Relations: {', '.join(obj.relationships)}")
        print(f"\n{'='*60}\n")

    # ---- Serialisation ----

    def to_json(self) -> str:
        """Serialise scene graph to JSON."""
        data = {
            "frame_count": self.frame_count,
            "objects": {}
        }
        for key, obj in self.objects.items():
            d = {
                "track_id": obj.track_id,
                "id": obj.id,
                "label": obj.label,
                "position_world": obj.position_world,
                "orientation": obj.orientation,
                "collision_primitive": obj.collision_primitive,
                "dimensions_m": obj.dimensions_m,
                "mass_kg": obj.mass_kg,
                "friction": obj.friction,
                "material": obj.material,
                "is_graspable": obj.is_graspable,
                "is_deformable": obj.is_deformable,
                "confidence": obj.confidence,
                "last_seen_frame": obj.last_seen_frame,
                "frames_since_seen": obj.frames_since_seen,
                "velocity_estimate": obj.velocity_estimate,
                "relationships": obj.relationships,
            }
            data["objects"][key] = d
        return json.dumps(data, indent=2)

    def save_json(self, path: str):
        """Save scene graph to a JSON file."""
        with open(path, "w") as f:
            f.write(self.to_json())
        print(f"[SceneGraph] Saved to {path}")

    @classmethod
    def from_json(cls, path: str) -> "SceneGraph":
        """Load scene graph from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        sg = cls()
        sg.frame_count = data.get("frame_count", 0)

        for key, obj_data in data.get("objects", {}).items():
            track_id = obj_data.get("track_id", -1)
            obj = SceneObject(
                id=obj_data.get("id", key),
                label=obj_data["label"],
                position_world=obj_data["position_world"],
                orientation=obj_data.get("orientation", [1, 0, 0, 0]),
                collision_primitive=obj_data.get("collision_primitive", "box"),
                dimensions_m=obj_data.get("dimensions_m", {}),
                mass_kg=obj_data.get("mass_kg", 0.2),
                friction=obj_data.get("friction", 0.4),
                material=obj_data.get("material", "unknown"),
                is_graspable=obj_data.get("is_graspable", True),
                is_deformable=obj_data.get("is_deformable", False),
                track_id=track_id,
                confidence=obj_data.get("confidence", 0.0),
                last_seen_frame=obj_data.get("last_seen_frame", 0),
                frames_since_seen=obj_data.get("frames_since_seen", 0),
                velocity_estimate=obj_data.get("velocity_estimate", [0, 0, 0]),
                relationships=obj_data.get("relationships", []),
            )
            if track_id >= sg._next_track_id:
                sg._next_track_id = track_id + 1
            sg.objects[key] = obj

        return sg
