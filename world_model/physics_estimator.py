"""
world_model/physics_estimator.py
==================================
Physics Parameter Estimation — learns real physics from observation.

Instead of trusting the VLM's guess of "mass=0.1kg" for everything,
this module OBSERVES how objects actually behave and reverse-engineers
their true physical properties:

  1. Mass estimation: observe free-fall time or bounce height
  2. Friction estimation: observe how far an object slides
  3. Restitution estimation: observe bounce ratio
  4. Material classification: combine visual + physical cues

Usage:
    estimator = PhysicsEstimator()
    
    # After observing an object fall:
    params = estimator.estimate_from_drop(
        initial_height=0.5,      # metres
        time_to_impact=0.32,     # seconds (from temporal tracker)
        bounce_height=0.05,      # metres (from temporal tracker)
    )
    print(params.mass_kg, params.restitution)
    
    # After observing an object slide:
    params = estimator.estimate_from_slide(
        initial_velocity=0.3,    # m/s
        slide_distance=0.12,     # metres
        surface_material="wood", # from VLM
    )
    print(params.friction_coefficient)
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class PhysicsParams:
    """Estimated physical parameters for an object."""
    mass_kg: float = 0.2
    friction_static: float = 0.4
    friction_dynamic: float = 0.3
    restitution: float = 0.3       # bounciness, 0=no bounce, 1=perfect bounce
    density_kg_m3: float = 500.0
    material_class: str = "unknown" # rigid_light, rigid_heavy, soft, liquid, fragile
    confidence: float = 0.5        # how confident are we in these estimates
    estimation_method: str = "vlm_guess"  # how were these estimated


@dataclass
class DropObservation:
    """Observed data from watching an object fall/bounce."""
    initial_height_m: float
    time_to_impact_s: float
    bounce_height_m: float = 0.0
    num_bounces: int = 0
    final_settled: bool = True
    object_dimensions_m: Dict[str, float] = field(default_factory=dict)


@dataclass
class SlideObservation:
    """Observed data from watching an object slide on a surface."""
    initial_velocity_ms: float
    slide_distance_m: float
    time_to_stop_s: float
    surface_material: str = "unknown"


@dataclass
class ContactObservation:
    """Observed data from watching two objects interact."""
    object_a_label: str
    object_b_label: str
    relative_velocity_ms: float
    separation_velocity_ms: float = 0.0  # after impact
    deformation_detected: bool = False


class PhysicsEstimator:
    """
    Estimates physical properties of objects by observing their behaviour.
    
    This is how the world model improves its physics accuracy over time.
    Instead of relying on VLM guesses, it watches what actually happens
    and calibrates its internal simulation to match reality.
    """
    
    # Known material properties for calibration
    MATERIAL_DB = {
        "ceramic": PhysicsParams(density_kg_m3=2400, friction_static=0.5, restitution=0.1, material_class="rigid_heavy"),
        "plastic": PhysicsParams(density_kg_m3=950, friction_static=0.35, restitution=0.4, material_class="rigid_light"),
        "metal":   PhysicsParams(density_kg_m3=7800, friction_static=0.4, restitution=0.3, material_class="rigid_heavy"),
        "wood":    PhysicsParams(density_kg_m3=600, friction_static=0.5, restitution=0.2, material_class="rigid_light"),
        "glass":   PhysicsParams(density_kg_m3=2500, friction_static=0.3, restitution=0.15, material_class="fragile"),
        "rubber":  PhysicsParams(density_kg_m3=1100, friction_static=0.8, restitution=0.8, material_class="soft"),
        "foam":    PhysicsParams(density_kg_m3=30, friction_static=0.6, restitution=0.1, material_class="soft"),
        "paper":   PhysicsParams(density_kg_m3=700, friction_static=0.4, restitution=0.05, material_class="rigid_light"),
    }
    
    GRAVITY = 9.81  # m/s²
    
    def estimate_from_drop(self, observation: DropObservation) -> PhysicsParams:
        """
        Estimate physics from a drop observation.
        
        Uses projectile physics:
          - Free-fall time reveals air resistance (and thus mass-to-area ratio)
          - Bounce height reveals coefficient of restitution
          - Number of bounces reveals energy dissipation
        """
        h = observation.initial_height_m
        t = observation.time_to_impact_s
        
        # Expected free-fall time (no air resistance): t = sqrt(2h/g)
        expected_t = np.sqrt(2 * h / self.GRAVITY)
        
        # If actual time is longer, object has high drag (light/large)
        # If actual time is shorter, measurement error (shouldn't be faster)
        drag_ratio = t / max(expected_t, 0.001)
        
        # Estimate restitution from bounce height
        # e = sqrt(bounce_height / drop_height)
        if observation.bounce_height_m > 0.001:
            restitution = np.sqrt(observation.bounce_height_m / max(h, 0.001))
            restitution = np.clip(restitution, 0.0, 1.0)
        else:
            restitution = 0.05  # no bounce = very low restitution
        
        # Estimate density from dimensions + drag behaviour
        if observation.object_dimensions_m:
            w = observation.object_dimensions_m.get("width", 0.05)
            h_dim = observation.object_dimensions_m.get("height", 0.05)
            d = observation.object_dimensions_m.get("depth", 0.05)
            volume = w * h_dim * d
            
            # Light objects fall slower due to drag
            if drag_ratio > 1.2:
                density = 200  # very light (foam, paper)
            elif drag_ratio > 1.05:
                density = 600  # moderate (plastic, wood)
            else:
                density = 2000  # heavy (ceramic, metal)
            
            mass = density * volume
        else:
            mass = 0.2
            density = 1000
        
        # Classify material based on restitution + density
        if restitution > 0.6:
            material_class = "rubber_like"
        elif restitution < 0.1 and density > 2000:
            material_class = "fragile"
        elif density > 5000:
            material_class = "rigid_heavy"
        elif density < 200:
            material_class = "soft"
        else:
            material_class = "rigid_light"
        
        # Confidence based on how many observations we used
        confidence = 0.6  # drop test gives moderate confidence
        if observation.num_bounces > 0:
            confidence += 0.1
        if observation.object_dimensions_m:
            confidence += 0.1
        
        result = PhysicsParams(
            mass_kg=float(np.clip(mass, 0.001, 50.0)),
            friction_static=0.4,  # can't estimate from drop alone
            friction_dynamic=0.3,
            restitution=float(restitution),
            density_kg_m3=float(density),
            material_class=material_class,
            confidence=min(confidence, 1.0),
            estimation_method="drop_observation",
        )
        
        self._print_estimate("Drop Test", result)
        return result
    
    def estimate_from_slide(self, observation: SlideObservation) -> PhysicsParams:
        """
        Estimate friction from a slide observation.
        
        Uses kinematic equation:
          v² = v₀² - 2μg·d
          At stop: 0 = v₀² - 2μg·d
          μ = v₀² / (2g·d)
        """
        v0 = observation.initial_velocity_ms
        d = observation.slide_distance_m
        
        if d > 0.001 and v0 > 0.001:
            mu = (v0 ** 2) / (2 * self.GRAVITY * d)
            mu = np.clip(mu, 0.05, 1.5)
        else:
            mu = 0.4  # default
        
        # Cross-reference with surface material
        surface_friction = 0.4
        if observation.surface_material in self.MATERIAL_DB:
            surface_friction = self.MATERIAL_DB[observation.surface_material].friction_static
        
        # The measured mu is the combined friction of object + surface
        # object_friction ≈ mu * 2 - surface_friction (rough approximation)
        object_friction = np.clip(mu * 2 - surface_friction, 0.1, 1.2)
        
        result = PhysicsParams(
            friction_static=float(object_friction),
            friction_dynamic=float(object_friction * 0.8),
            confidence=0.7,
            estimation_method="slide_observation",
        )
        
        self._print_estimate("Slide Test", result)
        return result
    
    def estimate_from_contact(self, observation: ContactObservation) -> PhysicsParams:
        """
        Estimate restitution from observing a collision between two objects.
        
        Coefficient of restitution = separation_velocity / approach_velocity
        """
        if observation.relative_velocity_ms > 0.001:
            e = observation.separation_velocity_ms / observation.relative_velocity_ms
            e = np.clip(e, 0.0, 1.0)
        else:
            e = 0.3
        
        material_class = "soft" if observation.deformation_detected else "rigid_light"
        
        result = PhysicsParams(
            restitution=float(e),
            material_class=material_class,
            confidence=0.65,
            estimation_method="contact_observation",
        )
        
        self._print_estimate("Contact Test", result)
        return result
    
    def estimate_from_material(self, material_name: str,
                               dimensions_m: Dict[str, float] = None) -> PhysicsParams:
        """
        Estimate physics from material name (VLM output) + known properties.
        
        This is the baseline estimate when no physical observations are available.
        Better than guessing, but worse than observing.
        """
        material_lower = material_name.lower()
        
        # Check known materials
        for key, params in self.MATERIAL_DB.items():
            if key in material_lower:
                result = PhysicsParams(
                    mass_kg=params.density_kg_m3 * self._compute_volume(dimensions_m),
                    friction_static=params.friction_static,
                    friction_dynamic=params.friction_static * 0.8,
                    restitution=params.restitution,
                    density_kg_m3=params.density_kg_m3,
                    material_class=params.material_class,
                    confidence=0.4,  # material lookup is moderate confidence
                    estimation_method="material_database",
                )
                self._print_estimate(f"Material DB ({key})", result)
                return result
        
        # Unknown material — return defaults
        return PhysicsParams(
            mass_kg=0.2,
            confidence=0.2,
            estimation_method="default_guess",
        )
    
    def merge_estimates(self, estimates: List[PhysicsParams]) -> PhysicsParams:
        """
        Merge multiple estimates into one best-guess.
        
        Uses confidence-weighted averaging across all estimation sources
        (drop test, slide test, material database, VLM guess).
        """
        if not estimates:
            return PhysicsParams()
        
        if len(estimates) == 1:
            return estimates[0]
        
        weights = np.array([e.confidence for e in estimates])
        if weights.sum() == 0:
            weights = np.ones(len(estimates))
        weights = weights / weights.sum()
        
        merged = PhysicsParams(
            mass_kg=float(np.average([e.mass_kg for e in estimates], weights=weights)),
            friction_static=float(np.average([e.friction_static for e in estimates], weights=weights)),
            friction_dynamic=float(np.average([e.friction_dynamic for e in estimates], weights=weights)),
            restitution=float(np.average([e.restitution for e in estimates], weights=weights)),
            density_kg_m3=float(np.average([e.density_kg_m3 for e in estimates], weights=weights)),
            material_class=max(estimates, key=lambda e: e.confidence).material_class,
            confidence=float(min(max(weights) + 0.1, 1.0)),
            estimation_method="merged_" + "+".join(e.estimation_method for e in estimates),
        )
        
        return merged
    
    def _compute_volume(self, dims: Dict[str, float] = None) -> float:
        """Compute volume from dimensions dict."""
        if not dims:
            return 0.05 * 0.05 * 0.05  # 5cm cube default
        w = dims.get("width", 0.05)
        h = dims.get("height", 0.05)
        d = dims.get("depth", 0.05)
        return w * h * d
    
    def _print_estimate(self, method: str, params: PhysicsParams):
        """Print estimation results."""
        print(f"\n[PhysicsEst] {method}:")
        print(f"  Mass: {params.mass_kg:.3f}kg | "
              f"Friction: {params.friction_static:.2f} | "
              f"Restitution: {params.restitution:.2f}")
        print(f"  Material: {params.material_class} | "
              f"Confidence: {params.confidence:.0%}")

