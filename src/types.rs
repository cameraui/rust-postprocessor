//! Internal types mirroring the JS SDK shapes. `lib.rs` converts at the napi boundary.

#[derive(Debug, Clone)]
pub struct Detection {
  pub x: f32,
  pub y: f32,
  pub width: f32,
  pub height: f32,
  pub confidence: f32,
  pub label: String,
}

/// Camera ego-motion per frame step, normalized. x>0 = panned right, y>0 = tilted down.
/// Keeps Kalman predictions aligned with the world frame across pans.
#[derive(Debug, Clone, Copy)]
pub struct CameraMotion {
  pub x: f64,
  pub y: f64,
}

/// Tracked detection with stable identity across frames.
#[derive(Debug, Clone)]
pub struct TrackedDetection {
  pub x: f32,
  pub y: f32,
  pub width: f32,
  pub height: f32,
  pub confidence: f32,
  pub label: String,
  pub track_id: u32,
  pub track_age: u32,
  /// True when the box is kept alive via Kalman extrapolation (no fresh match).
  pub track_lost: bool,
  /// `sqrt(vx² + vy²)` in normalized units per frame step. 0 with only one sample.
  pub track_speed: f32,
  pub track_velocity_x: f32,
  pub track_velocity_y: f32,
}
