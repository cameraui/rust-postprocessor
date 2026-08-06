#[derive(Debug, Clone)]
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
pub struct Detection {
  pub x: f32,
  pub y: f32,
  pub width: f32,
  pub height: f32,
  pub confidence: f32,
  pub label: String,
}

#[cfg(feature = "replay")]
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct CameraMotion {
  pub x: f64,
  pub y: f64,
}
