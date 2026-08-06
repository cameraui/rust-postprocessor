#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "replay", serde(rename_all = "camelCase"))]
pub enum TrackState {
  Tentative,
  Active,
  Stationary,
  Lost,
  Departed,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "replay", serde(rename_all = "camelCase"))]
pub struct TrackSnapshot {
  pub track_id: u32,
  pub label: String,
  pub x: f32,
  pub y: f32,
  pub width: f32,
  pub height: f32,
  pub confidence: f32,
  pub speed: f32,
  pub velocity_x: f32,
  pub velocity_y: f32,
  pub state: TrackState,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "replay", serde(tag = "type", rename_all = "camelCase"))]
pub enum SemanticEvent {
  ObjectEntered(TrackSnapshot),
  ObjectLost(TrackSnapshot),
  ObjectRecovered(TrackSnapshot),
  ObjectSettled(TrackSnapshot),
  ObjectWoke(TrackSnapshot),
  ObjectDeparted(TrackSnapshot),
  BestShotUpdated(TrackSnapshot),
}

impl SemanticEvent {
  pub fn kind(&self) -> &'static str {
    match self {
      SemanticEvent::ObjectEntered(_) => "objectEntered",
      SemanticEvent::ObjectLost(_) => "objectLost",
      SemanticEvent::ObjectRecovered(_) => "objectRecovered",
      SemanticEvent::ObjectSettled(_) => "objectSettled",
      SemanticEvent::ObjectWoke(_) => "objectWoke",
      SemanticEvent::ObjectDeparted(_) => "objectDeparted",
      SemanticEvent::BestShotUpdated(_) => "bestShotUpdated",
    }
  }
}
