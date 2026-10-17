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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "replay", serde(rename_all = "camelCase"))]
pub enum WakeCause {
  Travel,
  Depth,
  Oscillation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "replay", serde(rename_all = "camelCase"))]
pub enum Handover {
  Recovered,
  Reassociated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "replay", serde(rename_all = "camelCase"))]
pub enum MatchTier {
  Overlap,
  Contained,
  Near,
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
  #[cfg_attr(feature = "replay", serde(default))]
  pub score: f32,
  pub speed: f32,
  pub velocity_x: f32,
  pub velocity_y: f32,
  pub state: TrackState,
  #[cfg_attr(feature = "replay", serde(default))]
  pub stationary_since_ms: Option<f64>,
  #[cfg_attr(feature = "replay", serde(default))]
  pub last_seen_ms: f64,
  #[cfg_attr(feature = "replay", serde(default))]
  pub attested: bool,
  #[cfg_attr(feature = "replay", serde(default))]
  pub wake_cause: Option<WakeCause>,
  // wake box centre to the anchor centre, in anchor sizes
  #[cfg_attr(feature = "replay", serde(default))]
  pub anchor_distance: Option<f32>,
  #[cfg_attr(feature = "replay", serde(default))]
  pub far_from_anchor: Option<bool>,
  // only on the tick a re-association bound the identity to a new engine track
  #[cfg_attr(feature = "replay", serde(default))]
  pub handover: Option<Handover>,
  #[cfg_attr(feature = "replay", serde(default))]
  pub handover_tier: Option<MatchTier>,
  #[cfg_attr(feature = "replay", serde(default))]
  pub unseen_ms: Option<f64>,
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
