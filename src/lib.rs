mod iou;
mod line_crossing;
mod merge;
mod nms;
#[cfg(feature = "replay")]
pub mod replay;
pub mod semantic;
mod types;
pub mod world;
mod zone_filter;

use napi_derive::napi;

use crate::line_crossing::{
  CrossingDirection as InnerCrossingDirection, DetectionLineInput as InnerDetectionLineInput,
  LineDirectionFilter as InnerLineDirectionFilter,
};
use crate::zone_filter::{
  ZoneFilterMode as InnerZoneFilterMode, ZoneInput as InnerZoneInput,
  ZoneMatchType as InnerZoneMatchType,
};

#[napi(object)]
pub struct Detection {
  pub x: f64,
  pub y: f64,
  pub width: f64,
  pub height: f64,
  pub confidence: f64,
  pub label: String,
}

#[napi(object)]
pub struct BoundingBox {
  pub x: f64,
  pub y: f64,
  pub width: f64,
  pub height: f64,
}

#[napi(object)]
pub struct CameraMotion {
  pub x: f64,
  pub y: f64,
}

#[napi(string_enum = "kebab-case")]
pub enum ZoneFilterMode {
  Include,
  Exclude,
}

#[napi(string_enum = "kebab-case")]
pub enum ZoneMatchType {
  Intersect,
  Contain,
}

#[napi(object)]
pub struct DetectionZone {
  pub labels: Vec<String>,
  pub filter: ZoneFilterMode,
  pub match_type: ZoneMatchType,
  pub is_privacy_mask: bool,
  pub points: Vec<Vec<f64>>,
}

#[napi(string_enum = "kebab-case")]
pub enum LineDirection {
  Both,
  AToB,
  BToA,
}

#[napi(object)]
pub struct DetectionLine {
  pub name: String,
  pub direction: LineDirection,
  pub labels: Vec<String>,
  pub points: Vec<Vec<f64>>,
}

#[napi(object)]
pub struct LineCrossingEvent {
  pub line_name: String,
  pub direction: LineDirection,
  pub track_id: u32,
  pub label: String,
  pub confidence: f64,
  pub timestamp_ms: f64,
  pub prev_x: f64,
  pub prev_y: f64,
  pub curr_x: f64,
  pub curr_y: f64,
}

fn to_internal(d: Detection) -> crate::types::Detection {
  crate::types::Detection {
    x: d.x as f32,
    y: d.y as f32,
    width: d.width as f32,
    height: d.height as f32,
    confidence: d.confidence as f32,
    label: d.label,
  }
}

fn from_internal(d: crate::types::Detection) -> Detection {
  Detection {
    x: d.x as f64,
    y: d.y as f64,
    width: d.width as f64,
    height: d.height as f64,
    confidence: d.confidence as f64,
    label: d.label,
  }
}

fn zone_filter_to_internal(m: ZoneFilterMode) -> InnerZoneFilterMode {
  match m {
    ZoneFilterMode::Include => InnerZoneFilterMode::Include,
    ZoneFilterMode::Exclude => InnerZoneFilterMode::Exclude,
  }
}

fn zone_match_to_internal(m: ZoneMatchType) -> InnerZoneMatchType {
  match m {
    ZoneMatchType::Intersect => InnerZoneMatchType::Intersect,
    ZoneMatchType::Contain => InnerZoneMatchType::Contain,
  }
}

fn detection_zone_to_internal(zone: DetectionZone) -> Option<InnerZoneInput> {
  let mut points: Vec<[f64; 2]> = Vec::with_capacity(zone.points.len());
  for p in zone.points {
    if p.len() != 2 {
      return None;
    }
    points.push([p[0], p[1]]);
  }
  Some(InnerZoneInput {
    labels: zone.labels,
    filter: zone_filter_to_internal(zone.filter),
    match_type: zone_match_to_internal(zone.match_type),
    is_privacy_mask: zone.is_privacy_mask,
    points,
  })
}

fn line_direction_to_internal(d: LineDirection) -> InnerLineDirectionFilter {
  match d {
    LineDirection::Both => InnerLineDirectionFilter::Both,
    LineDirection::AToB => InnerLineDirectionFilter::AToB,
    LineDirection::BToA => InnerLineDirectionFilter::BToA,
  }
}

fn line_direction_from_internal(d: InnerCrossingDirection) -> LineDirection {
  match d {
    InnerCrossingDirection::AToB => LineDirection::AToB,
    InnerCrossingDirection::BToA => LineDirection::BToA,
  }
}

fn detection_line_to_internal(line: DetectionLine) -> Option<InnerDetectionLineInput> {
  if line.points.len() != 2 {
    return None;
  }
  let p1 = &line.points[0];
  let p2 = &line.points[1];
  if p1.len() != 2 || p2.len() != 2 {
    return None;
  }
  Some(InnerDetectionLineInput {
    name: line.name,
    direction: line_direction_to_internal(line.direction),
    labels: line.labels,
    points: [[p1[0], p1[1]], [p2[0], p2[1]]],
  })
}

fn crossing_from_internal(c: crate::line_crossing::LineCrossingEvent) -> LineCrossingEvent {
  LineCrossingEvent {
    line_name: c.line_name,
    direction: line_direction_from_internal(c.direction),
    track_id: c.track_id,
    label: c.label,
    confidence: c.confidence as f64,
    timestamp_ms: c.timestamp_ms,
    prev_x: c.prev_pos[0] as f64,
    prev_y: c.prev_pos[1] as f64,
    curr_x: c.curr_pos[0] as f64,
    curr_y: c.curr_pos[1] as f64,
  }
}

#[napi]
pub fn nms(
  detections: Vec<Detection>,
  iou_threshold: f64,
  max_detections: Option<u32>,
) -> Vec<Detection> {
  let internal: Vec<crate::types::Detection> = detections.into_iter().map(to_internal).collect();
  let max = max_detections.map(|n| n as usize);
  let kept = crate::nms::nms(internal, iou_threshold as f32, max);
  kept.into_iter().map(from_internal).collect()
}

#[napi(js_name = "nmsIndices")]
pub fn nms_indices(detections: Vec<Detection>, iou_threshold: f64) -> Vec<u32> {
  let internal: Vec<crate::types::Detection> = detections.into_iter().map(to_internal).collect();
  crate::nms::nms_indices(&internal, iou_threshold as f32)
    .into_iter()
    .map(|i| i as u32)
    .collect()
}

#[napi]
pub fn merge(
  detections: Vec<Detection>,
  iou_threshold: f64,
  close_threshold: f64,
) -> Vec<Detection> {
  let internal: Vec<crate::types::Detection> = detections.into_iter().map(to_internal).collect();
  let merged =
    crate::merge::merge_detections(internal, iou_threshold as f32, close_threshold as f32);
  merged.into_iter().map(from_internal).collect()
}

#[napi(js_name = "boxIou")]
pub fn box_iou(a: BoundingBox, b: BoundingBox) -> f64 {
  let aa = [a.x as f32, a.y as f32, a.width as f32, a.height as f32];
  let bb = [b.x as f32, b.y as f32, b.width as f32, b.height as f32];
  crate::iou::box_iou(&aa, &bb) as f64
}

#[napi(object)]
pub struct WorldObject {
  pub track_id: u32,
  pub label: String,
  pub x: f64,
  pub y: f64,
  pub width: f64,
  pub height: f64,
  pub confidence: f64,
  pub speed: f64,
  pub velocity_x: f64,
  pub velocity_y: f64,
  /// One of: tentative, active, stationary, lost, departed.
  pub state: String,
}

#[napi(object)]
pub struct WorldEvent {
  /// One of: objectEntered, objectLost, objectRecovered, objectSettled,
  /// objectWoke, objectDeparted, bestShotUpdated.
  pub event_type: String,
  pub object: WorldObject,
}

#[napi(object)]
pub struct WorldIngestResult {
  pub tracked: Vec<WorldObject>,
  pub created: Vec<u32>,
  pub removed: Vec<u32>,
  pub events: Vec<WorldEvent>,
  pub crossings: Vec<LineCrossingEvent>,
}

fn world_object(s: &crate::semantic::TrackSnapshot) -> WorldObject {
  WorldObject {
    track_id: s.track_id,
    label: s.label.clone(),
    x: s.x as f64,
    y: s.y as f64,
    width: s.width as f64,
    height: s.height as f64,
    confidence: s.confidence as f64,
    speed: s.speed as f64,
    velocity_x: s.velocity_x as f64,
    velocity_y: s.velocity_y as f64,
    state: match s.state {
      crate::semantic::TrackState::Tentative => "tentative",
      crate::semantic::TrackState::Active => "active",
      crate::semantic::TrackState::Stationary => "stationary",
      crate::semantic::TrackState::Lost => "lost",
      crate::semantic::TrackState::Departed => "departed",
    }
    .to_string(),
  }
}

fn world_event(e: &crate::semantic::SemanticEvent) -> WorldEvent {
  use crate::semantic::SemanticEvent as E;
  let snapshot = match e {
    E::ObjectEntered(s)
    | E::ObjectLost(s)
    | E::ObjectRecovered(s)
    | E::ObjectSettled(s)
    | E::ObjectWoke(s)
    | E::ObjectDeparted(s)
    | E::BestShotUpdated(s) => s,
  };
  WorldEvent {
    event_type: e.kind().to_string(),
    object: world_object(snapshot),
  }
}

#[napi]
pub struct CameraWorld {
  inner: crate::world::CameraWorld,
}

impl Default for CameraWorld {
  fn default() -> Self {
    Self::new()
  }
}

#[napi]
impl CameraWorld {
  #[napi(constructor)]
  pub fn new() -> Self {
    Self {
      inner: crate::world::CameraWorld::new(crate::world::WorldConfig::default()),
    }
  }

  #[napi]
  pub fn ingest(
    &mut self,
    timestamp_ms: f64,
    detections: Vec<Detection>,
    camera_motion: Option<CameraMotion>,
  ) -> WorldIngestResult {
    let internal: Vec<crate::types::Detection> = detections.into_iter().map(to_internal).collect();
    let motion = camera_motion.map(|m| (m.x as f32, m.y as f32));
    let update = self.inner.ingest(timestamp_ms, &internal, motion);
    WorldIngestResult {
      tracked: update.tracked.iter().map(world_object).collect(),
      created: update.created,
      removed: update.removed,
      events: update.events.iter().map(world_event).collect(),
      crossings: update
        .crossings
        .into_iter()
        .map(crossing_from_internal)
        .collect(),
    }
  }

  #[napi]
  pub fn set_lines(&mut self, lines: Vec<DetectionLine>, aspect_ratio: f64) {
    let internal: Vec<_> = lines
      .into_iter()
      .filter_map(detection_line_to_internal)
      .collect();
    self.inner.set_lines(internal, aspect_ratio as f32);
  }

  #[napi]
  pub fn notify_camera_move(&mut self) {
    self.inner.notify_camera_move();
  }

  #[napi]
  pub fn set_zones(&mut self, zones: Vec<DetectionZone>) {
    let internal: Vec<_> = zones
      .into_iter()
      .filter_map(detection_zone_to_internal)
      .collect();
    self.inner.set_zones(internal);
  }

  #[napi]
  pub fn set_min_confidence(&mut self, min_confidence: f64) {
    self.inner.set_min_confidence(min_confidence as f32);
  }

  #[napi]
  pub fn filter_indices(&self, detections: Vec<Detection>) -> Vec<u32> {
    let internal: Vec<crate::types::Detection> = detections.into_iter().map(to_internal).collect();
    self.inner.filter_indices(&internal)
  }
}
