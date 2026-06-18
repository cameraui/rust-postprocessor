//! NMS, IoU and object tracking exposed to Node.js via napi-rs.
//! All coordinates are normalized to `[0.0, 1.0]`.

mod embedding;
mod iou;
mod line_crossing;
mod merge;
mod nms;
mod tracker;
mod types;
mod zone_filter;

use napi_derive::napi;

use crate::line_crossing::{
  CrossingDirection as InnerCrossingDirection, DetectionLineInput as InnerDetectionLineInput,
  LineDirectionFilter as InnerLineDirectionFilter,
};
use crate::tracker::{ObjectTracker as InnerObjectTracker, ObjectTrackerConfig};
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
pub struct TrackedDetection {
  pub x: f64,
  pub y: f64,
  pub width: f64,
  pub height: f64,
  pub confidence: f64,
  pub label: String,
  pub track_id: u32,
  pub track_age: u32,
  /// True when the box is kept alive by Kalman extrapolation, not a fresh match.
  pub track_lost: bool,
  /// `sqrt(vx² + vy²)` in normalized units per frame; 0 with only one sample.
  pub track_speed: f64,
  pub track_velocity_x: f64,
  pub track_velocity_y: f64,
}

#[napi(object)]
pub struct BoundingBox {
  pub x: f64,
  pub y: f64,
  pub width: f64,
  pub height: f64,
}

/// Camera ego-motion for one frame step in normalized coords; when supplied
/// to `update()`, Kalman predictions are transformed to survive pans.
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
  /// Any overlap with the polygon counts.
  Intersect,
  /// All four corners must be inside the polygon.
  Contain,
}

/// `points` are polygon vertices in `[0, 100]` UI coordinates; auto-closed.
#[napi(object)]
pub struct DetectionZone {
  pub labels: Vec<String>,
  pub filter: ZoneFilterMode,
  /// Mapped to `type` on the JS side of the SDK.
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

/// `points` are the two handle endpoints in `[0, 100]` UI coordinates;
/// empty `labels` means "any label".
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

#[napi(object)]
pub struct UpdateResult {
  pub tracked: Vec<TrackedDetection>,
  pub crossings: Vec<LineCrossingEvent>,
}

#[napi(object)]
pub struct ObjectTrackerOptions {
  /// Default 0.3.
  pub iou_threshold: Option<f64>,
  /// Frames a track survives without a fresh detection. Default 15.
  pub hit_counter_max: Option<i32>,
  /// Frames a new track must be matched before getting a permanent id. Default 3.
  pub initialization_delay: Option<i32>,
  /// Frames a dead track stays available for ReID re-matching. Default: disabled.
  pub reid_hit_counter_max: Option<i32>,
  /// Cosine distance threshold for appearance ReID; only used when a frame
  /// buffer is passed to `update()`. Default 0.4.
  pub reid_embedding_threshold: Option<f64>,
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

fn from_internal_tracked(d: crate::types::TrackedDetection) -> TrackedDetection {
  TrackedDetection {
    x: d.x as f64,
    y: d.y as f64,
    width: d.width as f64,
    height: d.height as f64,
    confidence: d.confidence as f64,
    label: d.label,
    track_id: d.track_id,
    track_age: d.track_age,
    track_lost: d.track_lost,
    track_speed: d.track_speed as f64,
    track_velocity_x: d.track_velocity_x as f64,
    track_velocity_y: d.track_velocity_y as f64,
  }
}

/// Greedy NMS, suppressing only against higher-confidence boxes of the same
/// `label`. Output sorted by confidence descending.
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

/// Like [`nms`] but returns the surviving input indices, so callers can keep
/// extra fields that the Detection round-trip would drop.
#[napi(js_name = "nmsIndices")]
pub fn nms_indices(detections: Vec<Detection>, iou_threshold: f64) -> Vec<u32> {
  let internal: Vec<crate::types::Detection> = detections.into_iter().map(to_internal).collect();
  crate::nms::nms_indices(&internal, iou_threshold as f32)
    .into_iter()
    .map(|i| i as u32)
    .collect()
}

/// Cluster nearby/overlapping same-label detections into union boxes. Boxes
/// join when their top-left corners are within `closeThreshold` on both axes
/// or their IoU exceeds `iouThreshold`; the cluster keeps the max confidence.
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

/// IoU between two normalized `[x, y, width, height]` boxes.
#[napi(js_name = "boxIou")]
pub fn box_iou(a: BoundingBox, b: BoundingBox) -> f64 {
  let aa = [a.x as f32, a.y as f32, a.width as f32, a.height as f32];
  let bb = [b.x as f32, b.y as f32, b.width as f32, b.height as f32];
  crate::iou::box_iou(&aa, &bb) as f64
}

/// Multi-class IoU + Kalman tracker (norfair-rs, one sub-tracker per label).
/// Track ids are stable and globally unique; feed normalized coordinates.
#[napi]
pub struct ObjectTracker {
  inner: InnerObjectTracker,
}

#[napi]
impl ObjectTracker {
  #[napi(constructor)]
  pub fn new(options: Option<ObjectTrackerOptions>) -> Self {
    let mut config = ObjectTrackerConfig::default();
    if let Some(opts) = options {
      if let Some(t) = opts.iou_threshold {
        config.iou_threshold = t as f32;
      }
      if let Some(h) = opts.hit_counter_max {
        config.hit_counter_max = h;
      }
      if let Some(i) = opts.initialization_delay {
        config.initialization_delay = i;
      }
      if let Some(r) = opts.reid_hit_counter_max {
        config.reid_hit_counter_max = if r > 0 { Some(r) } else { None };
      }
      if let Some(t) = opts.reid_embedding_threshold {
        config.reid_embedding_threshold = t;
      }
    }
    Self {
      inner: InnerObjectTracker::new(config),
    }
  }

  /// Process one frame's detections and return active tracks plus any
  /// crossing events. `timestampMs` is forwarded onto crossing events.
  /// `cameraMotion` (optional) stabilizes Kalman predictions across pans.
  #[napi]
  pub fn update(
    &mut self,
    detections: Vec<Detection>,
    timestamp_ms: f64,
    frame: Option<napi::bindgen_prelude::Buffer>,
    frame_width: Option<u32>,
    frame_height: Option<u32>,
    camera_motion: Option<CameraMotion>,
  ) -> UpdateResult {
    let internal: Vec<crate::types::Detection> = detections.into_iter().map(to_internal).collect();
    let frame_ref = match (&frame, frame_width, frame_height) {
      (Some(buf), Some(w), Some(h)) => Some((buf.as_ref(), w, h)),
      _ => None,
    };
    let motion = camera_motion.map(|m| crate::types::CameraMotion { x: m.x, y: m.y });
    let result = self.inner.update(internal, timestamp_ms, frame_ref, motion);
    UpdateResult {
      tracked: result
        .tracked
        .into_iter()
        .map(from_internal_tracked)
        .collect(),
      crossings: result
        .crossings
        .into_iter()
        .map(|c| LineCrossingEvent {
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
        })
        .collect(),
    }
  }

  /// Replace the configured crossing lines (empty array disables them).
  /// `aspectRatio` is `width / height` so the line renders perpendicular to
  /// the drawn handle. Crossing memory is cleared on every call.
  #[napi]
  pub fn set_lines(&mut self, lines: Vec<DetectionLine>, aspect_ratio: f64) {
    let internal: Vec<InnerDetectionLineInput> = lines
      .into_iter()
      .filter_map(detection_line_to_internal)
      .collect();
    self.inner.set_lines(internal, aspect_ratio as f32);
  }

  /// Replace the configured detection zones (empty array disables filtering).
  /// Coordinates are in `[0, 100]` UI space; normalized and auto-closed
  /// internally. Applied at the start of every `update()`.
  #[napi]
  pub fn set_zones(&mut self, zones: Vec<DetectionZone>) {
    let internal: Vec<InnerZoneInput> = zones
      .into_iter()
      .filter_map(detection_zone_to_internal)
      .collect();
    self.inner.set_zones(internal);
  }

  /// Drop detections below this score at the start of every `update()`,
  /// before zone filtering or tracking. Default 0.0 (no threshold).
  #[napi]
  pub fn set_min_confidence(&mut self, min_confidence: f64) {
    self.inner.set_min_confidence(min_confidence as f32);
  }

  /// How many frames a dead track stays available for ReID re-matching;
  /// the old id is preserved on merge. Pass 0 to disable.
  #[napi]
  pub fn set_reid_hit_counter_max(&mut self, frames: i32) {
    self.inner.set_reid_hit_counter_max(frames);
  }

  /// Refresh the ReID counter for all dead tracks back to max. Call each
  /// frame while a cascade is active so dead tracks don't expire mid-window.
  #[napi]
  pub fn refresh_reid(&mut self) {
    self.inner.refresh_reid();
  }

  /// Apply zones + confidence threshold without advancing tracker state,
  /// returning the surviving input indices. Used by the external-sensor-write
  /// path that needs zone filtering without running the full tracker.
  #[napi]
  pub fn filter_indices(&self, detections: Vec<Detection>) -> Vec<u32> {
    let internal: Vec<crate::types::Detection> = detections.into_iter().map(to_internal).collect();
    self.inner.filter_indices(&internal)
  }

  /// Drop every active track; the next `update()` restarts ids from 1.
  #[napi]
  pub fn reset(&mut self) {
    self.inner.reset();
  }

  #[napi(getter)]
  pub fn track_count(&self) -> u32 {
    self.inner.track_count() as u32
  }
}
