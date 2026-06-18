//! Per-class IoU tracker over `norfair-rs` with stable global track ids.
//!
//! One `norfair_rs::Tracker` per class label; norfair's per-instance ids are
//! remapped into a single global namespace so classes never collide.

use std::collections::{HashMap, HashSet};

use nalgebra::DMatrix;
use norfair_rs::camera_motion::TranslationTransformation;
use norfair_rs::distances::{distance_function_by_name, DistanceFunction, ScalarDistance};
use norfair_rs::{Detection as NfDetection, TrackedObject, Tracker, TrackerConfig};

use crate::embedding;
use crate::line_crossing::{
  prepare_lines, segment_intersection, CrossingDirection, DetectionLineInput, LineCrossingEvent,
  LineDirectionFilter, PreparedLine,
};
use crate::types::{CameraMotion, Detection, TrackedDetection};
use crate::zone_filter::{filter_indices, prepare_zones, PreparedZones, ZoneInput};

/// Move items out of `src` at the given sorted indices without cloning.
fn extract_by_indices(mut src: Vec<Detection>, indices: &[u32]) -> Vec<Detection> {
  if indices.len() == src.len() {
    return src;
  }
  let mut out = Vec::with_capacity(indices.len());
  // Option lets us take ownership of the label String out of each slot.
  let mut slots: Vec<Option<Detection>> = src.drain(..).map(Some).collect();
  for &i in indices {
    if let Some(det) = slots[i as usize].take() {
      out.push(det);
    }
  }
  out
}

/// Configuration for [`ObjectTracker`].
#[derive(Debug, Clone)]
pub struct ObjectTrackerConfig {
  /// IoU match threshold. Higher = stricter.
  pub iou_threshold: f32,
  /// Frames a track survives without a fresh detection before being dropped.
  pub hit_counter_max: i32,
  /// Frames a new track must be matched before it gets a permanent id.
  pub initialization_delay: i32,
  /// Frames a dead track stays available for ReID re-matching; None disables.
  pub reid_hit_counter_max: Option<i32>,
  /// Cosine distance threshold for appearance-based ReID matching.
  pub reid_embedding_threshold: f64,
}

impl Default for ObjectTrackerConfig {
  fn default() -> Self {
    Self {
      iou_threshold: 0.3,
      hit_counter_max: 15,
      initialization_delay: 3,
      reid_hit_counter_max: None,
      reid_embedding_threshold: 0.4,
    }
  }
}

struct ClassTracker {
  tracker: Tracker,
  /// norfair's per-instance `global_id` → our externally-visible track id.
  id_map: HashMap<i32, u32>,
  /// Previous-frame centroid `(cx, cy)` per track, for line crossing segments.
  prev_centroid_map: HashMap<u32, (f32, f32)>,
}

/// Embedded in each norfair Detection's `data`; after `update()` a matching
/// `frame_seq` on `last_detection` reliably means the track was matched this
/// frame, regardless of hit_counter semantics.
#[derive(Debug, Clone)]
struct FrameTag {
  frame_seq: u64,
}

#[derive(Debug, Default)]
pub struct UpdateResult {
  pub tracked: Vec<TrackedDetection>,
  pub crossings: Vec<LineCrossingEvent>,
}

pub struct ObjectTracker {
  config: ObjectTrackerConfig,
  trackers: HashMap<String, ClassTracker>,
  next_track_id: u32,
  prepared_lines: Vec<PreparedLine>,
  /// `(track_id, line_name)` pairs that have already fired, so a track can't
  /// trigger the same line twice in its lifetime.
  crossing_memory: HashSet<(u32, String)>,
  prepared_zones: PreparedZones,
  min_confidence: f32,
  frame_seq: u64,
  /// Reused across frames to avoid re-allocating per-frame label buckets.
  by_label: HashMap<String, Vec<Detection>>,
}

impl ObjectTracker {
  pub fn new(config: ObjectTrackerConfig) -> Self {
    Self {
      config,
      trackers: HashMap::new(),
      next_track_id: 1,
      prepared_lines: Vec::new(),
      crossing_memory: HashSet::new(),
      prepared_zones: PreparedZones::default(),
      min_confidence: 0.0,
      frame_seq: 0,
      by_label: HashMap::new(),
    }
  }

  /// Replace the detection zones. Coordinates are in `[0, 100]` UI space.
  pub fn set_zones(&mut self, zones: Vec<ZoneInput>) {
    self.prepared_zones = prepare_zones(&zones);
  }

  pub fn set_min_confidence(&mut self, min_confidence: f32) {
    self.min_confidence = min_confidence.max(0.0);
  }

  /// Apply zones + confidence threshold without advancing the tracker,
  /// returning the indices of detections that pass. Used by the external
  /// sensor write path that needs zone filtering but not full tracking.
  pub fn filter_indices(&self, detections: &[Detection]) -> Vec<u32> {
    filter_indices(detections, &self.prepared_zones, self.min_confidence)
  }

  /// Replace the crossing lines. `aspect_ratio` is the camera's `width/height`.
  /// Crossing memory is cleared so edits can be validated immediately.
  pub fn set_lines(&mut self, lines: Vec<DetectionLineInput>, aspect_ratio: f32) {
    self.prepared_lines = prepare_lines(&lines, aspect_ratio);
    self.crossing_memory.clear();
  }

  /// Frames a dead track stays available for ReID re-matching; 0 disables.
  pub fn set_reid_hit_counter_max(&mut self, frames: i32) {
    let value = if frames > 0 { Some(frames) } else { None };
    self.config.reid_hit_counter_max = value;
    for class in self.trackers.values_mut() {
      class.tracker.config.reid_hit_counter_max = value;
    }
  }

  /// Reset the ReID counter for all dead tracks. Calling this every frame
  /// while a cascade is active keeps dead tracks from expiring.
  pub fn refresh_reid(&mut self) {
    let max = match self.config.reid_hit_counter_max {
      Some(m) => m,
      None => return,
    };
    for class in self.trackers.values_mut() {
      for obj in &mut class.tracker.tracked_objects {
        if obj.hit_counter < 0 {
          if let Some(ref mut rc) = obj.reid_hit_counter {
            *rc = max;
          }
        }
      }
    }
  }

  /// Drop all tracks; track ids restart from 1.
  pub fn reset(&mut self) {
    self.trackers.clear();
    self.next_track_id = 1;
    self.crossing_memory.clear();
  }

  pub fn track_count(&self) -> usize {
    self
      .trackers
      .values()
      .map(|c| c.tracker.tracked_objects.len())
      .sum()
  }

  /// Process a frame's detections and return the active tracks plus any
  /// line-crossing events that fired. Pipeline: confidence filter, zone
  /// filter, per-class norfair tracking, line-crossing detection.
  pub fn update(
    &mut self,
    detections: Vec<Detection>,
    timestamp_ms: f64,
    frame: Option<(&[u8], u32, u32)>,
    camera_motion: Option<CameraMotion>,
  ) -> UpdateResult {
    self.frame_seq += 1;

    let detections = {
      let indices = filter_indices(&detections, &self.prepared_zones, self.min_confidence);
      extract_by_indices(detections, &indices)
    };

    let mut tracked: Vec<TrackedDetection> = Vec::new();

    if detections.is_empty() {
      // Still tick every sub-tracker so Kalman filters extrapolate and age.
      tracked.extend(self.tick_empty(timestamp_ms, camera_motion));
    } else {
      for vec in self.by_label.values_mut() {
        vec.clear();
      }
      for det in detections {
        self
          .by_label
          .entry(det.label.clone())
          .or_default()
          .push(det);
      }

      // Keys collected upfront so we don't borrow self while calling run_class.
      let empty_labels: Vec<String> = self
        .trackers
        .keys()
        .filter(|k| self.by_label.get(k.as_str()).is_none_or(|v| v.is_empty()))
        .cloned()
        .collect();
      for label in &empty_labels {
        tracked.extend(self.run_class(label, Vec::new(), timestamp_ms, frame, camera_motion));
      }

      let active_labels: Vec<String> = self
        .by_label
        .keys()
        .filter(|k| !self.by_label[k.as_str()].is_empty())
        .cloned()
        .collect();
      for label in active_labels {
        let dets = std::mem::take(self.by_label.get_mut(&label).unwrap());
        tracked.extend(self.run_class(&label, dets, timestamp_ms, frame, camera_motion));
      }
    }

    // Skip all crossing bookkeeping when no lines are configured.
    let crossings = if self.prepared_lines.is_empty() {
      Vec::new()
    } else {
      let c = self.compute_crossings(&tracked, timestamp_ms);
      self.refresh_centroid_history(&tracked);
      self.gc_crossing_memory();
      c
    };

    UpdateResult { tracked, crossings }
  }

  /// Tick every sub-tracker with no detections.
  fn tick_empty(
    &mut self,
    timestamp_ms: f64,
    camera_motion: Option<CameraMotion>,
  ) -> Vec<TrackedDetection> {
    let labels: Vec<String> = self.trackers.keys().cloned().collect();
    let mut output = Vec::new();
    for label in labels {
      output.extend(self.run_class(&label, Vec::new(), timestamp_ms, None, camera_motion));
    }
    output
  }

  /// Emit one event per (track, line) whose prev→current centroid segment
  /// crosses the line, firing each pair at most once.
  fn compute_crossings(
    &mut self,
    tracked: &[TrackedDetection],
    timestamp_ms: f64,
  ) -> Vec<LineCrossingEvent> {
    if self.prepared_lines.is_empty() || tracked.is_empty() {
      return Vec::new();
    }
    let mut events = Vec::new();

    for det in tracked {
      let class = match self.trackers.get(&det.label) {
        Some(c) => c,
        None => continue,
      };
      let prev = match class.prev_centroid_map.get(&det.track_id) {
        Some(&p) => p,
        None => continue, // first frame for this track — no segment yet
      };
      let curr_cx = det.x + det.width * 0.5;
      let curr_cy = det.y + det.height * 0.5;
      if (prev.0 - curr_cx).abs() < 1e-9 && (prev.1 - curr_cy).abs() < 1e-9 {
        continue;
      }

      let det_label_lc = det.label.to_lowercase();
      for line in &self.prepared_lines {
        if !line.labels.is_empty() && !line.labels.contains(&det_label_lc) {
          continue;
        }
        let memory_key = (det.track_id, line.name.clone());
        if self.crossing_memory.contains(&memory_key) {
          continue;
        }

        let cross = segment_intersection(
          prev.0,
          prev.1,
          curr_cx,
          curr_cy,
          line.line_a[0],
          line.line_a[1],
          line.line_b[0],
          line.line_b[1],
        );
        if cross == 0.0 {
          continue;
        }

        let direction = if cross > 0.0 {
          CrossingDirection::AToB
        } else {
          CrossingDirection::BToA
        };
        let allowed = match line.direction {
          LineDirectionFilter::Both => true,
          LineDirectionFilter::AToB => direction == CrossingDirection::AToB,
          LineDirectionFilter::BToA => direction == CrossingDirection::BToA,
        };
        if !allowed {
          continue;
        }

        self.crossing_memory.insert(memory_key);
        events.push(LineCrossingEvent {
          line_name: line.name.clone(),
          direction,
          track_id: det.track_id,
          label: det.label.clone(),
          confidence: det.confidence,
          timestamp_ms,
          prev_pos: [prev.0, prev.1],
          curr_pos: [curr_cx, curr_cy],
        });
      }
    }

    events
  }

  /// Store each track's latest centroid for next frame's comparison.
  fn refresh_centroid_history(&mut self, tracked: &[TrackedDetection]) {
    let mut by_class: HashMap<String, Vec<(u32, f32, f32)>> = HashMap::new();
    for det in tracked {
      let cx = det.x + det.width * 0.5;
      let cy = det.y + det.height * 0.5;
      by_class
        .entry(det.label.clone())
        .or_default()
        .push((det.track_id, cx, cy));
    }
    for (label, entries) in by_class {
      if let Some(class) = self.trackers.get_mut(&label) {
        for (id, cx, cy) in entries {
          class.prev_centroid_map.insert(id, (cx, cy));
        }
      }
    }
  }

  /// Drop crossing memory for expired track ids to bound the set's growth.
  fn gc_crossing_memory(&mut self) {
    if self.crossing_memory.is_empty() {
      return;
    }
    let mut alive: HashSet<u32> = HashSet::new();
    for class in self.trackers.values() {
      for id in class.id_map.values() {
        alive.insert(*id);
      }
    }
    self
      .crossing_memory
      .retain(|(track_id, _)| alive.contains(track_id));
  }

  /// Run one frame through the sub-tracker for `label`, creating it if needed.
  fn run_class(
    &mut self,
    label: &str,
    detections: Vec<Detection>,
    _timestamp_ms: f64,
    frame: Option<(&[u8], u32, u32)>,
    camera_motion: Option<CameraMotion>,
  ) -> Vec<TrackedDetection> {
    let current_frame = self.frame_seq;

    if !self.trackers.contains_key(label) {
      let nf_config = self.build_nf_config();
      let tracker = match Tracker::new(nf_config) {
        Ok(t) => t,
        Err(_) => return Vec::new(),
      };
      self.trackers.insert(
        label.to_string(),
        ClassTracker {
          tracker,
          id_map: HashMap::new(),
          prev_centroid_map: HashMap::new(),
        },
      );
    }

    // All detections in a frame share one Arc<FrameTag> (one allocation).
    let tag: std::sync::Arc<dyn std::any::Any + Send + Sync> = std::sync::Arc::new(FrameTag {
      frame_seq: current_frame,
    });
    let nf_detections: Vec<NfDetection> = detections
      .iter()
      .filter_map(|det| {
        let x2 = det.x + det.width;
        let y2 = det.y + det.height;
        let points =
          DMatrix::from_row_slice(1, 4, &[det.x as f64, det.y as f64, x2 as f64, y2 as f64]);
        let mut nf = NfDetection::new(points).ok()?;
        nf.label = Some(det.label.clone());
        nf.scores = Some(vec![det.confidence as f64]);
        nf.data = Some(tag.clone());

        if let Some((pixels, img_w, img_h)) = frame {
          nf.embedding = embedding::compute_embedding(
            pixels,
            img_w,
            img_h,
            [det.x, det.y, det.width, det.height],
          );
        }

        Some(nf)
      })
      .collect();

    let class = self.trackers.get_mut(label).expect("inserted above");
    let transform = camera_motion.map(|m| TranslationTransformation::new([m.x, m.y]));
    let active = class.tracker.update(
      nf_detections,
      1,
      transform
        .as_ref()
        .map(|t| t as &dyn norfair_rs::CoordinateTransformation),
    );

    // Snapshot norfair tracked objects before releasing the borrow.
    struct Raw {
      norfair_global_id: i32,
      x1: f64,
      y1: f64,
      x2: f64,
      y2: f64,
      confidence: f32,
      age: u32,
      speed: f32,
      vx: f32,
      vy: f32,
      matched_this_frame: bool,
    }
    // `obj.estimate` stays in the relative (image) frame, so it's already in
    // normalized `[0, 1]` coords for both matched and extrapolated tracks.
    let raw: Vec<Raw> = active
      .into_iter()
      .filter_map(|obj| {
        let est = &obj.estimate;
        if est.ncols() < 4 || est.nrows() < 1 {
          return None;
        }
        let confidence = obj
          .last_detection
          .as_ref()
          .and_then(|d| d.scores.as_ref())
          .and_then(|s| s.first().copied())
          .unwrap_or(0.0) as f32;
        let vel = &obj.estimate_velocity;
        let (vx, vy, speed) = if vel.ncols() >= 4 && vel.nrows() >= 1 {
          let vcx = ((vel[(0, 0)] + vel[(0, 2)]) / 2.0) as f32;
          let vcy = ((vel[(0, 1)] + vel[(0, 3)]) / 2.0) as f32;
          (vcx, vcy, (vcx * vcx + vcy * vcy).sqrt())
        } else {
          (0.0, 0.0, 0.0)
        };
        let matched_this_frame = obj
          .last_detection
          .as_ref()
          .and_then(|d| d.data.as_ref())
          .and_then(|d| d.downcast_ref::<FrameTag>())
          .is_some_and(|tag| tag.frame_seq == current_frame);

        Some(Raw {
          norfair_global_id: obj.global_id,
          x1: est[(0, 0)],
          y1: est[(0, 1)],
          x2: est[(0, 2)],
          y2: est[(0, 3)],
          confidence,
          age: obj.age.max(0) as u32,
          speed,
          vx,
          vy,
          matched_this_frame,
        })
      })
      .collect();

    let label_owned = label.to_string();
    let mut output: Vec<TrackedDetection> = Vec::with_capacity(raw.len());
    for r in raw {
      let track_id = match class.id_map.get(&r.norfair_global_id) {
        Some(&id) => id,
        None => {
          let id = self.next_track_id;
          self.next_track_id = self.next_track_id.wrapping_add(1).max(1);
          class.id_map.insert(r.norfair_global_id, id);
          id
        }
      };
      let width = (r.x2 - r.x1).max(0.0) as f32;
      let height = (r.y2 - r.y1).max(0.0) as f32;

      output.push(TrackedDetection {
        x: r.x1 as f32,
        y: r.y1 as f32,
        width,
        height,
        confidence: r.confidence,
        label: label_owned.clone(),
        track_id,
        track_age: r.age,
        track_lost: !r.matched_this_frame,
        track_speed: r.speed,
        track_velocity_x: r.vx,
        track_velocity_y: r.vy,
      });
    }

    // Drop remap entries for tracks gone from norfair's store. ReID-phase
    // tracks remain in tracked_objects, so their id_map entries survive and
    // the old track_id is preserved when norfair later merges them back.
    let norfair_alive: HashSet<i32> = class
      .tracker
      .tracked_objects
      .iter()
      .map(|o| o.global_id)
      .collect();
    class
      .id_map
      .retain(|nf_id, _| norfair_alive.contains(nf_id));
    let mapped_ids: HashSet<u32> = class.id_map.values().copied().collect();
    class
      .prev_centroid_map
      .retain(|k, _| mapped_ids.contains(k));

    output
  }

  fn build_nf_config(&self) -> TrackerConfig {
    // norfair's IoU distance is 1 - IoU, so the match threshold is inverted.
    let distance_threshold = (1.0 - self.config.iou_threshold).max(0.0) as f64;
    let mut cfg = TrackerConfig::new(distance_function_by_name("iou"), distance_threshold);
    cfg.hit_counter_max = self.config.hit_counter_max;
    cfg.initialization_delay = self.config.initialization_delay;

    // ReID via embedding cosine distance, falling back to IoU when either
    // side has no embedding (e.g. no frame was provided).
    if let Some(reid_max) = self.config.reid_hit_counter_max {
      let reid_distance = ScalarDistance::new(
        move |candidate: &NfDetection, dead_track: &TrackedObject| -> f64 {
          let cand_emb = candidate.embedding.as_ref();
          let track_emb = dead_track
            .past_detections
            .iter()
            .rev()
            .find_map(|d| d.embedding.as_ref());

          match (cand_emb, track_emb) {
            (Some(a), Some(b)) => embedding::embedding_distance(a, b),
            _ => {
              let cand_pts = &candidate.points;
              let track_est = &dead_track.estimate;
              if cand_pts.ncols() >= 4 && track_est.ncols() >= 4 {
                let a = [
                  cand_pts[(0, 0)] as f32,
                  cand_pts[(0, 1)] as f32,
                  cand_pts[(0, 2)] as f32,
                  cand_pts[(0, 3)] as f32,
                ];
                let b = [
                  track_est[(0, 0)] as f32,
                  track_est[(0, 1)] as f32,
                  track_est[(0, 2)] as f32,
                  track_est[(0, 3)] as f32,
                ];
                let iou = crate::iou::box_iou(&a, &b);
                (1.0 - iou) as f64
              } else {
                f64::INFINITY
              }
            }
          }
        },
      );

      cfg.reid_distance_function = Some(DistanceFunction::Frobenius(reid_distance));
      cfg.reid_distance_threshold = self.config.reid_embedding_threshold;
      cfg.reid_hit_counter_max = Some(reid_max);
      cfg.past_detections_length = 3;
    }

    cfg
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn det(x: f32, y: f32, w: f32, h: f32, label: &str) -> Detection {
    Detection {
      x,
      y,
      width: w,
      height: h,
      confidence: 0.9,
      label: label.to_string(),
    }
  }

  #[test]
  fn assigns_ids_to_new_detections() {
    let mut t = ObjectTracker::new(ObjectTrackerConfig {
      hit_counter_max: 5,
      initialization_delay: 0,
      ..Default::default()
    });
    let res = t.update(vec![det(0.1, 0.1, 0.2, 0.2, "person")], 0.0, None, None);
    assert!(!res.tracked.is_empty());
    assert!(res.tracked[0].track_id >= 1);
  }

  #[test]
  fn maintains_id_across_frames() {
    let mut t = ObjectTracker::new(ObjectTrackerConfig {
      hit_counter_max: 5,
      initialization_delay: 0,
      ..Default::default()
    });
    let res1 = t.update(vec![det(0.1, 0.1, 0.2, 0.2, "person")], 0.0, None, None);
    let res2 = t.update(vec![det(0.11, 0.11, 0.2, 0.2, "person")], 1.0, None, None);
    if !res1.tracked.is_empty() && !res2.tracked.is_empty() {
      assert_eq!(res1.tracked[0].track_id, res2.tracked[0].track_id);
    }
  }

  #[test]
  fn different_classes_get_different_ids() {
    let mut t = ObjectTracker::new(ObjectTrackerConfig {
      hit_counter_max: 5,
      initialization_delay: 0,
      ..Default::default()
    });
    let res = t.update(
      vec![
        det(0.1, 0.1, 0.2, 0.2, "person"),
        det(0.5, 0.5, 0.2, 0.2, "car"),
      ],
      0.0,
      None,
      None,
    );
    let ids: std::collections::HashSet<u32> = res.tracked.iter().map(|d| d.track_id).collect();
    assert_eq!(ids.len(), 2);
  }

  #[test]
  fn matched_frame_not_lost() {
    let mut t = ObjectTracker::new(ObjectTrackerConfig {
      hit_counter_max: 5,
      initialization_delay: 0,
      ..Default::default()
    });
    let r1 = t.update(vec![det(0.1, 0.1, 0.2, 0.2, "person")], 0.0, None, None);
    let r2 = t.update(vec![det(0.11, 0.11, 0.2, 0.2, "person")], 1.0, None, None);
    let r3 = t.update(vec![det(0.12, 0.12, 0.2, 0.2, "person")], 2.0, None, None);
    assert!(
      !r1.tracked.is_empty() && !r1.tracked[0].track_lost,
      "first frame must be matched"
    );
    assert!(
      !r2.tracked.is_empty() && !r2.tracked[0].track_lost,
      "second matched frame"
    );
    assert!(
      !r3.tracked.is_empty() && !r3.tracked[0].track_lost,
      "third matched frame"
    );
  }

  #[test]
  fn unmatched_frame_marked_lost() {
    let mut t = ObjectTracker::new(ObjectTrackerConfig {
      hit_counter_max: 5,
      initialization_delay: 0,
      ..Default::default()
    });
    let _ = t.update(vec![det(0.1, 0.1, 0.2, 0.2, "person")], 0.0, None, None);
    let _ = t.update(vec![det(0.11, 0.11, 0.2, 0.2, "person")], 1.0, None, None);
    let res = t.update(Vec::new(), 2.0, None, None);
    if !res.tracked.is_empty() {
      assert!(res.tracked[0].track_lost, "unmatched frame must be lost");
    }
  }

  #[test]
  fn reset_clears_state() {
    let mut t = ObjectTracker::new(ObjectTrackerConfig {
      hit_counter_max: 5,
      initialization_delay: 0,
      ..Default::default()
    });
    t.update(vec![det(0.1, 0.1, 0.2, 0.2, "person")], 0.0, None, None);
    assert!(t.track_count() >= 1);
    t.reset();
    assert_eq!(t.track_count(), 0);
  }

  #[test]
  fn line_crossing_a_to_b() {
    let mut t = ObjectTracker::new(ObjectTrackerConfig {
      hit_counter_max: 5,
      initialization_delay: 0,
      ..Default::default()
    });
    t.set_lines(
      vec![DetectionLineInput {
        name: "gate".to_string(),
        direction: LineDirectionFilter::Both,
        labels: vec![],
        points: [[30.0, 50.0], [70.0, 50.0]],
      }],
      16.0 / 9.0,
    );

    let r1 = t.update(vec![det(0.35, 0.40, 0.20, 0.20, "person")], 0.0, None, None);
    assert_eq!(r1.crossings.len(), 0, "first frame: no prev → no crossing");
    let r2 = t.update(vec![det(0.45, 0.40, 0.20, 0.20, "person")], 1.0, None, None);
    assert_eq!(r2.crossings.len(), 1, "movement should fire one crossing");
    let crossing = &r2.crossings[0];
    assert_eq!(crossing.line_name, "gate");
    assert_eq!(crossing.label, "person");
  }

  #[test]
  fn line_crossing_label_filter() {
    let mut t = ObjectTracker::new(ObjectTrackerConfig {
      hit_counter_max: 5,
      initialization_delay: 0,
      ..Default::default()
    });
    t.set_lines(
      vec![DetectionLineInput {
        name: "vehicle-only".to_string(),
        direction: LineDirectionFilter::Both,
        labels: vec!["car".to_string()],
        points: [[30.0, 50.0], [70.0, 50.0]],
      }],
      16.0 / 9.0,
    );
    let _ = t.update(vec![det(0.35, 0.40, 0.20, 0.20, "person")], 0.0, None, None);
    let r2 = t.update(vec![det(0.45, 0.40, 0.20, 0.20, "person")], 1.0, None, None);
    assert_eq!(
      r2.crossings.len(),
      0,
      "person should not match vehicle-only line"
    );
  }

  #[test]
  fn track_speed_static_then_moving() {
    let mut t = ObjectTracker::new(ObjectTrackerConfig {
      hit_counter_max: 5,
      initialization_delay: 0,
      ..Default::default()
    });
    let r1 = t.update(vec![det(0.10, 0.10, 0.20, 0.20, "person")], 0.0, None, None);
    assert_eq!(r1.tracked.len(), 1);
    assert!(
      r1.tracked[0].track_speed < 0.01,
      "first frame speed should be ~0"
    );

    let r2 = t.update(
      vec![det(0.10, 0.10, 0.20, 0.20, "person")],
      100.0,
      None,
      None,
    );
    assert!(
      r2.tracked[0].track_speed < 0.01,
      "static frame speed should be ~0"
    );

    let mut last_speed = 0.0f32;
    for i in 2..8 {
      let x = 0.10 + (i - 1) as f32 * 0.05;
      let r = t.update(
        vec![det(x, 0.10, 0.20, 0.20, "person")],
        (i * 100) as f64,
        None,
        None,
      );
      last_speed = r.tracked[0].track_speed;
    }
    assert!(
      last_speed > 0.001,
      "expected Kalman velocity > 0 after sustained movement, got {}",
      last_speed
    );
  }

  #[test]
  fn tracked_detection_carries_input_confidence() {
    let mut t = ObjectTracker::new(ObjectTrackerConfig {
      hit_counter_max: 5,
      initialization_delay: 0,
      ..Default::default()
    });
    let mut d = det(0.10, 0.10, 0.20, 0.20, "person");
    d.confidence = 0.87;
    let r = t.update(vec![d], 0.0, None, None);
    assert!(!r.tracked.is_empty());
    assert!(
      (r.tracked[0].confidence - 0.87).abs() < 1e-3,
      "expected ~0.87, got {}",
      r.tracked[0].confidence
    );
  }

  #[test]
  fn confidence_threshold_drops_low_score() {
    let mut t = ObjectTracker::new(ObjectTrackerConfig {
      hit_counter_max: 5,
      initialization_delay: 0,
      ..Default::default()
    });
    t.set_min_confidence(0.5);
    let mut low = det(0.1, 0.1, 0.2, 0.2, "person");
    low.confidence = 0.4;
    let mut high = det(0.5, 0.5, 0.2, 0.2, "person");
    high.confidence = 0.8;
    let res = t.update(vec![low, high], 0.0, None, None);
    assert_eq!(res.tracked.len(), 1);
    assert!((res.tracked[0].x - 0.5).abs() < 1e-6);
  }

  #[test]
  fn zone_exclude_drops_detection_in_zone() {
    use crate::zone_filter::{ZoneFilterMode, ZoneInput, ZoneMatchType};
    let mut t = ObjectTracker::new(ObjectTrackerConfig {
      hit_counter_max: 5,
      initialization_delay: 0,
      ..Default::default()
    });
    t.set_zones(vec![ZoneInput {
      labels: vec![],
      filter: ZoneFilterMode::Exclude,
      match_type: ZoneMatchType::Intersect,
      is_privacy_mask: false,
      points: vec![[0.0, 0.0], [50.0, 0.0], [50.0, 50.0], [0.0, 50.0]],
    }]);
    let inside = det(0.10, 0.10, 0.20, 0.20, "person");
    let outside = det(0.60, 0.60, 0.20, 0.20, "person");
    let res = t.update(vec![inside, outside], 0.0, None, None);
    assert_eq!(res.tracked.len(), 1);
    assert!(res.tracked[0].x > 0.5);
  }

  #[test]
  fn line_crossing_only_fires_once_per_track() {
    let mut t = ObjectTracker::new(ObjectTrackerConfig {
      hit_counter_max: 5,
      initialization_delay: 0,
      ..Default::default()
    });
    t.set_lines(
      vec![DetectionLineInput {
        name: "gate".to_string(),
        direction: LineDirectionFilter::Both,
        labels: vec![],
        points: [[30.0, 50.0], [70.0, 50.0]],
      }],
      16.0 / 9.0,
    );
    let _ = t.update(vec![det(0.35, 0.40, 0.20, 0.20, "person")], 0.0, None, None);
    let r2 = t.update(vec![det(0.45, 0.40, 0.20, 0.20, "person")], 1.0, None, None);
    let r3 = t.update(vec![det(0.55, 0.40, 0.20, 0.20, "person")], 2.0, None, None);
    assert_eq!(r2.crossings.len(), 1);
    assert_eq!(r3.crossings.len(), 0, "memory should suppress repeat");
  }

  #[test]
  fn track_expires_without_reid() {
    let mut t = ObjectTracker::new(ObjectTrackerConfig {
      hit_counter_max: 3,
      initialization_delay: 0,
      ..Default::default()
    });
    let _ = t.update(vec![det(0.1, 0.1, 0.2, 0.2, "person")], 0.0, None, None);
    let _ = t.update(vec![det(0.11, 0.11, 0.2, 0.2, "person")], 100.0, None, None);

    let mut alive_count = 0;
    for i in 2..20 {
      let r = t.update(Vec::new(), (i * 100) as f64, None, None);
      if !r.tracked.is_empty() {
        alive_count += 1;
      }
    }
    assert!(
      alive_count < 18,
      "without ReID, track should expire (alive for {} frames)",
      alive_count
    );
  }

  #[test]
  fn reid_preserves_id_after_gap() {
    // initialization_delay=1 is required: norfair only ReID-matches dead
    // tracks against newly-matched, still-initializing objects.
    let mut t = ObjectTracker::new(ObjectTrackerConfig {
      hit_counter_max: 3,
      initialization_delay: 1,
      reid_hit_counter_max: Some(20),
      ..Default::default()
    });

    let _ = t.update(vec![det(0.30, 0.30, 0.20, 0.20, "person")], 0.0, None, None);
    let r2 = t.update(
      vec![det(0.30, 0.30, 0.20, 0.20, "person")],
      100.0,
      None,
      None,
    );
    assert_eq!(r2.tracked.len(), 1);
    let id = r2.tracked[0].track_id;
    let _ = t.update(
      vec![det(0.30, 0.30, 0.20, 0.20, "person")],
      200.0,
      None,
      None,
    );

    // Disappear long enough to die but stay within the ReID window.
    for i in 3..13 {
      t.update(Vec::new(), (i * 100) as f64, None, None);
    }

    let _ = t.update(
      vec![det(0.30, 0.30, 0.20, 0.20, "person")],
      1300.0,
      None,
      None,
    );
    let r_back = t.update(
      vec![det(0.30, 0.30, 0.20, 0.20, "person")],
      1400.0,
      None,
      None,
    );

    let found = r_back.tracked.iter().any(|t| t.track_id == id);
    assert!(
      found,
      "ReID should restore old track id {} (got: {:?})",
      id,
      r_back
        .tracked
        .iter()
        .map(|t| t.track_id)
        .collect::<Vec<_>>()
    );
  }

  #[test]
  fn reid_expires_after_reid_hit_counter_max() {
    let mut t = ObjectTracker::new(ObjectTrackerConfig {
      hit_counter_max: 3,
      initialization_delay: 1,
      reid_hit_counter_max: Some(5),
      ..Default::default()
    });

    let _ = t.update(vec![det(0.30, 0.30, 0.20, 0.20, "person")], 0.0, None, None);
    let r2 = t.update(
      vec![det(0.30, 0.30, 0.20, 0.20, "person")],
      100.0,
      None,
      None,
    );
    let id = r2.tracked[0].track_id;

    // Die and exhaust the ReID window.
    for i in 2..20 {
      t.update(Vec::new(), (i * 100) as f64, None, None);
    }

    let _ = t.update(
      vec![det(0.30, 0.30, 0.20, 0.20, "person")],
      2000.0,
      None,
      None,
    );
    let r_back = t.update(
      vec![det(0.30, 0.30, 0.20, 0.20, "person")],
      2100.0,
      None,
      None,
    );

    let has_old = r_back.tracked.iter().any(|t| t.track_id == id);
    assert!(
      !has_old,
      "ReID should have expired — old id {} should not appear",
      id
    );
  }

  #[test]
  fn set_reid_hit_counter_max_dynamically() {
    let mut t = ObjectTracker::new(ObjectTrackerConfig {
      hit_counter_max: 3,
      initialization_delay: 1,
      reid_hit_counter_max: None,
      ..Default::default()
    });

    t.set_reid_hit_counter_max(50);

    let _ = t.update(vec![det(0.30, 0.30, 0.20, 0.20, "person")], 0.0, None, None);
    let r2 = t.update(
      vec![det(0.30, 0.30, 0.20, 0.20, "person")],
      100.0,
      None,
      None,
    );
    let id = r2.tracked[0].track_id;
    let _ = t.update(
      vec![det(0.30, 0.30, 0.20, 0.20, "person")],
      200.0,
      None,
      None,
    );

    for i in 3..13 {
      t.update(Vec::new(), (i * 100) as f64, None, None);
    }

    let _ = t.update(
      vec![det(0.30, 0.30, 0.20, 0.20, "person")],
      1300.0,
      None,
      None,
    );
    let r_back = t.update(
      vec![det(0.30, 0.30, 0.20, 0.20, "person")],
      1400.0,
      None,
      None,
    );

    let found = r_back.tracked.iter().any(|t| t.track_id == id);
    assert!(found, "Dynamic ReID should restore old track id {}", id);

    t.set_reid_hit_counter_max(0);
  }
}
