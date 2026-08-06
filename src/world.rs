use std::collections::{HashMap, HashSet};

use trackforge::trackers::byte_track::ByteTrack;

use crate::iou::box_iou;
use crate::line_crossing::{
  prepare_lines, segment_intersection, CrossingDirection, DetectionLineInput, LineCrossingEvent,
  LineDirectionFilter, PreparedLine,
};
use crate::semantic::{SemanticEvent, TrackSnapshot, TrackState};
use crate::types::Detection;
use crate::zone_filter::{filter_indices, prepare_zones, PreparedZones, ZoneInput};

// (world id, label, prev centroid, curr centroid) of one tick's movement
type TrackSegment = (u32, String, (f32, f32), (f32, f32));

const ASSOCIATION_FLOOR: f32 = 0.1;

pub struct WorldUpdate {
  pub tracked: Vec<TrackSnapshot>,
  pub created: Vec<u32>,
  pub removed: Vec<u32>,
  pub events: Vec<SemanticEvent>,
  pub crossings: Vec<LineCrossingEvent>,
}

pub struct WorldConfig {
  pub gap_reset_ms: f64,
  pub depart_grace_ms: f64,
  pub settle_default_ms: f64,
  pub settle_vehicle_ms: f64,
  pub settle_person_ms: f64,
  pub stationary_speed: f32,
  pub reassoc_iou: f32,
  pub wake_ticks: u32,
  pub confirm_ms: f64,
  pub max_dormant: usize,
}

impl Default for WorldConfig {
  fn default() -> Self {
    Self {
      gap_reset_ms: 5_000.0,
      depart_grace_ms: 5_000.0,
      settle_default_ms: 10_000.0,
      settle_vehicle_ms: 8_000.0,
      settle_person_ms: 120_000.0,
      stationary_speed: 0.002,
      reassoc_iou: 0.3,
      wake_ticks: 3,
      confirm_ms: 500.0,
      max_dormant: 50,
    }
  }
}

struct WorldTrack {
  label: String,
  bbox: [f32; 4],
  // last box while stationary; a stolen engine box must not drag the anchor
  anchor: [f32; 4],
  confidence: f32,
  speed: f32,
  velocity: (f32, f32),
  state: TrackState,
  still_since_ms: f64,
  last_seen_ms: f64,
  moving_streak: u32,
  first_seen_ms: f64,
  best_shot_score: f32,
  dormant: bool,
}

pub struct CameraWorld {
  config: WorldConfig,
  engine: ByteTrack,
  tracks: HashMap<u32, WorldTrack>,
  engine_map: HashMap<u32, u32>,
  next_id: u32,
  last_t_ms: f64,
  prepared_lines: Vec<PreparedLine>,
  crossing_memory: HashSet<(u32, u32)>,
  prepared_zones: PreparedZones,
  min_confidence: f32,
  // class ids live as long as the engine hands out tracks that carry them
  labels: Vec<String>,
  label_ids: HashMap<String, i64>,
  pose_baseline: Option<(f32, f32)>,
}

impl CameraWorld {
  pub fn new(config: WorldConfig) -> Self {
    Self {
      config,
      engine: new_engine(),
      tracks: HashMap::new(),
      engine_map: HashMap::new(),
      next_id: 1,
      last_t_ms: 0.0,
      prepared_lines: Vec::new(),
      crossing_memory: HashSet::new(),
      prepared_zones: PreparedZones::default(),
      min_confidence: 0.0,
      labels: Vec::new(),
      label_ids: HashMap::new(),
      pose_baseline: None,
    }
  }

  pub fn set_zones(&mut self, zones: Vec<ZoneInput>) {
    self.prepared_zones = prepare_zones(&zones);
  }

  pub fn set_min_confidence(&mut self, min_confidence: f32) {
    self.min_confidence = min_confidence.max(0.0);
  }

  pub fn filter_indices(&self, detections: &[Detection]) -> Vec<u32> {
    filter_indices(detections, &self.prepared_zones, self.min_confidence)
  }

  pub fn set_lines(&mut self, lines: Vec<DetectionLineInput>, aspect_ratio: f32) {
    self.prepared_lines = prepare_lines(&lines, aspect_ratio);
    self.crossing_memory.clear();
  }

  pub fn notify_camera_move(&mut self) {
    self
      .tracks
      .retain(|_, track| !track.dormant && track.state != TrackState::Stationary);
    self
      .engine_map
      .retain(|_, world_id| self.tracks.contains_key(world_id));
    self.crossing_memory.clear();
    self.pose_baseline = None;
  }

  pub fn ingest(
    &mut self,
    t_ms: f64,
    detections: &[Detection],
    camera_motion: Option<(f32, f32)>,
  ) -> WorldUpdate {
    let mut events = Vec::new();
    let mut created = Vec::new();

    if self.last_t_ms != 0.0 && t_ms - self.last_t_ms > self.config.gap_reset_ms {
      // a decode gap is not evidence anything left: engine restarts, world
      // tracks sleep and wait for re-association; an unconfirmed flicker has
      // no identity to preserve and must not become a re-association anchor
      self.engine = new_engine();
      self.engine_map.clear();
      self.tracks.retain(|_, t| t.state != TrackState::Tentative);
      for track in self.tracks.values_mut() {
        track.dormant = true;
        track.moving_streak = 0;
      }
      self.evict_dormant();
    }
    self.last_t_ms = t_ms;

    // the caller reports an absolute pose offset, coarse and step-wise — too
    // coarse to stabilize engine coordinates (measured: it breaks association).
    // The engine works on raw image coords; stored positions shift by the
    // offset change so speed and anchors stay pan-clean
    if let Some((mx, my)) = camera_motion {
      let (lx, ly) = *self.pose_baseline.get_or_insert((mx, my));
      let (dx, dy) = (mx - lx, my - ly);
      if dx != 0.0 || dy != 0.0 {
        for track in self.tracks.values_mut() {
          track.bbox[0] += dx;
          track.bbox[1] += dy;
          track.anchor[0] += dx;
          track.anchor[1] += dy;
        }
      }
      self.pose_baseline = Some((mx, my));
    }

    // the engine gets everything down to the association floor — ByteTrack's
    // second pass continues established tracks through low-confidence
    // stretches; the user threshold gates track BIRTH below, not association
    let kept = filter_indices(detections, &self.prepared_zones, ASSOCIATION_FLOOR);
    let detections: Vec<&Detection> = kept.iter().map(|&i| &detections[i as usize]).collect();

    let detections: Vec<([f32; 4], f32, i64)> = detections
      .iter()
      .map(|d| {
        let id = match self.label_ids.get(&d.label) {
          Some(&id) => id,
          None => {
            let id = self.labels.len() as i64;
            self.labels.push(d.label.clone());
            self.label_ids.insert(d.label.clone(), id);
            id
          }
        };
        ([d.x, d.y, d.width, d.height], d.confidence, id)
      })
      .collect();

    let mut seen: Vec<u32> = Vec::new();
    let mut segments: Vec<TrackSegment> = Vec::new();
    for t in self.engine.update(detections) {
      if !t.is_activated {
        continue;
      }
      let engine_id = t.track_id as u32;
      let Some(label) = self.labels.get(t.class_id as usize).cloned() else {
        continue;
      };
      let bbox = t.tlwh;

      let world_id = match self.engine_map.get(&engine_id) {
        Some(id) if self.tracks.contains_key(id) => *id,
        _ => {
          // dormant resume is continuation and allowed at any score; a NEW
          // identity needs the user threshold at least once
          let reassociated = self.reassociate(&label, &bbox);
          if reassociated.is_none() && t.score < self.min_confidence {
            continue;
          }
          let id = reassociated.unwrap_or_else(|| {
            let id = self.next_id;
            self.next_id += 1;
            self.tracks.insert(
              id,
              WorldTrack {
                label: label.clone(),
                bbox,
                anchor: bbox,
                confidence: t.score,
                speed: 0.0,
                velocity: (0.0, 0.0),
                state: TrackState::Tentative,
                still_since_ms: t_ms,
                last_seen_ms: t_ms,
                moving_streak: 0,
                first_seen_ms: t_ms,
                best_shot_score: 0.0,
                dormant: false,
              },
            );
            id
          });
          self.engine_map.insert(engine_id, id);
          id
        }
      };

      let confirm_ms = self.config.confirm_ms;
      let settle_ms = self.settle_ms(&label);
      let track = self.tracks.get_mut(&world_id).unwrap();
      if track.state == TrackState::Tentative {
        // an identity is persistence over time, independent of tick rate: a
        // single flicker never re-sights after the window and dies silently
        if t_ms - track.first_seen_ms < confirm_ms {
          track.bbox = bbox;
          track.anchor = bbox;
          track.confidence = t.score;
          track.last_seen_ms = t_ms;
          maybe_best_shot(world_id, track, &mut events);
          continue;
        }
        track.state = TrackState::Active;
        created.push(world_id);
        events.push(SemanticEvent::ObjectEntered(snapshot(world_id, track)));
      }
      let dt = (t_ms - track.last_seen_ms).max(1.0);
      let center = |b: &[f32; 4]| (b[0] + b[2] / 2.0, b[1] + b[3] / 2.0);
      let (cx, cy) = center(&bbox);
      let (px, py) = center(&track.bbox);
      let was_dormant = track.dormant;
      if was_dormant {
        track.speed = 0.0;
        track.velocity = (0.0, 0.0);
      } else {
        // EMA over signed components so detector box jitter cancels out
        // instead of reading as perpetual movement; speed is the magnitude
        // of the averaged vector, like a Kalman velocity would give
        let scale = (200.0_f32 / dt as f32).min(1.0);
        let (dx, dy) = ((cx - px) * scale, (cy - py) * scale);
        track.velocity = (
          track.velocity.0 * 0.6 + dx * 0.4,
          track.velocity.1 * 0.6 + dy * 0.4,
        );
        track.speed = (track.velocity.0.powi(2) + track.velocity.1.powi(2)).sqrt();
      }
      track.dormant = false;
      // a segment across a decode gap is not a movement, it is missing time
      if !was_dormant {
        segments.push((world_id, label.clone(), (px, py), (cx, cy)));
      } else {
        // resuming from sleep re-opens visibility; the snapshot carries the
        // state, consumers decide whether stationary scenery counts
        events.push(SemanticEvent::ObjectRecovered(snapshot(world_id, track)));
      }
      track.confidence = t.score;
      track.bbox = bbox;
      track.last_seen_ms = t_ms;

      let moving = track.speed >= self.config.stationary_speed;
      if track.state == TrackState::Stationary {
        let anchor_iou = box_iou(&track.anchor, &bbox);
        // waking needs the box to have left the anchor, not merely deformed:
        // an occluder shifting the visible box must not wake a parked object
        let departed_anchor = anchor_iou < 0.15;
        if moving && departed_anchor {
          track.moving_streak += 1;
          if track.moving_streak >= self.config.wake_ticks {
            track.state = TrackState::Active;
            track.still_since_ms = t_ms;
            track.moving_streak = 0;
            events.push(SemanticEvent::ObjectWoke(snapshot(world_id, track)));
          }
        } else {
          track.moving_streak = 0;
          // drift correction only at rest, so a departing box can't drag the anchor along
          if !moving && anchor_iou >= 0.6 {
            track.anchor = bbox;
          }
        }
      } else {
        // stillness is position truth, not velocity noise: the box of a still
        // object keeps overlapping its anchor (IoU-relative, so box size and
        // jitter scale out); anything actually going somewhere re-anchors
        // constantly and never accumulates still time
        if box_iou(&track.anchor, &bbox) < 0.6 {
          track.anchor = bbox;
          track.still_since_ms = t_ms;
        }
        if t_ms - track.still_since_ms >= settle_ms {
          track.state = TrackState::Stationary;
          track.anchor = bbox;
          events.push(SemanticEvent::ObjectSettled(snapshot(world_id, track)));
        }
      }
      maybe_best_shot(world_id, track, &mut events);
      seen.push(world_id);
    }

    let crossings = self.compute_crossings(&segments, t_ms);

    let mut removed = Vec::new();
    let mut removed_quiet = Vec::new();
    let depart_grace_ms = self.config.depart_grace_ms;
    self.tracks.retain(|id, track| {
      if seen.contains(id) || track.dormant {
        return true;
      }
      if track.state == TrackState::Tentative {
        // an unconfirmed flicker dies silently
        if t_ms - track.last_seen_ms > depart_grace_ms {
          removed_quiet.push(*id);
          return false;
        }
        return true;
      }
      if t_ms - track.last_seen_ms > depart_grace_ms {
        // only something that was going somewhere can leave; a still object
        // (anchor held at least as long as the leaver grace) sleeps instead
        let was_still = track.still_since_ms == track.first_seen_ms
          || track.last_seen_ms - track.still_since_ms >= depart_grace_ms;
        if was_still || track.state == TrackState::Stationary {
          track.dormant = true;
          track.moving_streak = 0;
          // the span consumer needs the silent sleep too: a track fading out
          // without departing still ends its visibility
          events.push(SemanticEvent::ObjectLost(snapshot(*id, track)));
        } else {
          let mut gone = snapshot(*id, track);
          gone.state = TrackState::Departed;
          events.push(SemanticEvent::ObjectDeparted(gone));
          removed.push(*id);
          return false;
        }
      }
      true
    });
    if !removed.is_empty() {
      self
        .engine_map
        .retain(|_, world_id| !removed.contains(world_id));
      self.crossing_memory.retain(|(id, _)| !removed.contains(id));
    }

    let tracked = seen
      .iter()
      .filter_map(|id| {
        self
          .tracks
          .get(id)
          .filter(|t| t.state != TrackState::Tentative)
          .map(|t| snapshot(*id, t))
      })
      .collect();

    WorldUpdate {
      tracked,
      created,
      removed,
      events,
      crossings,
    }
  }

  fn compute_crossings(&mut self, segments: &[TrackSegment], t_ms: f64) -> Vec<LineCrossingEvent> {
    if self.prepared_lines.is_empty() || segments.is_empty() {
      return Vec::new();
    }
    let mut events = Vec::new();
    for (world_id, label, prev, curr) in segments {
      let prev = (prev.0, prev.1);
      let curr = (curr.0, curr.1);
      if (prev.0 - curr.0).abs() < 1e-9 && (prev.1 - curr.1).abs() < 1e-9 {
        continue;
      }
      let label_lc = label.to_lowercase();
      for (line_idx, line) in self.prepared_lines.iter().enumerate() {
        if !line.labels.is_empty() && !line.labels.contains(&label_lc) {
          continue;
        }
        let memory_key = (*world_id, line_idx as u32);
        if self.crossing_memory.contains(&memory_key) {
          continue;
        }
        let cross = segment_intersection(
          prev.0,
          prev.1,
          curr.0,
          curr.1,
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
          track_id: *world_id,
          label: label.clone(),
          confidence: self
            .tracks
            .get(world_id)
            .map(|t| t.confidence)
            .unwrap_or(0.0),
          timestamp_ms: t_ms,
          prev_pos: [prev.0, prev.1],
          curr_pos: [curr.0, curr.1],
        });
      }
    }
    events
  }

  fn reassociate(&self, label: &str, bbox: &[f32; 4]) -> Option<u32> {
    let mut best: Option<(u32, f32)> = None;
    for (id, track) in &self.tracks {
      if !track.dormant || track.label != label {
        continue;
      }
      let iou = box_iou(&track.anchor, bbox);
      if iou >= self.config.reassoc_iou && best.is_none_or(|(_, b)| iou > b) {
        best = Some((*id, iou));
      }
    }
    best.map(|(id, _)| id)
  }

  fn settle_ms(&self, label: &str) -> f64 {
    match label {
      "vehicle" => self.config.settle_vehicle_ms,
      "person" => self.config.settle_person_ms,
      _ => self.config.settle_default_ms,
    }
  }

  fn evict_dormant(&mut self) {
    while self.tracks.values().filter(|t| t.dormant).count() > self.config.max_dormant {
      let oldest = self
        .tracks
        .iter()
        .filter(|(_, t)| t.dormant)
        .min_by(|a, b| a.1.last_seen_ms.total_cmp(&b.1.last_seen_ms))
        .map(|(id, _)| *id);
      match oldest {
        Some(id) => self.tracks.remove(&id),
        None => break,
      };
    }
  }
}

fn maybe_best_shot(world_id: u32, track: &mut WorldTrack, events: &mut Vec<SemanticEvent>) {
  let b = &track.bbox;
  let clipped = b[0] <= 0.005 || b[1] <= 0.005 || b[0] + b[2] >= 0.995 || b[1] + b[3] >= 0.995;
  let edge_penalty = if clipped { 0.3 } else { 1.0 };
  let score = track.confidence * (b[2] * b[3]).sqrt() * edge_penalty;
  if score > track.best_shot_score * 1.25 {
    track.best_shot_score = score;
    events.push(SemanticEvent::BestShotUpdated(snapshot(world_id, track)));
  }
}

fn new_engine() -> ByteTrack {
  ByteTrack::new(0.5, 30, 0.9, 0.5)
}

fn snapshot(id: u32, t: &WorldTrack) -> TrackSnapshot {
  TrackSnapshot {
    track_id: id,
    label: t.label.clone(),
    x: t.bbox[0],
    y: t.bbox[1],
    width: t.bbox[2],
    height: t.bbox[3],
    confidence: t.confidence,
    speed: t.speed,
    velocity_x: t.velocity.0,
    velocity_y: t.velocity.1,
    state: t.state,
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn person(x: f32) -> Detection {
    Detection {
      x,
      y: 0.4,
      width: 0.1,
      height: 0.2,
      confidence: 0.9,
      label: "person".to_string(),
    }
  }

  fn vertical_line_at(x_ui: f64, name: &str) -> DetectionLineInput {
    DetectionLineInput {
      name: name.to_string(),
      direction: LineDirectionFilter::Both,
      labels: Vec::new(),
      points: [[x_ui - 10.0, 50.0], [x_ui + 10.0, 50.0]],
    }
  }

  #[test]
  fn crossing_fires_once_per_track() {
    let mut world = CameraWorld::new(WorldConfig::default());
    world.set_lines(vec![vertical_line_at(50.0, "gate")], 1.0);
    let mut crossings = Vec::new();
    for i in 0..20 {
      let update = world.ingest(i as f64 * 200.0, &[person(0.2 + i as f32 * 0.03)], None);
      crossings.extend(update.crossings);
    }
    assert_eq!(crossings.len(), 1);
    assert_eq!(crossings[0].line_name, "gate");
  }

  #[test]
  fn no_crossing_across_decode_gap() {
    let mut world = CameraWorld::new(WorldConfig::default());
    world.set_lines(vec![vertical_line_at(37.0, "gap-line")], 1.0);
    for i in 0..10 {
      let update = world.ingest(i as f64 * 200.0, &[person(0.30)], None);
      assert!(update.crossings.is_empty());
    }
    let update = world.ingest(60_000.0, &[person(0.34)], None);
    assert!(
      update.created.is_empty(),
      "same spot re-appearance keeps its identity"
    );
    assert!(
      update.crossings.is_empty(),
      "a jump across a decode gap is not a movement"
    );
  }

  #[test]
  fn silent_sleep_emits_lost_and_resume_emits_recovered() {
    let mut world = CameraWorld::new(WorldConfig::default());
    for i in 0..8 {
      world.ingest(1_000.0 + i as f64 * 200.0, &[person(0.5)], None);
    }
    let mut lost = 0;
    let mut departed = 0;
    for i in 0..40 {
      let up = world.ingest(3_000.0 + i as f64 * 200.0, &[], None);
      lost += up
        .events
        .iter()
        .filter(|e| e.kind() == "objectLost")
        .count();
      departed += up.removed.len();
    }
    // never seen moving: fades into sleep, not a departure
    assert_eq!(lost, 1);
    assert_eq!(departed, 0);

    let up = world.ingest(12_000.0, &[person(0.5)], None);
    assert!(up.created.is_empty(), "resume keeps the identity");
    assert_eq!(
      up.events
        .iter()
        .filter(|e| e.kind() == "objectRecovered")
        .count(),
      1
    );
  }

  #[test]
  fn gap_drops_unconfirmed_flickers() {
    let mut world = CameraWorld::new(WorldConfig::default());
    world.ingest(1_000.0, &[person(0.5)], None);
    // a single sighting must not survive the gap as a dormant anchor: the
    // same-spot detection after the gap starts a fresh confirm window
    let update = world.ingest(11_000.0, &[person(0.5)], None);
    assert!(update.created.is_empty(), "no instant confirm after a gap");
    let update = world.ingest(11_100.0, &[person(0.5)], None);
    assert!(update.created.is_empty());
    let update = world.ingest(11_600.0, &[person(0.5)], None);
    assert_eq!(update.created.len(), 1);
  }

  #[test]
  fn identity_survives_a_pan() {
    let mut world = CameraWorld::new(WorldConfig::default());
    world.set_lines(vec![vertical_line_at(50.0, "pan-line")], 1.0);
    let mut created = 0;
    let mut crossings = 0;
    let mut woke = 0;
    for i in 0..60 {
      // camera pans right: image content shifts left in lockstep with the
      // reported absolute offset; the person is static in the world
      let shift = (i as f32 * 0.01).min(0.4);
      let update = world.ingest(
        i as f64 * 200.0,
        &[person(0.6 - shift)],
        Some((-shift, 0.0)),
      );
      created += update.created.len();
      crossings += update.crossings.len();
      woke += update
        .events
        .iter()
        .filter(|e| e.kind() == "objectWoke")
        .count();
      for t in &update.tracked {
        assert!(
          t.speed < 0.002,
          "pan must not read as movement, got {}",
          t.speed
        );
      }
    }
    assert!(created >= 1);
    assert_eq!(woke, 0, "a pan must never wake or move a compensated track");
    let _ = crossings;
  }

  #[test]
  fn best_shot_prefers_clean_frames_with_hysteresis() {
    let mut world = CameraWorld::new(WorldConfig::default());
    let mut shots = Vec::new();
    // enters clipped at the left edge, walks to mid-frame growing larger
    for i in 0..30 {
      let x = (i as f32 * 0.02 - 0.03).max(0.0);
      let w = 0.08 + i as f32 * 0.004;
      let d = Detection {
        x,
        y: 0.3,
        width: w,
        height: 0.25,
        confidence: 0.85,
        label: "person".to_string(),
      };
      let update = world.ingest(i as f64 * 200.0, &[d], None);
      for e in update.events {
        if e.kind() == "bestShotUpdated" {
          if let SemanticEvent::BestShotUpdated(s) = e {
            shots.push((i, s.x, s.width));
          }
        }
      }
    }
    assert!(
      !shots.is_empty(),
      "a shot exists from the tentative phase already"
    );
    assert!(
      shots[0].0 <= 2,
      "first candidate arrives with the first sightings"
    );
    assert!(
      shots.len() <= 6,
      "hysteresis bounds updates, got {}",
      shots.len()
    );
    let last = shots.last().unwrap();
    assert!(
      last.1 > 0.01,
      "the final best shot is not the edge-clipped entry"
    );
  }

  #[test]
  fn camera_move_clears_state() {
    let mut world = CameraWorld::new(WorldConfig::default());
    for i in 0..10 {
      world.ingest(i as f64 * 200.0, &[person(0.3)], None);
    }
    world.notify_camera_move();
    let mut created = 0;
    for i in 0..4 {
      let update = world.ingest(2_200.0 + i as f64 * 200.0, &[person(0.7)], None);
      created += update.created.len();
    }
    assert_eq!(created, 1, "after a camera move everything is new");
  }
}
