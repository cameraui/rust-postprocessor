use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use trackforge::trackers::byte_track::ByteTrack;

use crate::iou::box_iou;
use crate::line_crossing::{
  prepare_lines, segment_intersection, CrossingDirection, DetectionLineInput, LineCrossingEvent,
  LineDirectionFilter, PreparedLine,
};
use crate::semantic::{Handover, MatchTier, SemanticEvent, TrackSnapshot, TrackState, WakeCause};
use crate::types::Detection;
use crate::zone_filter::{filter_indices, prepare_zones, PreparedZones, ZoneInput};

// a box center and its size
type Sighting = ((f32, f32), (f32, f32));
// (world id, label, prev and curr sighting) of one tick's movement
type TrackSegment = (u32, String, Sighting, Sighting);

const ASSOCIATION_FLOOR: f32 = 0.1;
const ORPHAN_MS: f64 = 1_500.0;
const ATTEST_WINDOW_MS: f64 = 3_000.0;
const ENGINE_SPLIT: f32 = 0.5;
const DEPTH_WINDOW_MS: f64 = 1_000.0;
const DEPTH_STEP: f32 = 0.12;
const FAR_FROM_ANCHOR: f32 = 1.0;
// moving in place is weaker evidence than travel: it needs a detection clearly above the threshold
const SURE_MARGIN: f32 = 0.05;
// a center closer to a line than this share of its box's extent across it is on neither side
const LINE_DEAD_BAND: f32 = 0.1;

pub struct WorldUpdate {
  pub tracked: Vec<TrackSnapshot>,
  pub created: Vec<u32>,
  pub removed: Vec<u32>,
  pub events: Vec<SemanticEvent>,
  pub crossings: Vec<LineCrossingEvent>,
  pub sightings: Vec<TrackSnapshot>,
}

pub struct WorldConfig {
  pub gap_reset_ms: f64,
  pub depart_grace_ms: f64,
  pub still_lost_grace_ms: f64,
  pub settle_default_ms: f64,
  pub settle_vehicle_ms: f64,
  pub settle_person_ms: f64,
  pub stationary_speed: f32,
  pub reassoc_iou: f32,
  pub wake_ticks: u32,
  pub confirm_ms: f64,
  pub max_dormant: usize,
  pub dormant_animate_ms: f64,
}

impl Default for WorldConfig {
  fn default() -> Self {
    Self {
      gap_reset_ms: 5_000.0,
      depart_grace_ms: 5_000.0,
      still_lost_grace_ms: 60_000.0,
      settle_default_ms: 10_000.0,
      settle_vehicle_ms: 8_000.0,
      // swept over live traces: 60s matches 120s on premature
      // settles but halves how long spans stay open; 30s doubles the noise
      settle_person_ms: 60_000.0,
      stationary_speed: 0.002,
      reassoc_iou: 0.3,
      wake_ticks: 3,
      confirm_ms: 500.0,
      max_dormant: 50,
      dormant_animate_ms: 300_000.0,
    }
  }
}

struct WorldTrack {
  label: String,
  bbox: [f32; 4],
  // last box while stationary; a stolen engine box must not drag the anchor
  anchor: [f32; 4],
  // a fade's last box is a guess until the object is seen on it; till then a
  // person or animal also wakes by leaving the spot it was last seen still
  anchor_seen: bool,
  rest: [f32; 4],
  rest_dist: f32,
  // box at confirmation, net travel since then separates leavers from sitters
  origin: [f32; 4],
  // the score of the last sighting above the user threshold: a weak tick keeps
  // the track alive but must not report the object as a 0.3 guess
  confidence: f32,
  score: f32,
  speed: f32,
  velocity: (f32, f32),
  // the same average in frame fractions per second, what consumers get: the
  // still logic keeps its per-tick scale, tuned on replays
  speed_s: f32,
  velocity_s: (f32, f32),
  // recent box sizes, and since when (and which way) every sighting found both
  // sides scaled past DEPTH_STEP in each of the last two windows
  sizes: VecDeque<(f64, f32, f32)>,
  scaling: Option<(f64, f32)>,
  state: TrackState,
  still_since_ms: f64,
  last_seen_ms: f64,
  moving_streak: u32,
  first_seen_ms: f64,
  best_shot_score: f32,
  dormant: bool,
  // a decode gap put it to sleep while consumers still count it visible: when the world saw again
  held_since_ms: Option<f64>,
  slow_confirm: bool,
  wake_dist: f32,
  // times the box cleared its spot since it last rested there
  departures: u32,
  departure_counted: bool,
  away: bool,
  rest_seen: u32,
  // when another body last showed up where this one went out of sight
  covered_ms: f64,
  confirmed_by_witness: bool,
  // per line, the last sighting clearly on one side of it
  line_sides: HashMap<u32, Sighting>,
}

pub struct CameraWorld {
  config: WorldConfig,
  engine: ByteTrack,
  // in id order: events, output and every first-found rule come out the same in every process
  tracks: BTreeMap<u32, WorldTrack>,
  engine_map: HashMap<u32, u32>,
  next_id: u32,
  last_t_ms: f64,
  prepared_lines: Vec<PreparedLine>,
  crossing_memory: HashSet<(u32, u32)>,
  prepared_zones: PreparedZones,
  min_confidence: f32,
  min_confidence_by_label: HashMap<String, f32>,
  // class ids live as long as the engine hands out tracks that carry them
  labels: Vec<String>,
  label_ids: HashMap<String, i64>,
  pose_baseline: Option<(f32, f32)>,
  // when the pose last moved: the engine lost every track the jump outran, so
  // tracks unseen since then re-associate without waiting out the orphan delay
  pose_moved_ms: f64,
  // label -> when a camera-side detector last reported it
  attestations: HashMap<String, f64>,
  // tracks a camera move dropped, reported as removed with the next tick
  dropped: Vec<u32>,
}

impl CameraWorld {
  pub fn new(config: WorldConfig) -> Self {
    Self {
      config,
      engine: new_engine(ENGINE_SPLIT),
      tracks: BTreeMap::new(),
      engine_map: HashMap::new(),
      next_id: 1,
      last_t_ms: 0.0,
      prepared_lines: Vec::new(),
      crossing_memory: HashSet::new(),
      prepared_zones: PreparedZones::default(),
      min_confidence: 0.0,
      min_confidence_by_label: HashMap::new(),
      labels: Vec::new(),
      label_ids: HashMap::new(),
      pose_baseline: None,
      pose_moved_ms: 0.0,
      attestations: HashMap::new(),
      dropped: Vec::new(),
    }
  }

  pub fn attest(&mut self, label: &str, t_ms: f64) {
    self.attestations.insert(label.to_string(), t_ms);
  }

  fn attested(&self, label: &str, t_ms: f64) -> bool {
    self
      .attestations
      .get(label)
      .is_some_and(|at| (t_ms - at).abs() <= ATTEST_WINDOW_MS)
  }

  pub fn set_zones(&mut self, zones: Vec<ZoneInput>) {
    self.prepared_zones = prepare_zones(&zones);
  }

  pub fn set_min_confidence(&mut self, min_confidence: f32) {
    self.min_confidence = min_confidence.max(0.0);
    let split = self.engine_split();
    self.engine.set_thresholds(split, split);
  }

  pub fn set_min_confidences(&mut self, by_label: HashMap<String, f32>) {
    self.min_confidence_by_label = by_label
      .into_iter()
      .map(|(k, v)| (k.to_lowercase(), v.max(0.0)))
      .collect();
  }

  pub fn min_confidence_for(&self, label: &str) -> f32 {
    self
      .min_confidence_by_label
      .get(&label.to_lowercase())
      .copied()
      .unwrap_or(self.min_confidence)
  }

  pub fn filter_indices(&self, detections: &[Detection]) -> Vec<u32> {
    filter_indices(
      detections,
      &self.prepared_zones,
      self.min_confidence,
      &self.min_confidence_by_label,
    )
  }

  pub fn set_lines(&mut self, lines: Vec<DetectionLineInput>, _aspect_ratio: f32) {
    let prepared = prepare_lines(&lines);
    // a zone change sends the lines again, a track standing on one must keep its side
    if prepared == self.prepared_lines {
      return;
    }
    self.prepared_lines = prepared;
    self.crossing_memory.clear();
    for track in self.tracks.values_mut() {
      track.line_sides.clear();
    }
  }

  pub fn notify_camera_move(&mut self) {
    let dropped = &mut self.dropped;
    self.tracks.retain(|id, track| {
      let keep = !track.dormant && track.state != TrackState::Stationary;
      if !keep {
        dropped.push(*id);
      }
      keep
    });
    self
      .engine_map
      .retain(|_, world_id| self.tracks.contains_key(world_id));
    self.crossing_memory.clear();
    for track in self.tracks.values_mut() {
      track.line_sides.clear();
    }
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
    let mut removed = std::mem::take(&mut self.dropped);
    let mut sightings = Vec::new();

    if self.last_t_ms != 0.0 && t_ms - self.last_t_ms > self.config.gap_reset_ms {
      // a decode gap is not evidence anything left: engine restarts, world
      // tracks sleep and wait for re-association; an unconfirmed flicker has
      // no identity to preserve and must not become a re-association anchor
      self.engine = new_engine(self.engine_split());
      self.engine_map.clear();
      self.tracks.retain(|_, t| t.state != TrackState::Tentative);
      for track in self.tracks.values_mut() {
        if !track.dormant || track.held_since_ms.is_some() {
          track.held_since_ms = Some(t_ms);
        }
        track.dormant = true;
        track.moving_streak = 0;
      }
      removed.extend(self.evict_dormant());
    }
    self.last_t_ms = t_ms;

    let dormant_animate_ms = self.config.dormant_animate_ms;
    self.tracks.retain(|id, t| {
      let expired = t.dormant && animate(&t.label) && t_ms - t.last_seen_ms > dormant_animate_ms;
      if expired {
        removed.push(*id);
      }
      !expired
    });
    self
      .engine_map
      .retain(|_, world_id| self.tracks.contains_key(world_id));

    // the caller reports an absolute pose offset, coarse and late: shifting the
    // engine by it misplaces every track the engine already followed through the
    // move (measured, the report lands after the picture moved). The engine keeps
    // raw image coords, its lost tracks drop the camera's motion they coast on,
    // and stored positions shift by the offset change so speed and anchors stay
    // pan-clean
    if let Some((mx, my)) = camera_motion {
      let (lx, ly) = *self.pose_baseline.get_or_insert((mx, my));
      let (dx, dy) = (mx - lx, my - ly);
      if dx != 0.0 || dy != 0.0 {
        for track in self.tracks.values_mut() {
          track.bbox[0] += dx;
          track.bbox[1] += dy;
          track.anchor[0] += dx;
          track.anchor[1] += dy;
          track.rest[0] += dx;
          track.rest[1] += dy;
          track.origin[0] += dx;
          track.origin[1] += dy;
          // the spot is an estimate now, departures counted against the old one say nothing
          forget_excursions(track);
          for (center, _) in track.line_sides.values_mut() {
            center.0 += dx;
            center.1 += dy;
          }
        }
        self.engine.forget_lost_motion();
        self.pose_moved_ms = t_ms;
      }
      self.pose_baseline = Some((mx, my));
    }

    // the engine gets everything down to the association floor — ByteTrack's
    // second pass continues established tracks through low-confidence
    // stretches; the user threshold gates track BIRTH below, not association
    let kept = filter_indices(
      detections,
      &self.prepared_zones,
      ASSOCIATION_FLOOR,
      &HashMap::new(),
    );
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
    let mut claimed: HashSet<u32> = HashSet::new();
    let mut handed: HashMap<u32, (Handover, MatchTier, f64)> = HashMap::new();
    let mut segments: Vec<TrackSegment> = Vec::new();
    let tick_detections = detections.clone();
    // continuations claim their identities before a newcomer may re-associate
    // one of them; otherwise a fragment born this tick grabs an identity whose
    // own engine track the engine refinds in the very same tick
    let mut engine_tracks = self.engine.update(detections);
    engine_tracks.sort_by_key(|t| !self.engine_map.contains_key(&(t.track_id as u32)));
    for t in engine_tracks {
      if !t.is_activated {
        continue;
      }
      let engine_id = t.track_id as u32;
      let Some(label) = self.labels.get(t.class_id as usize).cloned() else {
        continue;
      };
      let bbox = t.tlwh;
      let min_confidence = self.min_confidence_for(&label);
      let strong = t.score >= min_confidence;
      let sure = t.score >= min_confidence + SURE_MARGIN;

      // a same-class passer-by steals a stationary track's engine identity the
      // moment their boxes overlap and drags it along. Two signatures: some
      // detection still sits on the anchor (the sitter is visible), or the box
      // teleported — a parked object beginning to move travels a fraction of
      // its own size per tick, a steal jumps to wherever the passer is. Either
      // way unbind the engine id: the passer enters as its own identity next
      // tick and the sitter re-associates back onto its anchor
      if let Some(&world_id) = self.engine_map.get(&engine_id) {
        if let Some(track) = self.tracks.get(&world_id) {
          if track.state == TrackState::Stationary && box_iou(&track.anchor, &bbox) < 0.15 {
            let label_id = self.label_ids.get(&track.label).copied();
            let anchor_held = tick_detections.iter().any(|(dbox, _, dlabel)| {
              Some(*dlabel) == label_id && box_iou(dbox, &track.anchor) >= 0.3
            });
            let teleported = {
              let (px, py) = (
                track.bbox[0] + track.bbox[2] / 2.0,
                track.bbox[1] + track.bbox[3] / 2.0,
              );
              let (bx, by) = (bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0);
              let reach = track.anchor[2].max(track.anchor[3]);
              ((bx - px).powi(2) + (by - py).powi(2)).sqrt() > reach * 0.5
            };
            if anchor_held || teleported {
              self.engine_map.remove(&engine_id);
              continue;
            }
          }
        }
      }

      // one engine track per identity and tick: a second one is another view
      // of a body that already reported
      if let Some(id) = self.engine_map.get(&engine_id) {
        if claimed.contains(id) {
          self.engine_map.remove(&engine_id);
          continue;
        }
      }

      let continued = self
        .engine_map
        .get(&engine_id)
        .is_some_and(|id| self.tracks.contains_key(id));
      let world_id = match self.engine_map.get(&engine_id) {
        Some(id) if self.tracks.contains_key(id) => *id,
        _ => {
          // an orphan inside its grace continues at any score; a sleeper
          // reopens a span when it resumes and a NEW identity starts one, both
          // need the user threshold
          let reassociated = self.reassociate(&label, &bbox, t_ms, &tick_detections);
          let resumes_sleeper = reassociated.is_some_and(|(id, _)| self.tracks[&id].dormant);
          if (reassociated.is_none() || resumes_sleeper) && !strong {
            continue;
          }
          // a second engine track overlapping a live, freshly-seen world track
          // of the same label is another view of the same body, not a newcomer;
          // a real second person separates and gets their identity then
          if reassociated.is_none() && self.duplicates_live_track(&label, &bbox, t_ms) {
            continue;
          }
          // overlapping a known track of ANOTHER label smells like a
          // misclassification flicker (a sitting person read as a vehicle for a
          // second); such a birth must persist longer before it confirms —
          // a real rider (person + bicycle stacked) passes that bar easily
          let slow_confirm =
            reassociated.is_none() && self.contradicts_other_label(&label, &bbox, t_ms);
          // a tentative identity confirms on this tick at the earliest, nobody saw it to hand it over
          if let Some((id, tier)) =
            reassociated.filter(|(id, _)| self.tracks[id].state != TrackState::Tentative)
          {
            let track = &self.tracks[&id];
            let kind = if track.dormant {
              Handover::Recovered
            } else {
              Handover::Reassociated
            };
            handed.insert(id, (kind, tier, t_ms - track.last_seen_ms));
          }
          let id = reassociated.map(|(id, _)| id).unwrap_or_else(|| {
            #[cfg(feature = "replay")]
            if std::env::var("WORLD_DEBUG_BIRTH").is_ok() {
              eprintln!("BIRTH t={} label={} bbox={:?}", t_ms, label, bbox);
              for (tid, tr) in &self.tracks {
                if tr.label == label {
                  eprintln!(
                    "  cand #{tid} dormant={} unseen={}ms bbox={:?} anchor={:?}",
                    tr.dormant,
                    (t_ms - tr.last_seen_ms) as i64,
                    tr.bbox,
                    tr.anchor
                  );
                }
              }
            }
            let id = self.next_id;
            self.next_id += 1;
            let born = id;
            self.tracks.insert(
              id,
              WorldTrack {
                label: label.clone(),
                bbox,
                anchor: bbox,
                anchor_seen: false,
                rest: bbox,
                rest_dist: 0.0,
                origin: bbox,
                confidence: t.score,
                score: t.score,
                speed: 0.0,
                velocity: (0.0, 0.0),
                speed_s: 0.0,
                velocity_s: (0.0, 0.0),
                sizes: VecDeque::new(),
                scaling: None,
                state: TrackState::Tentative,
                still_since_ms: t_ms,
                last_seen_ms: t_ms,
                moving_streak: 0,
                first_seen_ms: t_ms,
                best_shot_score: 0.0,
                dormant: false,
                held_since_ms: None,
                slow_confirm,
                wake_dist: 0.0,
                departures: 0,
                departure_counted: false,
                away: false,
                rest_seen: 0,
                covered_ms: f64::NEG_INFINITY,
                confirmed_by_witness: false,
                line_sides: HashMap::new(),
              },
            );
            sightings.push(snapshot(born, &self.tracks[&born]));
            id
          });
          // the identity moves to this engine track; a lost engine track still
          // bound to it must not come back as a second writer
          self.engine_map.retain(|_, world_id| *world_id != id);
          self.engine_map.insert(engine_id, id);
          id
        }
      };

      let confirm_ms = self.config.confirm_ms;
      let settle_ms = self.settle_ms(&label);
      let attested = self.attested(&label, t_ms);
      let label_id = self.label_ids.get(&label).copied();
      // a newcomer that confirms on the spot of a body gone since before its
      // birth is that body under a new engine track (someone who left the spot
      // for a moment, a flicker on a parked car); one seen alongside it is someone else
      let world_id = match self.tracks.get(&world_id) {
        Some(track) if track.state == TrackState::Tentative => {
          let wait = if track.slow_confirm {
            confirm_ms * 4.0
          } else {
            confirm_ms
          };
          let confirms = (t_ms - track.first_seen_ms >= wait && strong) || attested;
          let born = track.first_seen_ms;
          let born_at = track.origin;
          let reassoc_iou = self.config.reassoc_iou;
          let held = confirms
            .then(|| self.reassociate(&label, &bbox, t_ms, &tick_detections))
            .flatten()
            .filter(|(held, _)| {
              let body = &self.tracks[held];
              // a still body is found again where it stands, not by something that came from elsewhere
              let stood_there = body.state != TrackState::Stationary
                || [body.anchor, body.bbox]
                  .iter()
                  .any(|spot| spot_match(spot, &born_at, reassoc_iou).is_some());
              body.state != TrackState::Tentative && body.last_seen_ms < born && stood_there
            });
          match held {
            Some((held, tier)) => {
              let body = &self.tracks[&held];
              let kind = if body.dormant {
                Handover::Recovered
              } else {
                Handover::Reassociated
              };
              handed.insert(held, (kind, tier, t_ms - body.last_seen_ms));
              self.tracks.remove(&world_id);
              self
                .engine_map
                .retain(|_, id| *id != held && *id != world_id);
              self.engine_map.insert(engine_id, held);
              held
            }
            None => world_id,
          }
        }
        _ => world_id,
      };
      claimed.insert(world_id);
      let track = self.tracks.get_mut(&world_id).unwrap();
      if track.state == TrackState::Tentative {
        // an identity is persistence over time, independent of tick rate: a
        // single flicker never re-sights after the window and dies silently.
        // Two detectors agreeing is persistence enough. A sighting below the
        // user threshold keeps a newcomer alive but never confirms it
        let confirm_ms = if track.slow_confirm {
          confirm_ms * 4.0
        } else {
          confirm_ms
        };
        let unconfirmed = t_ms - track.first_seen_ms < confirm_ms || !strong;
        if unconfirmed && !attested {
          track.bbox = bbox;
          track.anchor = bbox;
          track.score = t.score;
          if strong {
            track.confidence = t.score;
          }
          track.last_seen_ms = t_ms;
          maybe_best_shot(world_id, track, &mut events);
          continue;
        }
        track.confirmed_by_witness = unconfirmed;
        track.state = TrackState::Active;
        track.origin = track.bbox;
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
        track.speed_s = 0.0;
        track.velocity_s = (0.0, 0.0);
        // time unseen says nothing about a person standing still: a walker who
        // comes back restarts the still clock instead of settling on arrival.
        // Back within the still grace the clock runs on, as it would have while
        // held. A vehicle that drove in and is seen again on the same spot has parked
        if track.state != TrackState::Stationary
          && animate(&track.label)
          && t_ms - track.last_seen_ms > self.config.still_lost_grace_ms
        {
          track.still_since_ms = t_ms;
        }
        // a side seen before the gap would turn the next step into a crossing across it
        track.line_sides.clear();
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
        // real time between sightings, whatever the frame or tick rate was
        let per_s = 1000.0 / dt.max(50.0) as f32;
        track.velocity_s = (
          track.velocity_s.0 * 0.6 + (cx - px) * per_s * 0.4,
          track.velocity_s.1 * 0.6 + (cy - py) * per_s * 0.4,
        );
        track.speed_s = (track.velocity_s.0.powi(2) + track.velocity_s.1.powi(2)).sqrt();
      }
      track.dormant = false;
      track.held_since_ms = None;
      // a segment across a decode gap is not a movement, it is missing time
      if !was_dormant {
        segments.push((
          world_id,
          label.clone(),
          ((px, py), (track.bbox[2], track.bbox[3])),
          ((cx, cy), (bbox[2], bbox[3])),
        ));
      } else {
        // resuming from sleep re-opens visibility; the snapshot carries the
        // state, consumers decide whether stationary scenery counts
        events.push(SemanticEvent::ObjectRecovered(handed_over(
          snapshot(world_id, track),
          handed.get(&world_id),
        )));
      }
      track.score = t.score;
      if strong {
        track.confidence = t.score;
      }
      track.bbox = bbox;
      track.last_seen_ms = t_ms;
      track
        .sizes
        .retain(|(at, _, _)| t_ms - at <= DEPTH_WINDOW_MS * 3.0);
      // a view swap jumps in one window, a body in depth motion keeps scaling
      // the same way through both
      let step = size_before(&track.sizes, t_ms).and_then(|(then, w, h)| {
        let recent = depth_scale((w, h), (bbox[2], bbox[3]))?;
        let (_, w0, h0) = size_before(&track.sizes, then)?;
        let earlier = depth_scale((w0, h0), (w, h))?;
        (recent * earlier > 0.0 && recent.abs().min(earlier.abs()) >= DEPTH_STEP).then_some(recent)
      });
      track.scaling = match (step, track.scaling) {
        (Some(s), Some((since, sign))) if s.signum() == sign => Some((since, sign)),
        (Some(s), _) => Some((t_ms, s.signum())),
        _ => None,
      };
      track.sizes.push_back((t_ms, bbox[2], bbox[3]));

      let moving = track.speed >= self.config.stationary_speed;
      if track.state == TrackState::Stationary {
        let anchor_iou = box_iou(&track.anchor, &bbox);
        // waking needs the center to have left the anchor's reach, not merely a
        // deformed or swapped box view: an occluder or a torso-vs-legs detector
        // flap must not wake a parked object. The reach comes from the anchor
        // alone: the current box of someone standing up fills half the frame
        // and must not raise the bar for their own wake. Real departure keeps
        // gaining distance tick over tick; a detector view swap jumps once and
        // then holds still, and must not wake anything
        let leaves = |home: &[f32; 4], gained: f32| {
          let (ax, ay) = (home[0] + home[2] / 2.0, home[1] + home[3] / 2.0);
          let (bx, by) = (bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0);
          let dist = ((bx - ax).powi(2) + (by - ay).powi(2)).sqrt();
          let reach = home[2].max(home[3]);
          (box_iou(home, &bbox) < 0.15 && dist > reach * 0.5 && dist > gained + 0.005)
            .then_some(dist)
        };
        // a re-association may have handed the identity to another body: where
        // it reappears is the only rest this chain of sightings knows
        if !continued {
          track.rest = bbox;
        }
        let from_anchor = leaves(&track.anchor, track.wake_dist);
        // a view that still holds the rest spot is a body coming back into sight
        // from behind someone, not one leaving
        let from_rest =
          if track.anchor_seen || !animate(&track.label) || center_inside(&track.rest, &bbox) {
            None
          } else {
            leaves(&track.rest, track.rest_dist)
          };
        // walking in depth scales the box away from the size it rested at while the
        // center stays; a partial view growing back over the body scales toward it
        let size = (bbox[2], bbox[3]);
        let in_depth = track.scaling.is_some_and(|(since, sign)| {
          t_ms - since >= DEPTH_WINDOW_MS / 2.0
            && depth_scale((track.anchor[2], track.anchor[3]), size)
              .is_some_and(|s| s * sign >= DEPTH_STEP)
        });
        if moving && (from_anchor.is_some() || from_rest.is_some()) {
          track.wake_dist = from_anchor.unwrap_or(track.wake_dist);
          track.rest_dist = from_rest.unwrap_or(track.rest_dist);
          track.moving_streak += 1;
        } else {
          track.moving_streak = 0;
          track.wake_dist = 0.0;
          track.rest_dist = 0.0;
          // drift correction only at rest, so a departing box can't drag the anchor along
          if !moving && anchor_iou >= 0.6 {
            track.anchor = bbox;
            track.anchor_seen = true;
          }
          if !moving {
            track.rest = bbox;
          }
        }
        let (ax, ay) = center(&track.anchor);
        let anchor_dist = ((cx - ax).powi(2) + (cy - ay).powi(2)).sqrt();
        let mut oscillated = false;
        if track.label == "person" {
          // moving back and forth clears the spot and comes back without ever gaining distance; gestures
          // and view flaps never clear it, and a body still on the spot means this identity
          // rides someone else
          let view = track.anchor[2]
            .max(track.anchor[3])
            .max(bbox[2])
            .max(bbox[3]);
          let out = anchor_iou < 0.15
            && anchor_dist > view * 0.5
            && !center_inside(&bbox, &track.anchor)
            && !center_inside(&track.anchor, &bbox);
          let at_home = center_inside(&bbox, &track.anchor) || anchor_iou >= 0.3;
          let held = out
            && tick_detections.iter().any(|(dbox, _, dlabel)| {
              Some(*dlabel) == label_id && box_iou(dbox, &track.anchor) >= 0.3
            });
          if out && (sure || was_dormant) {
            if !track.away {
              track.away = true;
              // only a departure seen happening counts: a body found outside after a
              // sleep or a gap may have got there any way
              let left = !held && !was_dormant;
              track.departure_counted = left;
              track.departures += u32::from(left);
            } else if held && track.departure_counted {
              track.departure_counted = false;
              track.departures = track.departures.saturating_sub(1);
            }
          } else if at_home {
            track.away = false;
          }
          track.rest_seen = if !moving && at_home {
            track.rest_seen.saturating_add(1)
          } else {
            0
          };
          if track.rest_seen >= self.config.wake_ticks {
            track.departures = 0;
          }
          oscillated = out && sure && !held && track.departures >= 2;
        }
        // waking opens a span: the tick that completes it has to be strong
        if (track.moving_streak >= self.config.wake_ticks || in_depth || oscillated) && strong {
          let cause = if track.moving_streak >= self.config.wake_ticks {
            WakeCause::Travel
          } else if in_depth {
            WakeCause::Depth
          } else {
            WakeCause::Oscillation
          };
          let anchor_distance =
            anchor_dist / track.anchor[2].max(track.anchor[3]).max(f32::EPSILON);
          track.state = TrackState::Active;
          track.still_since_ms = t_ms;
          track.moving_streak = 0;
          track.wake_dist = 0.0;
          track.rest_dist = 0.0;
          forget_excursions(track);
          let mut woke = snapshot(world_id, track);
          woke.wake_cause = Some(cause);
          woke.anchor_distance = Some(anchor_distance);
          woke.far_from_anchor = Some(anchor_distance > FAR_FROM_ANCHOR);
          events.push(SemanticEvent::ObjectWoke(woke));
        }
      } else {
        // stillness is position truth, not velocity noise: the still clock only
        // resets when the box center leaves the anchor (or overlap collapses).
        // A partial detection of the same object (torso-only vs full body at a
        // frame edge, window-seam size flap) keeps its center inside the anchor
        // and must not restart the clock
        if box_iou(&track.anchor, &bbox) < 0.3 || !center_inside(&bbox, &track.anchor) {
          track.anchor = bbox;
          track.still_since_ms = t_ms;
        }
        if t_ms - track.still_since_ms >= settle_ms {
          track.state = TrackState::Stationary;
          track.anchor = bbox;
          track.anchor_seen = true;
          events.push(SemanticEvent::ObjectSettled(snapshot(world_id, track)));
        }
      }
      maybe_best_shot(world_id, track, &mut events);
      seen.push(world_id);
    }

    let crossings = self.compute_crossings(&segments, t_ms);

    let sighted: Vec<(u32, String, [f32; 4])> = seen
      .iter()
      .filter_map(|id| self.tracks.get(id).map(|t| (*id, t.label.clone(), t.bbox)))
      .collect();
    let reassoc_iou = self.config.reassoc_iou;
    for (id, track) in self.tracks.iter_mut() {
      if track.dormant
        || track.state != TrackState::Active
        || !animate(&track.label)
        || seen.contains(id)
      {
        continue;
      }
      let covered = sighted.iter().any(|(other, label, bbox)| {
        other != id && *label == track.label && spot_match(&track.bbox, bbox, reassoc_iou).is_some()
      });
      if covered {
        track.covered_ms = t_ms;
      }
    }

    let mut removed_quiet = Vec::new();
    let depart_grace_ms = self.config.depart_grace_ms;
    let still_lost_grace_ms = self.config.still_lost_grace_ms;
    let attestations = &self.attestations;
    // a track confirmed without being seen this tick: its sighting was
    // withheld until now, so it is delivered with this tick, dated by
    // last_seen_ms
    let mut late_confirmed: Vec<TrackSnapshot> = Vec::new();
    self.tracks.retain(|id, track| {
      if seen.contains(id) {
        return true;
      }
      if track.dormant {
        // slept through a gap and not back within the grace after it: out of sight
        let grace = if track.state == TrackState::Stationary {
          still_lost_grace_ms
        } else {
          depart_grace_ms
        };
        if track
          .held_since_ms
          .is_some_and(|since| t_ms - since > grace)
        {
          track.held_since_ms = None;
          events.push(SemanticEvent::ObjectLost(snapshot(*id, track)));
        }
        return true;
      }
      if track.state == TrackState::Tentative {
        // the camera's report may land a tick after our only sighting: the
        // sighting still confirms, with the box it left behind
        let attested = attestations
          .get(&track.label)
          .is_some_and(|at| (at - track.last_seen_ms).abs() <= ATTEST_WINDOW_MS);
        if attested {
          track.confirmed_by_witness = true;
          track.state = TrackState::Active;
          track.origin = track.bbox;
          created.push(*id);
          events.push(SemanticEvent::ObjectEntered(snapshot(*id, track)));
          late_confirmed.push(snapshot(*id, track));
          return true;
        }
        // an unconfirmed flicker dies silently
        if t_ms - track.last_seen_ms > depart_grace_ms {
          removed_quiet.push(*id);
          return false;
        }
        return true;
      }
      if t_ms - track.last_seen_ms > depart_grace_ms {
        // only something that went somewhere can leave; a still object sleeps.
        // Three still signals, each robust against a failure mode of the others:
        // anchor hold (classic), EMA speed (box-flap immune), and net travel
        // from the confirmation spot (a walk-in that sat down is not a leaver)
        let travel = {
          let (ox, oy) = (
            track.origin[0] + track.origin[2] / 2.0,
            track.origin[1] + track.origin[3] / 2.0,
          );
          let (bx, by) = (
            track.bbox[0] + track.bbox[2] / 2.0,
            track.bbox[1] + track.bbox[3] / 2.0,
          );
          ((bx - ox).powi(2) + (by - oy).powi(2)).sqrt()
        };
        let reach = {
          let diag = |b: &[f32; 4]| (b[2].powi(2) + b[3].powi(2)).sqrt();
          diag(&track.origin).max(diag(&track.bbox))
        };
        // a leaver's box gets clipped at the frame border, pinning the center
        // and collapsing the EMA speed right before it vanishes: whoever
        // traveled far AND faded at an edge has left, whatever the final speed
        let at_edge = track.bbox[0] <= 0.01
          || track.bbox[1] <= 0.01
          || track.bbox[0] + track.bbox[2] >= 0.99
          || track.bbox[1] + track.bbox[3] >= 0.99;
        let traveled_out = track.state != TrackState::Stationary && at_edge && travel >= reach;
        // a walker close to the camera has a box too large to ever travel its
        // own diagonal and fades before the border: its velocity carries it out
        let heading_out = {
          let cx = track.bbox[0] + track.bbox[2] / 2.0 + track.velocity_s.0;
          let cy = track.bbox[1] + track.bbox[3] / 2.0 + track.velocity_s.1;
          !(0.0..=1.0).contains(&cx) || !(0.0..=1.0).contains(&cy)
        };
        // walking at the camera grows the box into the border, the center stays: grown since
        // confirmation and still growing when it faded at an edge means it passed the camera
        let size = (track.bbox[2], track.bbox[3]);
        let approached_out = track.state != TrackState::Stationary
          && at_edge
          && depth_scale((track.origin[2], track.origin[3]), size).is_some_and(|s| s >= DEPTH_STEP)
          && size_before(&track.sizes, track.last_seen_ms)
            .and_then(|(_, w, h)| depth_scale((w, h), size))
            .is_some_and(|s| s >= DEPTH_STEP);
        let was_still = !traveled_out
          && !approached_out
          && (track.still_since_ms == track.first_seen_ms
            || track.last_seen_ms - track.still_since_ms >= depart_grace_ms
            || track.speed < self.config.stationary_speed * 2.0
            || (travel < reach && !heading_out));
        if was_still || track.state == TrackState::Stationary {
          // a track that faded without ever moving is scenery caught in detector
          // flickers, not a visitor: settle it the moment the hold starts, so its
          // span closes now and re-sightings stay silent instead of stamping
          // moving evidence for as long as the hold lasts. A body moved if its
          // still clock restarted, travel can't tell: someone moving in place goes
          // nowhere either. A vehicle that faded close to where it was confirmed
          // has parked
          let scenery = if animate(&track.label) {
            track.still_since_ms == track.first_seen_ms
          } else {
            travel < reach * 0.5
          };
          if track.state != TrackState::Stationary && scenery {
            track.state = TrackState::Stationary;
            track.anchor = track.bbox;
            track.anchor_seen = false;
            track.rest = track.bbox;
            events.push(SemanticEvent::ObjectSettled(snapshot(*id, track)));
          }
          // a body that moved and then vanished in the open is out of sight: its
          // span closes now and a re-sighting recovers it. One that another body
          // keeps showing up over is hidden behind it and stays
          let hidden = t_ms - track.covered_ms <= depart_grace_ms;
          if track.state != TrackState::Stationary && animate(&track.label) && !hidden {
            track.dormant = true;
            track.moving_streak = 0;
            events.push(SemanticEvent::ObjectLost(snapshot(*id, track)));
            return true;
          }
          // a still object that fades mid-frame is still there, the detector
          // just missed it; hold visibility through the longer still grace so
          // a re-sighting is silent continuation instead of a lost/recovered
          // cycle that reopens spans
          if t_ms - track.last_seen_ms <= still_lost_grace_ms {
            return true;
          }
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

    let mut tracked: Vec<TrackSnapshot> = seen
      .iter()
      .filter_map(|id| {
        self
          .tracks
          .get(id)
          .filter(|t| t.state != TrackState::Tentative)
          .map(|t| handed_over(snapshot(*id, t), handed.get(id)))
      })
      .collect();
    tracked.extend(late_confirmed);

    WorldUpdate {
      sightings,
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
      let Some(track) = self.tracks.get_mut(world_id) else {
        continue;
      };
      // a settled track is where its anchor is, its box flickering across a line goes nowhere
      if track.state == TrackState::Stationary {
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
        let key = line_idx as u32;
        if !track.line_sides.contains_key(&key) && line_side(line, *prev).is_some() {
          track.line_sides.insert(key, *prev);
        }
        let Some(side) = line_side(line, *curr) else {
          continue;
        };
        let Some(held) = track.line_sides.insert(key, *curr) else {
          continue;
        };
        if line_side(line, held).is_none_or(|was| was == side) {
          continue;
        }
        let ((from, _), (to, _)) = (held, *curr);
        let cross = segment_intersection(
          from.0, from.1, to.0, to.1, line.p1[0], line.p1[1], line.p2[0], line.p2[1],
        );
        if cross == 0.0 {
          continue;
        }
        // the editor labels the side at mid - rot90(p2 - p1) as A, so travel
        // from A to B has a negative path x line cross product
        let direction = if cross < 0.0 {
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
          confidence: track.confidence,
          timestamp_ms: t_ms,
          prev_pos: [from.0, from.1],
          curr_pos: [to.0, to.1],
        });
      }
    }
    events
  }

  fn reassociate(
    &self,
    label: &str,
    bbox: &[f32; 4],
    t_ms: f64,
    tick_detections: &[([f32; 4], f32, i64)],
  ) -> Option<(u32, MatchTier)> {
    let label_id = self.label_ids.get(label).copied();
    let mut best: Option<(u32, f32, MatchTier)> = None;
    // fixed order: a map walk would break score ties by hash seed, and a replay
    // has to land on the same identities as the live run
    let mut candidates: Vec<(&u32, &WorldTrack)> = self.tracks.iter().collect();
    candidates.sort_by_key(|(id, _)| **id);
    for (id, track) in candidates {
      // dormant tracks, plus orphans the engine already forgot while the world
      // grace still runs; a freshly-seen track keeps its identity to itself
      let orphaned =
        t_ms - track.last_seen_ms > ORPHAN_MS || track.last_seen_ms < self.pose_moved_ms;
      if (!track.dormant && !orphaned) || track.label != label {
        continue;
      }
      // a settled identity belongs to whoever claims its anchor best: a car
      // entering over a dormant parked spot must not resume the parker's
      // identity while the parker itself still sits on the anchor. Only for
      // stationary sleepers (an active sleeper has no sitter to protect), and
      // only against overlap-tier claims — the near tier below exists exactly
      // for same-body view flips, where a competing part-detection is expected
      let own_claim = box_iou(&track.anchor, bbox);
      let contested = track.state == TrackState::Stationary
        && own_claim >= 0.1
        && tick_detections.iter().any(|(dbox, _, dlabel)| {
          Some(*dlabel) == label_id
            && dbox != bbox
            && box_iou(dbox, &track.anchor) > own_claim.max(0.3)
        });
      if contested {
        continue;
      }
      // match against the anchor AND the last box, best of both: the anchor can
      // be stale (the standing box of someone who since sat down) while the
      // last box carries the freshest evidence
      for reference in [&track.anchor, &track.bbox] {
        let Some((score, tier)) = spot_match(reference, bbox, self.config.reassoc_iou) else {
          continue;
        };
        if best.is_none_or(|(_, b, _)| score > b) {
          best = Some((*id, score, tier));
        }
      }
    }
    best.map(|(id, _, tier)| (id, tier))
  }

  fn duplicates_live_track(&self, label: &str, bbox: &[f32; 4], t_ms: f64) -> bool {
    // no size guard here: a body fragment is naturally many times smaller than
    // the full box, and suppressing a birth costs nothing lasting — a real
    // second person separates and enters a moment later
    self.tracks.values().any(|track| {
      if track.dormant || track.label != label || t_ms - track.last_seen_ms > ORPHAN_MS {
        return false;
      }
      box_iou(&track.bbox, bbox) >= self.config.reassoc_iou
        || center_inside(bbox, &track.bbox)
        || center_inside(&track.bbox, bbox)
    })
  }

  fn contradicts_other_label(&self, label: &str, bbox: &[f32; 4], t_ms: f64) -> bool {
    self.tracks.values().any(|track| {
      if track.dormant || track.label == label {
        return false;
      }
      let position_stable = track.state == TrackState::Stationary
        || t_ms - track.last_seen_ms <= self.config.depart_grace_ms;
      position_stable
        && (box_iou(&track.bbox, bbox) >= self.config.reassoc_iou
          || center_inside(bbox, &track.bbox)
          || center_inside(&track.bbox, bbox))
    })
  }

  fn engine_split(&self) -> f32 {
    if self.min_confidence > 0.0 {
      self.min_confidence.min(ENGINE_SPLIT)
    } else {
      ENGINE_SPLIT
    }
  }

  fn settle_ms(&self, label: &str) -> f64 {
    match label {
      "vehicle" => self.config.settle_vehicle_ms,
      "person" => self.config.settle_person_ms,
      _ => self.config.settle_default_ms,
    }
  }

  fn evict_dormant(&mut self) -> Vec<u32> {
    let mut evicted = Vec::new();
    while self.tracks.values().filter(|t| t.dormant).count() > self.config.max_dormant {
      let oldest = self
        .tracks
        .iter()
        .filter(|(_, t)| t.dormant)
        .min_by(|a, b| {
          a.1
            .last_seen_ms
            .total_cmp(&b.1.last_seen_ms)
            .then(a.0.cmp(b.0))
        })
        .map(|(id, _)| *id);
      match oldest {
        Some(id) => {
          self.tracks.remove(&id);
          evicted.push(id);
        }
        None => break,
      };
    }
    evicted
  }
}

fn spot_match(reference: &[f32; 4], bbox: &[f32; 4], reassoc_iou: f32) -> Option<(f32, MatchTier)> {
  let iou = box_iou(reference, bbox);
  // a partial re-sighting (head-only vs full body) fails IoU but keeps one
  // center inside the other; the size guard only blocks depth confusion
  // (near vs far person, ~50x apart), body fragments run up to ~5x
  let area = bbox[2] * bbox[3];
  let reference_area = reference[2] * reference[3];
  let size_ok = area <= reference_area * 8.0 && reference_area <= area * 8.0;
  let contained = size_ok && (center_inside(bbox, reference) || center_inside(reference, bbox));
  // last resort for detector output flapping between two views of one body
  // (torso-only vs legs-only, a window seam cutting it in half): centers
  // within one width sideways and half a height up or down still count.
  // Per axis, not by the larger side: a standing person is three widths
  // tall, and a sideways tolerance of half a height would hand the
  // identity to the next person over
  let near = size_ok && {
    let (ax, ay) = (
      reference[0] + reference[2] / 2.0,
      reference[1] + reference[3] / 2.0,
    );
    let (bx, by) = (bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0);
    (bx - ax).abs() <= reference[2].max(bbox[2])
      && (by - ay).abs() <= reference[3].max(bbox[3]) * 0.5
  };
  if iou >= reassoc_iou {
    Some((1.0 + iou, MatchTier::Overlap))
  } else if contained {
    Some((iou, MatchTier::Contained))
  } else if near {
    Some((iou * 0.5, MatchTier::Near))
  } else {
    None
  }
}

fn animate(label: &str) -> bool {
  matches!(label, "person" | "animal")
}

fn size_before(sizes: &VecDeque<(f64, f32, f32)>, at: f64) -> Option<(f64, f32, f32)> {
  sizes
    .iter()
    .rev()
    .find(|(seen, _, _)| at - seen >= DEPTH_WINDOW_MS)
    .copied()
}

fn depth_scale(from: (f32, f32), to: (f32, f32)) -> Option<f32> {
  if from.0 <= 0.0 || from.1 <= 0.0 || to.0 <= 0.0 || to.1 <= 0.0 {
    return None;
  }
  let (sw, sh) = ((to.0 / from.0).ln(), (to.1 / from.1).ln());
  let (small, large) = (sw.abs().min(sh.abs()), sw.abs().max(sh.abs()));
  (sw * sh > 0.0 && small * 3.0 >= large).then_some(small.copysign(sw))
}

fn center_inside(of: &[f32; 4], within: &[f32; 4]) -> bool {
  let cx = of[0] + of[2] / 2.0;
  let cy = of[1] + of[3] / 2.0;
  cx >= within[0] && cx <= within[0] + within[2] && cy >= within[1] && cy <= within[1] + within[3]
}

// the side of a line a box center is clearly on, none inside the dead band
fn line_side(line: &PreparedLine, ((cx, cy), (w, h)): Sighting) -> Option<bool> {
  let (lx, ly) = (line.p2[0] - line.p1[0], line.p2[1] - line.p1[1]);
  // distance and extent along the same unnormalized normal (-ly, lx), only their ratio counts
  let dist = (cy - line.p1[1]) * lx - (cx - line.p1[0]) * ly;
  let extent = ly.abs() * w + lx.abs() * h;
  (dist.abs() > extent * LINE_DEAD_BAND).then_some(dist > 0.0)
}

fn forget_excursions(track: &mut WorldTrack) {
  track.departures = 0;
  track.departure_counted = false;
  track.away = false;
  track.rest_seen = 0;
}

fn maybe_best_shot(world_id: u32, track: &mut WorldTrack, events: &mut Vec<SemanticEvent>) {
  let b = &track.bbox;
  let clipped = b[0] <= 0.005 || b[1] <= 0.005 || b[0] + b[2] >= 0.995 || b[1] + b[3] >= 0.995;
  let edge_penalty = if clipped { 0.3 } else { 1.0 };
  let score = track.score * (b[2] * b[3]).sqrt() * edge_penalty;
  if score > track.best_shot_score * 1.25 {
    track.best_shot_score = score;
    events.push(SemanticEvent::BestShotUpdated(snapshot(world_id, track)));
  }
}

fn new_engine(split: f32) -> ByteTrack {
  ByteTrack::new(split, 30, 0.9, split)
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
    score: t.score,
    speed: t.speed_s,
    velocity_x: t.velocity_s.0,
    velocity_y: t.velocity_s.1,
    state: t.state,
    stationary_since_ms: (t.state == TrackState::Stationary).then_some(t.still_since_ms),
    last_seen_ms: t.last_seen_ms,
    attested: t.confirmed_by_witness,
    wake_cause: None,
    anchor_distance: None,
    far_from_anchor: None,
    handover: None,
    handover_tier: None,
    unseen_ms: None,
  }
}

fn handed_over(mut s: TrackSnapshot, handed: Option<&(Handover, MatchTier, f64)>) -> TrackSnapshot {
  if let Some(&(kind, tier, unseen_ms)) = handed {
    s.handover = Some(kind);
    s.handover_tier = Some(tier);
    s.unseen_ms = Some(unseen_ms);
  }
  s
}

#[cfg(test)]
mod tests {
  use super::*;

  fn person(x: f32) -> Detection {
    person_at(x, 0.4)
  }

  fn person_at(x: f32, y: f32) -> Detection {
    Detection {
      x,
      y,
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
      points: [[x_ui, 80.0], [x_ui, 20.0]],
    }
  }

  fn entrance_line() -> DetectionLineInput {
    DetectionLineInput {
      name: "entrance".to_string(),
      direction: LineDirectionFilter::Both,
      labels: Vec::new(),
      points: [[47.0, 33.0], [69.0, 33.0]],
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
    assert_eq!(crossings[0].direction, CrossingDirection::AToB);
  }

  #[test]
  fn two_worlds_fed_alike_report_in_the_same_order() {
    let run = || {
      let mut world = CameraWorld::new(WorldConfig::default());
      let mut said = Vec::new();
      for i in 0..60 {
        // eight people walk in side by side and leave together
        let crowd: Vec<Detection> = if i < 20 {
          (0..8)
            .map(|k| person_at(0.02 + k as f32 * 0.11 + i as f32 * 0.004, 0.4))
            .collect()
        } else {
          Vec::new()
        };
        let up = world.ingest(i as f64 * 200.0, &crowd, None);
        said.push(format!("{:?} {:?} {:?}", up.events, up.tracked, up.removed));
      }
      said
    };
    assert_eq!(run(), run());
  }

  #[test]
  fn crossing_the_drawn_line_fires_downward_as_a_to_b() {
    let mut world = CameraWorld::new(WorldConfig::default());
    world.set_lines(
      vec![DetectionLineInput {
        name: "garden".to_string(),
        direction: LineDirectionFilter::Both,
        labels: Vec::new(),
        points: [[7.0, 67.0], [50.0, 74.0]],
      }],
      1.0,
    );
    let mut crossings = Vec::new();
    for i in 0..15 {
      let update = world.ingest(
        i as f64 * 200.0,
        &[person_at(0.25, 0.45 + i as f32 * 0.03)],
        None,
      );
      crossings.extend(update.crossings);
    }
    assert_eq!(crossings.len(), 1);
    assert_eq!(crossings[0].direction, CrossingDirection::AToB);
  }

  #[test]
  fn one_way_line_ignores_the_other_direction() {
    let mut world = CameraWorld::new(WorldConfig::default());
    let mut line = vertical_line_at(50.0, "one-way");
    line.direction = LineDirectionFilter::AToB;
    world.set_lines(vec![line], 1.0);
    // A is the left half: walking left is b-to-a and must stay silent
    for i in 0..20 {
      let update = world.ingest(i as f64 * 200.0, &[person(0.8 - i as f32 * 0.03)], None);
      assert!(
        update.crossings.is_empty(),
        "b-to-a fired on an a-to-b line"
      );
    }
    let mut fired = 0;
    for i in 0..20 {
      let update = world.ingest(
        60_000.0 + i as f64 * 200.0,
        &[person(0.2 + i as f32 * 0.03)],
        None,
      );
      fired += update.crossings.len();
    }
    assert_eq!(fired, 1);
  }

  #[test]
  fn box_jitter_beside_the_line_never_crosses() {
    // regression: a parked vehicle wobbling where the A/B direction arrow is
    // rendered tripped the line it never touched
    let mut world = CameraWorld::new(WorldConfig::default());
    world.set_lines(
      vec![DetectionLineInput {
        name: "garden".to_string(),
        direction: LineDirectionFilter::Both,
        labels: Vec::new(),
        points: [[7.0, 67.0], [50.0, 74.0]],
      }],
      1.0,
    );
    let mut crossings = 0;
    for i in 0..20 {
      let x = if i % 2 == 0 { 0.25 } else { 0.27 };
      let update = world.ingest(i as f64 * 200.0, &[person_at(x, 0.4)], None);
      crossings += update.crossings.len();
    }
    assert_eq!(crossings, 0);
  }

  #[test]
  fn a_box_resting_on_the_line_never_crosses() {
    let mut world = CameraWorld::new(WorldConfig::default());
    world.set_lines(vec![entrance_line()], 1.0);
    let mut crossings = 0;
    for i in 0..200 {
      let jitter = if i % 2 == 0 { 0.018 } else { -0.018 };
      let update = world.ingest(
        i as f64 * 200.0,
        &[box_at(0.53, 0.23 + jitter, 0.1, 0.2, "person")],
        None,
      );
      crossings += update.crossings.len();
    }
    assert_eq!(
      crossings, 0,
      "jitter of 9 % of the box height flipped sides"
    );
  }

  #[test]
  fn a_vehicle_parked_on_the_line_stays_silent_through_drift_and_sleep() {
    // Hof 2026-10-17 02:00: a night detection parked across the line, its center
    // drifting by a sixth of its height and stepping 0.007 after a recovery
    let mut world = CameraWorld::new(WorldConfig::default());
    world.set_lines(vec![entrance_line()], 1.0);
    let car = |cy: f32| box_at(0.62, cy - 0.0965, 0.075, 0.193, "vehicle");
    let mut crossings = 0;
    let mut t = 0.0;
    for i in 0..60 {
      let jitter = if i % 2 == 0 { 0.001 } else { -0.001 };
      crossings += world
        .ingest(t, &[car(0.3301 + jitter)], None)
        .crossings
        .len();
      t += 200.0;
    }
    for i in 0..300 {
      let drift = if i < 150 {
        i as f32
      } else {
        450.0 - 2.0 * i as f32
      };
      crossings += world
        .ingest(t, &[car(0.3301 + drift * 0.0002)], None)
        .crossings
        .len();
      t += 200.0;
    }
    t += 70_000.0;
    for cy in [0.3301, 0.3228, 0.3301, 0.3228, 0.36, 0.30] {
      crossings += world.ingest(t, &[car(cy)], None).crossings.len();
      t += 100.0;
    }
    assert_eq!(crossings, 0);
  }

  #[test]
  fn a_walker_who_stops_on_the_line_crosses_once() {
    let mut world = CameraWorld::new(WorldConfig::default());
    world.set_lines(vec![entrance_line()], 1.0);
    let mut path: Vec<f32> = (0..20).map(|i| 0.13 + i as f32 * 0.01).collect();
    path.extend((0..25).map(|i| if i % 2 == 0 { 0.34 } else { 0.32 }));
    path.extend((1..=30).map(|i| 0.33 + i as f32 * 0.01));
    let mut crossings = Vec::new();
    for (i, cy) in path.iter().enumerate() {
      let update = world.ingest(
        i as f64 * 200.0,
        &[box_at(0.53, cy - 0.1, 0.1, 0.2, "person")],
        None,
      );
      crossings.extend(update.crossings);
    }
    assert_eq!(crossings.len(), 1);
    assert_eq!(crossings[0].direction, CrossingDirection::AToB);
  }

  #[test]
  fn the_same_lines_sent_again_keep_a_walker_who_stopped_on_the_line() {
    let mut world = CameraWorld::new(WorldConfig::default());
    world.set_lines(vec![entrance_line()], 1.0);
    let mut path: Vec<f32> = (0..20).map(|i| 0.13 + i as f32 * 0.01).collect();
    path.extend((0..25).map(|i| if i % 2 == 0 { 0.34 } else { 0.32 }));
    path.extend((1..=30).map(|i| 0.33 + i as f32 * 0.01));
    let mut crossings = Vec::new();
    for (i, cy) in path.iter().enumerate() {
      if i == 30 {
        world.set_lines(vec![entrance_line()], 1.0);
      }
      let update = world.ingest(
        i as f64 * 200.0,
        &[box_at(0.53, cy - 0.1, 0.1, 0.2, "person")],
        None,
      );
      crossings.extend(update.crossings);
    }
    assert_eq!(crossings.len(), 1);
    assert_eq!(crossings[0].direction, CrossingDirection::AToB);
  }

  #[test]
  fn a_sitter_beside_the_line_who_walks_across_crosses_once() {
    let mut world = CameraWorld::new(WorldConfig::default());
    world.set_lines(vec![entrance_line()], 1.0);
    let mut crossings = Vec::new();
    let mut settled = false;
    let mut t = 0.0;
    for i in 0..350 {
      let jitter = if i % 2 == 0 { 0.002 } else { -0.002 };
      let update = world.ingest(t, &[box_at(0.53, 0.18 + jitter, 0.1, 0.2, "person")], None);
      settled |= update.events.iter().any(|e| e.kind() == "objectSettled");
      crossings.extend(update.crossings);
      t += 200.0;
    }
    assert!(settled, "the sitter never settled");
    for i in 1..=20 {
      let update = world.ingest(
        t,
        &[box_at(0.53, 0.18 + i as f32 * 0.02, 0.1, 0.2, "person")],
        None,
      );
      crossings.extend(update.crossings);
      t += 200.0;
    }
    assert_eq!(crossings.len(), 1);
    assert_eq!(crossings[0].direction, CrossingDirection::AToB);
  }

  #[test]
  fn walking_past_the_end_of_the_line_never_crosses() {
    let mut world = CameraWorld::new(WorldConfig::default());
    world.set_lines(vec![entrance_line()], 1.0);
    let mut crossings = 0;
    for i in 0..45 {
      let cy = 0.15 + i as f32 * 0.01;
      let update = world.ingest(
        i as f64 * 200.0,
        &[box_at(0.75, cy - 0.1, 0.1, 0.2, "person")],
        None,
      );
      crossings += update.crossings.len();
    }
    assert_eq!(crossings, 0);
  }

  #[test]
  fn a_walker_growing_towards_the_camera_crosses() {
    // the step past the line widens the dead band over the last sighting before it
    let mut world = CameraWorld::new(WorldConfig::default());
    world.set_lines(vec![entrance_line()], 1.0);
    let path = [
      (0.2, 0.1),
      (0.2, 0.1),
      (0.2, 0.1),
      (0.2, 0.1),
      (0.25, 0.12),
      (0.29, 0.14),
      (0.31, 0.15),
      (0.355, 0.24),
      (0.4, 0.27),
      (0.45, 0.3),
    ];
    let mut crossings = Vec::new();
    for (i, (cy, h)) in path.iter().enumerate() {
      let update = world.ingest(
        i as f64 * 200.0,
        &[box_at(0.53, cy - h / 2.0, 0.1, *h, "person")],
        None,
      );
      crossings.extend(update.crossings);
    }
    assert_eq!(crossings.len(), 1);
    assert_eq!(crossings[0].direction, CrossingDirection::AToB);
  }

  #[test]
  fn a_newcomer_seen_first_next_to_the_line_crosses() {
    // Hof 2026-10-17 03:30: through the gate, first seen 0.22 box heights above
    // the line, the confirming step lands inside the dead band
    let mut world = CameraWorld::new(WorldConfig::default());
    world.set_lines(vec![entrance_line()], 1.0);
    let path = [0.286, 0.286, 0.286, 0.315, 0.355, 0.4, 0.45, 0.5];
    let mut crossings = Vec::new();
    for (i, cy) in path.iter().enumerate() {
      let update = world.ingest(
        i as f64 * 200.0,
        &[box_at(0.53, cy - 0.1, 0.1, 0.2, "person")],
        None,
      );
      crossings.extend(update.crossings);
    }
    assert_eq!(crossings.len(), 1);
    assert_eq!(crossings[0].direction, CrossingDirection::AToB);
  }

  #[test]
  fn a_faded_newcomer_settles_instead_of_haunting() {
    let mut world = CameraWorld::new(WorldConfig::default());
    for i in 0..4 {
      world.ingest(i as f64 * 200.0, &[person_at(0.3, 0.4)], None);
    }
    let mut settled = 0;
    let mut lost = 0;
    for i in 4..40 {
      let up = world.ingest(i as f64 * 200.0, &[], None);
      settled += up
        .events
        .iter()
        .filter(|e| e.kind() == "objectSettled")
        .count();
      lost += up
        .events
        .iter()
        .filter(|e| e.kind() == "objectLost")
        .count();
    }
    assert_eq!(settled, 1, "the span must close when the hold starts");
    assert_eq!(lost, 0, "the identity is held, not lost");

    // the flicker returning must not read as an arrival
    let up = world.ingest(8_200.0, &[person_at(0.3, 0.4)], None);
    assert!(up.created.is_empty());
    assert!(!up
      .events
      .iter()
      .any(|e| matches!(e.kind(), "objectEntered" | "objectWoke" | "objectRecovered")));
    assert!(up.tracked.iter().all(|t| t.state == TrackState::Stationary));
  }

  #[test]
  fn a_walker_close_to_the_camera_leaves_instead_of_lending_its_identity() {
    let mut world = CameraWorld::new(WorldConfig::default());
    world.set_min_confidence(0.7);
    // the detector drops him before his box, longer than the walk across, reaches the border
    let mut t = 0.0;
    let mut id = None;
    for i in 0..12 {
      let walker = box_at(0.08 + i as f32 * 0.055, 0.24, 0.24, 0.75, "person");
      let up = world.ingest(t, &[walker], None);
      id = id.or(up.tracked.first().map(|s| s.track_id));
      t += 200.0;
    }
    let id = id.expect("confirmed");
    let mut departed = false;
    while t < 37_000.0 {
      let up = world.ingest(t, &[], None);
      departed |= up.events.iter().any(|e| e.kind() == "objectDeparted");
      t += 500.0;
    }
    assert!(departed, "the walker left the frame");
    // half a minute later someone else steps in at the right edge
    let mut stranger = box_at(0.85, 0.10, 0.14, 0.89, "person");
    stranger.confidence = 0.58;
    let up = world.ingest(t, &[stranger], None);
    assert!(
      up.tracked.iter().all(|s| s.track_id != id),
      "the stranger continued the walker's identity"
    );
  }

  #[test]
  fn two_people_crossing_under_one_box_keep_their_identities() {
    // while their centers are close the detector draws one box around both and
    // may miss a few frames: after the crossing each walker must keep its own id
    let mut swapped = Vec::new();
    for speed in [0.004f32, 0.008, 0.012, 0.02] {
      for width in [0.08f32, 0.12] {
        for near in [0.6f32, 1.0, 1.4] {
          for misses in [0, 3, 6] {
            for dy in [0.0f32, 0.05] {
              let mut world = CameraWorld::new(WorldConfig::default());
              world.set_min_confidence(0.5);
              let person = |x: f32, y: f32, w: f32, h: f32, confidence: f32| {
                let mut d = box_at(x, y, w, h, "person");
                d.confidence = confidence;
                d
              };
              let (height, ya, yb) = (0.32f32, 0.3f32, 0.3 + dy);
              let (mut ax, mut bx) = (0.2f32, 0.8 - width);
              let mut t = 1_000_000.0;
              let mut merged = 0;
              let mut first: Option<(u32, u32)> = None;
              let ingest = |world: &mut CameraWorld, t: f64, dets: &[Detection]| {
                let mut ids: Vec<(f32, u32)> = world
                  .ingest(t, dets, None)
                  .tracked
                  .iter()
                  .map(|s| (s.x, s.track_id))
                  .collect();
                ids.sort_by(|a, b| a.0.total_cmp(&b.0));
                ids
              };
              while ax < 0.8 - width && bx > 0.2 {
                let dets = if ((ax - bx).abs()) < width * near {
                  merged += 1;
                  let (gx, gy) = (ax.min(bx), ya.min(yb));
                  let group = person(
                    gx,
                    gy,
                    ax.max(bx) + width - gx,
                    ya.max(yb) + height - gy,
                    0.7,
                  );
                  if merged > 1 && merged <= 1 + misses {
                    vec![]
                  } else {
                    vec![group]
                  }
                } else {
                  vec![
                    person(ax, ya, width, height, 0.8),
                    person(bx, yb, width, height, 0.8),
                  ]
                };
                let ids = ingest(&mut world, t, &dets);
                if first.is_none() && ids.len() == 2 {
                  first = Some((ids[0].1, ids[1].1));
                }
                t += 150.0;
                ax += speed;
                bx -= speed;
              }
              let mut last = Vec::new();
              for _ in 0..10 {
                let dets = [
                  person(ax, ya, width, height, 0.8),
                  person(bx, yb, width, height, 0.8),
                ];
                last = ingest(&mut world, t, &dets);
                t += 150.0;
                ax = (ax + speed).min(0.9 - width);
                bx = (bx - speed).max(0.02);
              }
              let (left_first, right_first) = first.expect("both confirmed before they met");
              // the one who started left ends on the right
              if last.len() != 2 || last[1].1 != left_first || last[0].1 != right_first {
                swapped.push((speed, width, near, misses, dy));
              }
            }
          }
        }
      }
    }
    assert!(
      swapped.is_empty(),
      "crossings that lost or swapped an identity: {swapped:?}"
    );
  }

  #[test]
  fn a_round_trip_that_fades_goes_out_of_sight_instead_of_scenery() {
    let mut world = CameraWorld::new(WorldConfig::default());
    let at = |b: [f32; 4]| box_at(b[0], b[1], b[2], b[3], "person");
    let lerp =
      |a: [f32; 4], b: [f32; 4], f: f32| at(std::array::from_fn(|i| a[i] + (b[i] - a[i]) * f));
    // from the trolley to the door and back, stopping beside the trolley
    let trolley_side = [0.25, 0.30, 0.15, 0.44];
    let door = [0.60, 0.11, 0.23, 0.58];
    let transit = [0.32, 0.34, 0.16, 0.64];
    let mut t = 0.0;
    let mut id = None;
    for leg in [(trolley_side, door), (door, transit)] {
      for i in 0..15 {
        let up = world.ingest(t, &[lerp(leg.0, leg.1, i as f32 / 14.0)], None);
        id = id.or(up.tracked.first().map(|s| s.track_id));
        t += 200.0;
      }
    }
    let last_seen = t - 200.0;
    let id = id.expect("confirmed");
    let mut fade = Vec::new();
    while t < 13_000.0 {
      let up = world.ingest(t, &[], None);
      fade.extend(up.events.iter().map(|e| (e.kind(), t)));
      t += 500.0;
    }
    assert_eq!(fade.len(), 1, "one event for the fade: {fade:?}");
    assert_eq!(fade[0].0, "objectLost");
    assert!(fade[0].1 - last_seen <= WorldConfig::default().depart_grace_ms + 500.0);
    // crouched at the trolley: the same identity is back in sight
    let crouched = [0.12, 0.75, 0.21, 0.24];
    let up = world.ingest(t, &[at(crouched)], None);
    assert_eq!(up.tracked.first().map(|s| s.track_id), Some(id));
    assert_eq!(kinds(&up), vec!["objectRecovered"]);
    assert_eq!(up.tracked[0].state, TrackState::Active);
    t += 500.0;
    // she walks off, never farther from the transit box than half its height
    let away = [0.41, 0.20, 0.16, 0.74];
    let mut states = Vec::new();
    for i in 1..=10 {
      let up = world.ingest(t, &[lerp(crouched, away, i as f32 / 10.0)], None);
      states.extend(
        up.tracked
          .iter()
          .filter(|s| s.track_id == id)
          .map(|s| s.state),
      );
      t += 200.0;
    }
    assert_eq!(states.len(), 10);
    assert!(
      states.iter().all(|s| *s == TrackState::Active),
      "a walker held as scenery: {states:?}"
    );
  }

  #[test]
  fn someone_moving_in_place_who_fades_goes_out_of_sight_instead_of_scenery() {
    let mut world = CameraWorld::new(WorldConfig::default());
    let mut t = 0.0;
    // back and forth every 2.5 s, the box never gets half its size away from where it started
    for i in 0..100 {
      let phase = (t / 2_500.0 * std::f64::consts::TAU).sin() as f32;
      world.ingest(
        t,
        &[box_at(0.45 + 0.05 * phase, 0.2, 0.09, 0.27, "person")],
        None,
      );
      if i % 25 == 24 {
        t += 1_500.0;
      }
      t += 200.0;
    }
    let mut events = Vec::new();
    for _ in 0..30 {
      events.extend(kinds(&world.ingest(t, &[], None)));
      t += 500.0;
    }
    assert_eq!(
      events,
      vec!["objectLost"],
      "someone moving in place is no scenery"
    );
  }

  #[test]
  fn a_body_hidden_behind_another_stays_in_sight() {
    let mut world = CameraWorld::new(WorldConfig::default());
    let mut t = 0.0;
    // two people side by side, both moving in place, the boxes overlapping
    let pair = |t: f64| {
      let phase = (t / 2_500.0 * std::f64::consts::TAU).sin() as f32;
      [
        box_at(0.44 + 0.04 * phase, 0.20, 0.09, 0.27, "person"),
        box_at(0.50 + 0.04 * phase, 0.24, 0.09, 0.27, "person"),
      ]
    };
    while t < 10_000.0 {
      world.ingest(t, &pair(t), None);
      t += 200.0;
    }
    // only the front one is detected for 20 s
    while t < 30_000.0 {
      let up = world.ingest(t, &pair(t)[1..], None);
      assert!(
        kinds(&up).is_empty(),
        "the hidden one left the picture: {:?}",
        kinds(&up)
      );
      t += 200.0;
    }
    // both walk out unseen
    let gone = t - 200.0;
    let mut lost = Vec::new();
    while t < 45_000.0 {
      let up = world.ingest(t, &[], None);
      for e in &up.events {
        assert_eq!(e.kind(), "objectLost");
        lost.push(t);
      }
      t += 200.0;
    }
    assert_eq!(lost.len(), 2);
    let grace = WorldConfig::default().depart_grace_ms;
    assert!(
      lost.iter().all(|at| at - gone <= grace + 400.0),
      "spans held past the walk-out: {lost:?}"
    );
  }

  #[test]
  fn a_car_that_faded_beside_its_arrival_spot_parks() {
    let mut world = CameraWorld::new(WorldConfig::default());
    let mut t = 0.0;
    // pulls in past half its width, short of half its diagonal, then stands
    for i in 0..20 {
      let x = 0.30 + (i.min(15) as f32) * 0.0128;
      world.ingest(t, &[box_at(x, 0.5, 0.3, 0.2, "vehicle")], None);
      t += 200.0;
    }
    let mut events = Vec::new();
    for _ in 0..20 {
      events.extend(kinds(&world.ingest(t, &[], None)));
      t += 500.0;
    }
    assert_eq!(events, vec!["objectSettled"]);
  }

  #[test]
  fn a_faded_sitter_uncovered_by_a_passer_by_stays_put() {
    let mut world = CameraWorld::new(WorldConfig::default());
    let mut t = 0.0;
    for _ in 0..6 {
      world.ingest(t, &[box_at(0.40, 0.25, 0.12, 0.30, "person")], None);
      t += 200.0;
    }
    while t < 8_000.0 {
      world.ingest(t, &[], None);
      t += 500.0;
    }
    // a passer-by hides all but the head, then uncovers the rest of the body
    let mut woke = 0;
    for i in 0..28 {
      let shown = 0.13 + 0.87 * (i as f32 - 5.0).clamp(0.0, 16.0) / 16.0;
      let seen = box_at(0.40, 0.25, 0.12, 0.30 * shown, "person");
      let up = world.ingest(t, &[seen], None);
      woke += up
        .events
        .iter()
        .filter(|e| e.kind() == "objectWoke")
        .count();
      t += 200.0;
    }
    assert_eq!(woke, 0, "a sitter uncovered in place woke");
  }

  fn lerp_box(a: [f32; 4], b: [f32; 4], f: f32) -> Detection {
    let b: [f32; 4] = std::array::from_fn(|i| a[i] + (b[i] - a[i]) * f);
    box_at(b[0], b[1], b[2], b[3], "person")
  }

  #[test]
  fn a_walker_coming_at_the_camera_leaves_instead_of_settling() {
    let mut world = CameraWorld::new(WorldConfig::default());
    world.set_min_confidence(0.5);
    // in through the door on the right, straight at the camera: the box grows,
    // its center hardly moves, the feet leave the frame at the bottom
    let door = [0.90, 0.37, 0.084, 0.49];
    let close = [0.84, 0.24, 0.14, 0.76];
    let mut t = 0.0;
    let mut id = None;
    for i in 0..12 {
      let up = world.ingest(t, &[lerp_box(door, close, i as f32 / 11.0)], None);
      id = id.or(up.tracked.first().map(|s| s.track_id));
      t += 200.0;
    }
    let id = id.expect("confirmed");
    let (mut departed, mut settled) = (false, false);
    while t < 37_000.0 {
      let up = world.ingest(t, &[], None);
      departed |= kinds(&up).contains(&"objectDeparted");
      settled |= kinds(&up).contains(&"objectSettled");
      t += 500.0;
    }
    assert!(
      departed && !settled,
      "she passed the camera, she is no scenery"
    );
    // back at the same edge half a minute later, walking into the room
    let back = [0.889, 0.195, 0.110, 0.794];
    let room = [0.899, 0.381, 0.073, 0.464];
    let mut seen = Vec::new();
    for i in 0..25 {
      let up = world.ingest(t, &[lerp_box(back, room, i as f32 / 24.0)], None);
      seen.extend(up.tracked.iter().map(|s| (s.track_id, s.state)));
      t += 200.0;
    }
    assert!(
      !seen.is_empty()
        && seen
          .iter()
          .all(|(sid, state)| *sid != id && *state == TrackState::Active),
      "her return is a held still identity: {seen:?}"
    );
  }

  #[test]
  fn a_visitor_waiting_close_at_the_edge_is_held() {
    let mut world = CameraWorld::new(WorldConfig::default());
    // walks up to the door, then waits there cut off at the bottom
    let path = [0.40, 0.30, 0.10, 0.30];
    let door = [0.36, 0.30, 0.20, 0.70];
    let mut t = 0.0;
    let mut id = None;
    for i in 0..12 {
      let up = world.ingest(t, &[lerp_box(path, door, i as f32 / 11.0)], None);
      id = id.or(up.tracked.first().map(|s| s.track_id));
      t += 200.0;
    }
    for i in 0..40 {
      let jitter = if i % 2 == 0 { 0.0 } else { 0.004 };
      let waiting = [door[0] + jitter, door[1], door[2] - jitter, door[3]];
      world.ingest(t, &[lerp_box(waiting, waiting, 0.0)], None);
      t += 200.0;
    }
    // the detector loses him for a while, he is still at the door
    let resumed = t + 20_000.0;
    while t < resumed {
      let up = world.ingest(t, &[], None);
      assert!(
        !kinds(&up).contains(&"objectDeparted"),
        "a visitor waiting at the door left"
      );
      t += 500.0;
    }
    let up = world.ingest(t, &[lerp_box(door, door, 0.0)], None);
    assert_eq!(up.tracked.first().map(|s| s.track_id), id);
  }

  #[test]
  fn a_view_growing_back_at_the_edge_before_a_fade_is_no_leaver() {
    let mut world = CameraWorld::new(WorldConfig::default());
    // leaning on the window at the top edge, seen head only for a moment, the
    // box growing back over the body, then lost for a while
    let body = [0.63, 0.0, 0.235, 0.53];
    let head = [0.80, 0.011, 0.064, 0.247];
    let mut t = 0.0;
    for _ in 0..15 {
      world.ingest(t, &[lerp_box(body, body, 0.0)], None);
      t += 200.0;
    }
    for i in 0..=8 {
      world.ingest(t, &[lerp_box(head, body, i as f32 / 8.0)], None);
      t += 200.0;
    }
    while t < 30_000.0 {
      let up = world.ingest(t, &[], None);
      assert!(
        !kinds(&up).contains(&"objectDeparted"),
        "a view growing back read as walking at the camera"
      );
      t += 500.0;
    }
  }

  #[test]
  fn a_settled_person_walking_away_from_the_camera_wakes() {
    let config = WorldConfig::default();
    let mut world = CameraWorld::new(WorldConfig::default());
    // stands close to the camera long enough to settle
    let near = [0.84, 0.23, 0.16, 0.76];
    let mut t = 0.0;
    while t <= config.settle_person_ms + 1_000.0 {
      world.ingest(t, &[lerp_box(near, near, 0.0)], None);
      t += 200.0;
    }
    assert!(world
      .tracks
      .values()
      .all(|tr| tr.state == TrackState::Stationary));
    // then walks straight into the room: the box shrinks, the center stays
    let far = [0.90, 0.38, 0.06, 0.38];
    let mut states = Vec::new();
    for i in 1..=20 {
      let up = world.ingest(t, &[lerp_box(near, far, i as f32 / 20.0)], None);
      states.extend(up.tracked.iter().map(|s| s.state));
      t += 200.0;
    }
    assert_eq!(
      states.last(),
      Some(&TrackState::Active),
      "a walker held as scenery: {states:?}"
    );
  }

  #[test]
  fn a_settled_photo_or_a_view_flap_never_wakes() {
    let config = WorldConfig::default();
    let settle = |world: &mut CameraWorld, b: [f32; 4]| {
      let mut t = 0.0;
      while t <= config.settle_person_ms + 1_000.0 {
        world.ingest(t, &[lerp_box(b, b, 0.0)], None);
        t += 100.0;
      }
      t
    };
    let woke = |up: &WorldUpdate| kinds(up).contains(&"objectWoke");

    // a framed photo read as a person: its box size jumps between detector
    // views for minutes and never goes anywhere
    let photo = [0.33, 0.35, 0.116, 0.284];
    let mut world = CameraWorld::new(WorldConfig::default());
    let mut t = settle(&mut world, photo);
    let mut seed = 7u32;
    let mut next = || {
      seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
      (seed >> 8) as f32 / (1u32 << 24) as f32
    };
    let (mut w, mut h, mut hold) = (photo[2], photo[3], 0);
    for _ in 0..3_000 {
      if hold == 0 {
        w = 0.081 + 0.061 * next();
        h = 0.264 + 0.044 * next();
        hold = 1 + (next() * 20.0) as u32;
      }
      hold -= 1;
      let seen = [photo[0], photo[1], w, h];
      assert!(!woke(&world.ingest(t, &[lerp_box(seen, seen, 0.0)], None)));
      t += 100.0;
    }

    // a sitter seen head only, the box growing back over the body for a few
    // seconds, and a view that swaps to a larger box once and stays there
    let body = [0.63, 0.015, 0.235, 0.53];
    let head = [0.80, 0.011, 0.064, 0.247];
    let wide = [0.60, 0.0, 0.33, 0.74];
    let mut world = CameraWorld::new(WorldConfig::default());
    let mut t = settle(&mut world, body);
    let sitter: Vec<u32> = world.tracks.keys().copied().collect();
    for _ in 0..3 {
      for i in 0..=25 {
        let up = world.ingest(t, &[lerp_box(head, body, i as f32 / 25.0)], None);
        assert!(!woke(&up), "a view growing back over the body woke");
        let ids: Vec<u32> = up.tracked.iter().map(|s| s.track_id).collect();
        assert_eq!(ids, sitter, "the flap left the sitter's identity");
        t += 100.0;
      }
      for _ in 0..30 {
        assert!(!woke(&world.ingest(t, &[lerp_box(body, body, 0.0)], None)));
        t += 100.0;
      }
    }
    for _ in 0..50 {
      let up = world.ingest(t, &[lerp_box(wide, wide, 0.0)], None);
      assert!(!woke(&up), "a view swap woke");
      t += 100.0;
    }

    // bends down, straightens up, and one box takes in the person beside her
    let standing = [0.50, 0.35, 0.25, 0.55];
    let bent = [0.50, 0.45, 0.246, 0.505];
    let rising = [0.50, 0.40, 0.307, 0.596];
    let group = [0.54, 0.15, 0.36, 0.85];
    let beside = [0.61, 0.19, 0.27, 0.53];
    let mut world = CameraWorld::new(WorldConfig::default());
    let mut t = settle(&mut world, standing);
    let seen = |world: &mut CameraWorld, b: [f32; 4], n: usize, t: &mut f64| {
      for _ in 0..n {
        assert!(
          !woke(&world.ingest(*t, &[lerp_box(b, b, 0.0)], None)),
          "one group box woke her"
        );
        *t += 200.0;
      }
    };
    seen(&mut world, bent, 10, &mut t);
    seen(&mut world, rising, 5, &mut t);
    seen(&mut world, group, 1, &mut t);
    seen(&mut world, beside, 15, &mut t);
  }

  #[test]
  fn an_occluded_traveler_goes_out_of_sight_and_keeps_its_identity() {
    let mut world = CameraWorld::new(WorldConfig::default());
    // walks far enough to be a traveler, not far enough to have left
    let mut id = None;
    for i in 0..15 {
      let up = world.ingest(
        i as f64 * 200.0,
        &[person_at(0.30 + i as f32 * 0.015, 0.4)],
        None,
      );
      id = id.or(up.tracked.first().map(|s| s.track_id));
    }
    let mut events = Vec::new();
    for i in 15..65 {
      events.extend(kinds(&world.ingest(i as f64 * 200.0, &[], None)));
    }
    assert_eq!(
      events,
      vec!["objectLost"],
      "an occluded traveler is neither scenery nor gone"
    );
    let up = world.ingest(13_000.0, &[person_at(0.51, 0.4)], None);
    assert_eq!(up.tracked.first().map(|s| s.track_id), id);
    assert_eq!(kinds(&up), vec!["objectRecovered"]);
  }

  #[test]
  fn a_body_back_within_the_still_grace_keeps_its_still_clock() {
    let mut world = CameraWorld::new(WorldConfig::default());
    let mut t = 0.0;
    // walks up, then stands
    for i in 0..10 {
      world.ingest(t, &[person_at(0.30 + i as f32 * 0.02, 0.4)], None);
      t += 200.0;
    }
    let stood = t;
    while t < stood + 3_000.0 {
      world.ingest(t, &[person_at(0.50, 0.4)], None);
      t += 200.0;
    }
    // out of sight for a while, back on the same spot
    while t < stood + 20_000.0 {
      world.ingest(t, &[], None);
      t += 200.0;
    }
    let mut settled = None;
    while settled.is_none() && t < stood + 120_000.0 {
      let up = world.ingest(t, &[person_at(0.50, 0.4)], None);
      if kinds(&up).contains(&"objectSettled") {
        settled = Some(t);
      }
      t += 200.0;
    }
    let settled = settled.expect("standing still settles");
    let settle_ms = WorldConfig::default().settle_person_ms;
    assert!(
      settled - stood <= settle_ms + 400.0,
      "the clock restarted on return: {}",
      settled - stood
    );
  }

  #[test]
  fn sparse_parked_flickers_never_stamp_a_second_visit() {
    // the Eingang phantom: a huge edge sliver seen for a moment, then again
    // half a minute later, held an event open for a minute
    let mut world = CameraWorld::new(WorldConfig::default());
    let sliver = || Detection {
      x: 0.0,
      y: 0.13,
      width: 0.33,
      height: 0.87,
      confidence: 0.75,
      label: "vehicle".to_string(),
    };
    let mut entered = 0;
    let mut settled = 0;
    for i in 0..4 {
      let up = world.ingest(i as f64 * 200.0, &[sliver()], None);
      entered += up
        .events
        .iter()
        .filter(|e| e.kind() == "objectEntered")
        .count();
    }
    for i in 4..40 {
      let up = world.ingest(i as f64 * 200.0, &[], None);
      settled += up
        .events
        .iter()
        .filter(|e| e.kind() == "objectSettled")
        .count();
    }
    assert_eq!(entered, 1);
    assert_eq!(settled, 1);

    // half a minute of silence crosses the decode-gap reset, so the identity
    // sleeps; the resight may recover it, but only as stationary scenery
    let up = world.ingest(30_000.0, &[sliver()], None);
    assert!(up.created.is_empty());
    for e in &up.events {
      match e {
        SemanticEvent::ObjectEntered(_) | SemanticEvent::ObjectWoke(_) => {
          panic!("a parked flicker resight read as an arrival")
        }
        SemanticEvent::ObjectRecovered(s) => assert_eq!(s.state, TrackState::Stationary),
        _ => {}
      }
    }
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
    let update = world.ingest(60_200.0, &[person(0.35)], None);
    assert!(
      update.crossings.is_empty(),
      "the side seen before the gap is not where it crossed"
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
    // a still object that fades mid-frame is held through the still grace
    assert_eq!(lost, 0);
    assert_eq!(departed, 0);

    let up = world.ingest(11_000.0, &[person(0.5)], None);
    assert!(up.created.is_empty(), "resume keeps the identity");
    assert_eq!(
      up.events
        .iter()
        .filter(|e| e.kind() == "objectRecovered")
        .count(),
      0,
      "a re-sighting within the still grace is silent continuation"
    );

    let mut lost = 0;
    let mut departed = 0;
    for i in 1..=350 {
      let up = world.ingest(11_000.0 + i as f64 * 200.0, &[], None);
      lost += up
        .events
        .iter()
        .filter(|e| e.kind() == "objectLost")
        .count();
      departed += up.removed.len();
    }
    // never seen moving: past the still grace it sleeps, not departs
    assert_eq!(lost, 1);
    assert_eq!(departed, 0);

    let up = world.ingest(11_000.0 + 351.0 * 200.0, &[person(0.5)], None);
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

  // the cascade paused while an animal walked off: its span must not stay open for hours
  #[test]
  fn a_gap_sleeper_that_stays_away_is_lost() {
    let grace = WorldConfig::default().depart_grace_ms;
    let mut world = CameraWorld::new(WorldConfig::default());
    for i in 0..8 {
      world.ingest(
        1_000.0 + i as f64 * 200.0,
        &[person(0.3 + i as f32 * 0.02)],
        None,
      );
    }
    let mut lost = Vec::new();
    let mut t = 10_000.0;
    while t <= 10_000.0 + grace * 2.0 {
      let up = world.ingest(t, &[], None);
      lost.extend(
        up.events
          .iter()
          .filter(|e| e.kind() == "objectLost")
          .map(|_| t),
      );
      t += 200.0;
    }
    assert_eq!(lost.len(), 1, "lost once: {lost:?}");
    assert!(
      lost[0] > 10_000.0 + grace,
      "the grace runs from the resume, lost at {}",
      lost[0]
    );
  }

  #[test]
  fn a_gap_sleeper_back_within_the_grace_is_continued() {
    let mut world = CameraWorld::new(WorldConfig::default());
    for i in 0..8 {
      world.ingest(
        1_000.0 + i as f64 * 200.0,
        &[person(0.3 + i as f32 * 0.02)],
        None,
      );
    }
    world.ingest(10_000.0, &[], None);
    let up = world.ingest(12_000.0, &[person(0.44)], None);
    assert!(up.created.is_empty(), "the identity is kept");
    assert_eq!(kinds(&up), vec!["objectRecovered"]);
    // walks on past the old grace: the resume ended the hold
    for i in 1..=40 {
      let up = world.ingest(
        12_000.0 + i as f64 * 200.0,
        &[person(0.44 + i as f32 * 0.005)],
        None,
      );
      assert!(
        !kinds(&up).contains(&"objectLost"),
        "lost while seen at tick {i}"
      );
    }
  }

  #[test]
  fn a_sleeper_dropped_after_a_long_gap_is_removed() {
    let mut world = CameraWorld::new(WorldConfig::default());
    for i in 0..8 {
      world.ingest(
        1_000.0 + i as f64 * 200.0,
        &[person(0.3 + i as f32 * 0.02)],
        None,
      );
    }
    let up = world.ingest(
      1_000.0 + WorldConfig::default().dormant_animate_ms * 2.0,
      &[],
      None,
    );
    assert_eq!(up.removed, vec![1]);
  }

  #[test]
  fn a_camera_move_reports_the_tracks_it_drops() {
    let mut world = CameraWorld::new(WorldConfig::default());
    let mut t = 0.0;
    while t <= WorldConfig::default().settle_vehicle_ms + 1_000.0 {
      world.ingest(t, &[box_at(0.2, 0.5, 0.3, 0.2, "vehicle")], None);
      t += 200.0;
    }
    world.notify_camera_move();
    let up = world.ingest(t, &[], None);
    assert_eq!(up.removed, vec![1], "the parked car went with the move");
    let up = world.ingest(t + 200.0, &[], None);
    assert!(up.removed.is_empty(), "reported once");
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
          t.speed < 0.01,
          "pan must not read as movement, got {}/s",
          t.speed
        );
      }
    }
    assert!(created >= 1);
    assert_eq!(woke, 0, "a pan must never wake or move a compensated track");
    assert_eq!(crossings, 0, "the pan carried the person across the line");
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

  fn box_at(x: f32, y: f32, width: f32, height: f32, label: &str) -> Detection {
    Detection {
      x,
      y,
      width,
      height,
      confidence: 0.9,
      label: label.to_string(),
    }
  }

  // settles a track at the given box, then lets it fade into sleep
  fn sleeper(world: &mut CameraWorld, det: &Detection, settle_ms: f64) -> f64 {
    let mut t = 0.0;
    while t <= settle_ms + 1_000.0 {
      world.ingest(t, std::slice::from_ref(det), None);
      t += 200.0;
    }
    let mut gone = t;
    while gone - t <= WorldConfig::default().still_lost_grace_ms + 1_000.0 {
      world.ingest(gone, &[], None);
      gone += 1_000.0;
    }
    gone
  }

  #[test]
  fn a_visitor_minutes_after_a_sitter_is_an_arrival() {
    // the gate: everyone appears at the same spot, small and far
    let gate = box_at(0.17, 0.49, 0.05, 0.27, "person");
    let mut world = CameraWorld::new(WorldConfig::default());
    let quiet = sleeper(&mut world, &gate, WorldConfig::default().settle_person_ms);

    let later = quiet + 13.0 * 60_000.0;
    let mut entered = 0;
    let mut recovered = 0;
    for i in 0..4 {
      let up = world.ingest(later + i as f64 * 200.0, std::slice::from_ref(&gate), None);
      entered += up
        .events
        .iter()
        .filter(|e| e.kind() == "objectEntered")
        .count();
      recovered += up
        .events
        .iter()
        .filter(|e| e.kind() == "objectRecovered")
        .count();
    }
    assert_eq!(
      entered, 1,
      "a person at the gate minutes later is a new visitor"
    );
    assert_eq!(recovered, 0, "the old sitter must not claim the visitor");
    assert!(world.tracks.values().all(|t| t.state == TrackState::Active));
  }

  #[test]
  fn a_parked_car_hours_later_is_still_parked() {
    let car = box_at(0.5, 0.5, 0.3, 0.2, "vehicle");
    let mut world = CameraWorld::new(WorldConfig::default());
    let quiet = sleeper(&mut world, &car, WorldConfig::default().settle_vehicle_ms);

    let up = world.ingest(quiet + 3.0 * 3_600_000.0, &[car], None);
    assert!(up.created.is_empty(), "the car keeps its identity");
    for e in &up.events {
      match e {
        SemanticEvent::ObjectEntered(_) | SemanticEvent::ObjectWoke(_) => {
          panic!("a parked car read as an arrival")
        }
        SemanticEvent::ObjectRecovered(s) => assert_eq!(s.state, TrackState::Stationary),
        _ => {}
      }
    }
  }

  #[test]
  fn a_sleeper_does_not_claim_the_next_person_over() {
    let sitter = box_at(0.17, 0.49, 0.05, 0.27, "person");
    let mut world = CameraWorld::new(WorldConfig::default());
    let quiet = sleeper(&mut world, &sitter, WorldConfig::default().settle_person_ms);

    // three body widths to the side, within the sleep window
    let neighbour = box_at(0.32, 0.5, 0.05, 0.2, "person");
    let mut entered = 0;
    for i in 0..4 {
      let up = world.ingest(
        quiet + 30_000.0 + i as f64 * 200.0,
        std::slice::from_ref(&neighbour),
        None,
      );
      entered += up
        .events
        .iter()
        .filter(|e| e.kind() == "objectEntered")
        .count();
    }
    assert_eq!(
      entered, 1,
      "a person beside the sleeper is their own identity"
    );
  }

  #[test]
  fn a_camera_report_lets_a_single_sighting_confirm() {
    let mut world = CameraWorld::new(WorldConfig::default());
    // without a report a lone sighting never becomes anyone
    world.ingest(0.0, &[person(0.3)], None);
    for i in 1..30 {
      let up = world.ingest(i as f64 * 200.0, &[], None);
      assert!(up.created.is_empty() && up.events.is_empty());
    }

    // the camera saw a person too, a moment before our one frame
    world.attest("person", 10_000.0);
    let up = world.ingest(10_400.0, &[person(0.3)], None);
    assert_eq!(
      up.created.len(),
      1,
      "two detectors agreeing confirm at once"
    );
    assert!(up.events.iter().any(|e| e.kind() == "objectEntered"));
  }

  #[test]
  fn a_report_landing_after_the_sighting_still_confirms() {
    let mut world = CameraWorld::new(WorldConfig::default());
    world.ingest(0.0, &[person(0.3)], None);
    world.attest("person", 800.0);
    let up = world.ingest(1_000.0, &[], None);
    assert_eq!(up.created.len(), 1);
    let entered = up.events.iter().find(|e| e.kind() == "objectEntered");
    assert!(
      entered.is_some(),
      "the sighting confirms with the box it left"
    );
  }

  #[test]
  fn a_report_of_another_label_or_another_time_confirms_nothing() {
    let mut world = CameraWorld::new(WorldConfig::default());
    world.attest("vehicle", 0.0);
    let up = world.ingest(100.0, &[person(0.3)], None);
    assert!(
      up.created.is_empty(),
      "a vehicle report says nothing about a person"
    );

    world.attest("person", 100.0);
    let up = world.ingest(20_000.0, &[person(0.6)], None);
    assert!(
      up.created.is_empty(),
      "a report from long ago is no witness"
    );
    let up = world.ingest(20_200.0, &[], None);
    assert!(up.created.is_empty());
  }

  #[test]
  fn a_first_sighting_is_reported_before_it_confirms() {
    let mut world = CameraWorld::new(WorldConfig::default());
    let up = world.ingest(0.0, &[person(0.3)], None);
    assert_eq!(up.sightings.len(), 1);
    assert_eq!(up.sightings[0].state, TrackState::Tentative);
    assert!(up.tracked.is_empty() && up.created.is_empty());

    let up = world.ingest(200.0, &[person(0.3)], None);
    assert!(
      up.sightings.is_empty(),
      "only the first sighting is reported"
    );
    let up = world.ingest(600.0, &[person(0.3)], None);
    assert_eq!(up.created.len(), 1);
    assert_eq!(up.tracked[0].last_seen_ms, 600.0);
  }

  #[test]
  fn a_late_witness_confirms_with_the_sighting_time() {
    let mut world = CameraWorld::new(WorldConfig::default());
    world.ingest(0.0, &[person(0.3)], None);
    world.attest("person", 800.0);
    let up = world.ingest(1_000.0, &[], None);
    let entered = up
      .events
      .iter()
      .find_map(|e| match e {
        SemanticEvent::ObjectEntered(s) => Some(s),
        _ => None,
      })
      .expect("the sighting confirms");
    assert_eq!(
      entered.last_seen_ms, 0.0,
      "the snapshot names the sighting, not the tick"
    );
    assert_eq!(
      up.tracked.len(),
      1,
      "the withheld sighting is delivered with this tick"
    );
    assert_eq!(up.tracked[0].last_seen_ms, 0.0);
    let up = world.ingest(1_200.0, &[], None);
    assert!(up.tracked.is_empty(), "delivered once, not every tick");
  }

  #[test]
  fn an_entered_snapshot_says_whether_a_witness_confirmed_it() {
    let entered = |up: &WorldUpdate| {
      up.events.iter().find_map(|e| match e {
        SemanticEvent::ObjectEntered(s) => Some(s.attested),
        _ => None,
      })
    };
    // the usual way: two sightings
    let mut world = CameraWorld::new(WorldConfig::default());
    world.ingest(0.0, &[person(0.3)], None);
    let up = world.ingest(600.0, &[person(0.3)], None);
    assert_eq!(entered(&up), Some(false));

    // a report before the sighting decides at once
    let mut world = CameraWorld::new(WorldConfig::default());
    world.attest("person", 0.0);
    let up = world.ingest(100.0, &[person(0.3)], None);
    assert_eq!(entered(&up), Some(true));

    // a report after the only sighting decides in the retain pass
    let mut world = CameraWorld::new(WorldConfig::default());
    world.ingest(0.0, &[person(0.3)], None);
    world.attest("person", 500.0);
    let up = world.ingest(1_000.0, &[], None);
    assert_eq!(entered(&up), Some(true));
  }

  #[test]
  fn a_recovered_walker_does_not_settle_on_arrival() {
    let mut world = CameraWorld::new(WorldConfig::default());
    // walks in and stands for twenty seconds, well short of settling
    let mut t = 0.0;
    for i in 0..15 {
      world.ingest(t, &[person(0.1 + i as f32 * 0.025)], None);
      t += 400.0;
    }
    while t < 26_000.0 {
      world.ingest(t, &[person(0.45)], None);
      t += 400.0;
    }
    // unseen past the still grace: the track sleeps
    let gone = t + WorldConfig::default().still_lost_grace_ms + 10_000.0;
    while t < gone {
      world.ingest(t, &[], None);
      t += 400.0;
    }
    let up = world.ingest(t, &[person(0.45)], None);
    assert!(up.events.iter().any(|e| e.kind() == "objectRecovered"));
    assert!(
      !up.events.iter().any(|e| e.kind() == "objectSettled"),
      "the minutes unseen must not count as standing still"
    );
    assert!(up.tracked.iter().all(|s| s.state == TrackState::Active));
  }

  fn ids_unique(up: &WorldUpdate) -> bool {
    let mut ids: Vec<u32> = up.tracked.iter().map(|t| t.track_id).collect();
    let n = ids.len();
    ids.sort_unstable();
    ids.dedup();
    ids.len() == n
  }

  #[test]
  fn a_fragment_never_shares_an_identity_with_its_refound_body() {
    let body = box_at(0.57, 0.15, 0.06, 0.21, "person");
    let legs = box_at(0.572, 0.289, 0.034, 0.071, "person");
    let mut world = CameraWorld::new(WorldConfig::default());
    let mut t = 0.0;
    while t <= 3_000.0 {
      world.ingest(t, std::slice::from_ref(&body), None);
      t += 200.0;
    }
    // unseen long enough to count as orphaned, short enough for the engine to
    // still hold its lost track
    while t <= 5_400.0 {
      world.ingest(t, &[], None);
      t += 200.0;
    }
    for _ in 0..10 {
      let up = world.ingest(t, &[body.clone(), legs.clone()], None);
      assert!(ids_unique(&up), "two engine tracks wrote one identity");
      t += 200.0;
    }
  }

  #[test]
  fn an_identity_that_moved_on_does_not_return_to_its_old_engine_track() {
    let mut world = CameraWorld::new(WorldConfig::default());
    let mut t = 0.0;
    while t <= 3_000.0 {
      world.ingest(t, &[box_at(0.44, 0.1, 0.06, 0.3, "person")], None);
      t += 200.0;
    }
    while t <= 5_400.0 {
      world.ingest(t, &[], None);
      t += 200.0;
    }
    // back a little to the side, too far for the engine, near enough to resume
    let mut walker_id = None;
    for i in 0..6 {
      let walker = box_at(0.49, 0.1 + i as f32 * 0.04, 0.06, 0.3, "person");
      let up = world.ingest(t, &[walker], None);
      walker_id = walker_id.or(up.tracked.first().map(|s| s.track_id));
      t += 200.0;
    }
    let walker_id = walker_id.expect("the walker resumed an identity");
    // someone else steps onto the old spot while the walker is out of sight
    let mut stranger = Vec::new();
    for _ in 0..5 {
      let up = world.ingest(t, &[box_at(0.44, 0.1, 0.06, 0.3, "person")], None);
      stranger.extend(up.tracked.iter().map(|s| s.track_id));
      t += 200.0;
    }
    assert!(
      !stranger.contains(&walker_id),
      "the stranger took over the walker's identity"
    );
  }

  #[test]
  fn a_threshold_below_the_engine_default_still_lets_objects_in() {
    let mut world = CameraWorld::new(WorldConfig::default());
    world.set_min_confidence(0.4);
    let mut created = 0;
    for i in 0..20 {
      let mut d = person(0.2 + i as f32 * 0.01);
      d.confidence = 0.45;
      created += world.ingest(i as f64 * 200.0, &[d], None).created.len();
    }
    assert_eq!(
      created, 1,
      "a 0.45 person under a 0.4 threshold is an arrival"
    );
  }

  #[test]
  fn weak_sightings_carry_a_track_but_never_confirm_one() {
    let mut world = CameraWorld::new(WorldConfig::default());
    world.set_min_confidence(0.7);
    let seen = |x: f32, confidence: f32| {
      let mut d = person(x);
      d.confidence = confidence;
      d
    };
    // one sighting above the threshold, then only weak ones: a stone read as a
    // person once is not a person
    let mut created = 0;
    world.ingest(0.0, &[seen(0.3, 0.75)], None);
    for i in 1..20 {
      created += world
        .ingest(i as f64 * 200.0, &[seen(0.3, 0.35)], None)
        .created
        .len();
    }
    assert_eq!(created, 0, "weak sightings confirmed a newcomer");

    // a confirmed walker keeps its identity through a weak stretch
    let mut world = CameraWorld::new(WorldConfig::default());
    world.set_min_confidence(0.7);
    let mut ids = HashSet::new();
    for i in 0..30 {
      let confidence = if (10..20).contains(&i) { 0.35 } else { 0.8 };
      let up = world.ingest(
        i as f64 * 200.0,
        &[seen(0.2 + i as f32 * 0.01, confidence)],
        None,
      );
      ids.extend(up.tracked.iter().map(|s| s.track_id));
      if (10..20).contains(&i) {
        assert_eq!(up.tracked.len(), 1, "weak tick {i} dropped the walker");
        assert_eq!(
          up.tracked[0].confidence, 0.8,
          "a weak tick reported its own score"
        );
        assert_eq!(
          up.tracked[0].score, 0.35,
          "the weak tick's own sighting got lost"
        );
      }
    }
    assert_eq!(ids.len(), 1, "the weak stretch split the walker");
  }

  fn kinds(up: &WorldUpdate) -> Vec<&'static str> {
    up.events.iter().map(|e| e.kind()).collect()
  }

  #[test]
  fn a_weak_sighting_never_resumes_a_sleeper() {
    let mut world = CameraWorld::new(WorldConfig::default());
    world.set_min_confidence(0.7);
    let seen = |x: f32, confidence: f32| {
      let mut d = person(x);
      d.confidence = confidence;
      d
    };
    for i in 0..10 {
      world.ingest(i as f64 * 200.0, &[seen(0.2 + i as f32 * 0.02, 0.8)], None);
    }
    // the cascade pauses: every track sleeps across the decode gap
    let mut t = 10_000.0;
    for _ in 0..5 {
      let up = world.ingest(t, &[seen(0.38, 0.6)], None);
      assert!(
        up.tracked.is_empty() && up.events.is_empty(),
        "a weak sighting woke the sleeper: {:?}",
        kinds(&up)
      );
      t += 200.0;
    }
    let up = world.ingest(t, &[seen(0.38, 0.8)], None);
    assert!(
      kinds(&up).contains(&"objectRecovered"),
      "a strong sighting resumes it"
    );
    assert!(up.created.is_empty(), "the identity is kept");
  }

  #[test]
  fn only_a_strong_sighting_completes_a_wake() {
    let mut world = CameraWorld::new(WorldConfig::default());
    world.set_min_confidence(0.7);
    let car = |x: f32, confidence: f32| {
      let mut d = box_at(x, 0.5, 0.3, 0.2, "vehicle");
      d.confidence = confidence;
      d
    };
    let mut t = 0.0;
    while t <= WorldConfig::default().settle_vehicle_ms + 2_000.0 {
      world.ingest(t, &[car(0.2, 0.8)], None);
      t += 200.0;
    }
    // drives off, seen only below the threshold
    let mut x = 0.2;
    for _ in 0..30 {
      x += 0.015;
      let up = world.ingest(t, &[car(x, 0.6)], None);
      assert!(
        !kinds(&up).contains(&"objectWoke"),
        "weak sightings woke the parked car"
      );
      t += 200.0;
    }
    x += 0.015;
    let up = world.ingest(t, &[car(x, 0.8)], None);
    assert!(
      kinds(&up).contains(&"objectWoke"),
      "the first strong sighting on the way completes the wake"
    );
  }

  #[test]
  fn reported_speed_is_per_second_whatever_the_tick_rate() {
    // 0.1 frame widths per second, seen every 100 ms and every 200 ms
    for tick_ms in [100.0_f32, 200.0] {
      let mut world = CameraWorld::new(WorldConfig::default());
      let mut last = None;
      for i in 0..40 {
        let t = i as f32 * tick_ms;
        let up = world.ingest(t as f64, &[person(0.2 + 0.1 * t / 1000.0)], None);
        last = up.tracked.first().map(|s| (s.speed, s.velocity_x));
      }
      let (speed, vx) = last.expect("tracked");
      assert!((speed - 0.1).abs() < 0.01, "{tick_ms} ms ticks: {speed}/s");
      assert!((vx - 0.1).abs() < 0.01);
    }
  }
  #[test]
  fn a_pan_that_outruns_the_engine_keeps_the_identity_at_once() {
    let mut world = CameraWorld::new(WorldConfig::default());
    let mut id = None;
    for i in 0..10 {
      let up = world.ingest(i as f64 * 100.0, &[person(0.5)], Some((0.0, 0.0)));
      id = id.or(up.tracked.first().map(|s| s.track_id));
    }
    let id = id.expect("confirmed");
    // the camera pans right by three box widths between two ticks
    let up = world.ingest(1_000.0, &[person(0.2)], Some((-0.3, 0.0)));
    assert_eq!(
      up.tracked.iter().map(|s| s.track_id).collect::<Vec<_>>(),
      vec![id],
      "the person vanished or was reborn after the pan"
    );
    assert!(up.created.is_empty());
  }

  struct Lcg(u32);

  impl Lcg {
    fn next(&mut self) -> f32 {
      self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
      (self.0 >> 8) as f32 / (1u32 << 24) as f32
    }

    fn jitter(&mut self, amp: f32) -> f32 {
      (self.next() * 2.0 - 1.0) * amp
    }
  }

  fn jittered(b: [f32; 4], rng: &mut Lcg, label: &str) -> Detection {
    box_at(
      b[0] + rng.jitter(0.003),
      b[1] + rng.jitter(0.003),
      b[2] + rng.jitter(0.003),
      b[3] + rng.jitter(0.003),
      label,
    )
  }

  fn settle_still(
    world: &mut CameraWorld,
    dets: &[[f32; 4]],
    labels: &[&str],
    rng: &mut Lcg,
  ) -> f64 {
    let mut t = 0.0;
    while t <= WorldConfig::default().settle_person_ms + 1_000.0 {
      let seen: Vec<Detection> = dets
        .iter()
        .zip(labels)
        .map(|(b, l)| jittered(*b, rng, l))
        .collect();
      world.ingest(t, &seen, None);
      t += 100.0;
    }
    assert!(
      world
        .tracks
        .values()
        .all(|tr| tr.state == TrackState::Stationary),
      "setup: everything settled"
    );
    t
  }

  fn count(up: &WorldUpdate, kind: &str) -> usize {
    up.events.iter().filter(|e| e.kind() == kind).count()
  }

  fn woke_by(up: &WorldUpdate, cause: WakeCause) -> Vec<u32> {
    up.events
      .iter()
      .filter_map(|e| match e {
        SemanticEvent::ObjectWoke(s) if s.wake_cause == Some(cause) => Some(s.track_id),
        _ => None,
      })
      .collect()
  }

  #[test]
  fn a_sitter_gesturing_never_wakes() {
    let sit = [0.40, 0.40, 0.10, 0.22];
    let mut rng = Lcg(11);
    let mut world = CameraWorld::new(WorldConfig::default());
    let mut t = settle_still(&mut world, &[sit], &["person"], &mut rng);
    let poses = [
      sit,
      [0.40, 0.40, 0.16, 0.22],
      [0.34, 0.40, 0.16, 0.22],
      [0.40, 0.34, 0.10, 0.28],
      [0.41, 0.42, 0.12, 0.20],
      [0.37, 0.38, 0.17, 0.25],
    ];
    let mut woke = 0;
    let mut pose = sit;
    let mut hold = 0;
    for _ in 0..6_000 {
      if hold == 0 {
        pose = poses[(rng.next() * poses.len() as f32) as usize % poses.len()];
        hold = 5 + (rng.next() * 25.0) as u32;
      }
      hold -= 1;
      woke += count(
        &world.ingest(t, &[jittered(pose, &mut rng, "person")], None),
        "objectWoke",
      );
      t += 100.0;
    }
    assert_eq!(woke, 0, "a gesture woke the sitter");
  }

  #[test]
  fn a_rocking_chair_never_wakes() {
    let sit = [0.40, 0.40, 0.10, 0.22];
    let mut rng = Lcg(12);
    let mut world = CameraWorld::new(WorldConfig::default());
    let mut t = settle_still(&mut world, &[sit], &["person"], &mut rng);
    let start = t;
    let mut woke = 0;
    for _ in 0..3_000 {
      let phase = ((t - start) / 2_000.0 * std::f64::consts::TAU).sin() as f32;
      let b = [
        sit[0] + 0.03 * phase,
        sit[1] + 0.015 * phase,
        sit[2],
        sit[3] - 0.01 * phase.abs(),
      ];
      woke += count(
        &world.ingest(t, &[jittered(b, &mut rng, "person")], None),
        "objectWoke",
      );
      t += 100.0;
    }
    assert_eq!(woke, 0, "rocking in place woke the sitter");
  }

  #[test]
  fn exercising_on_the_spot_never_wakes() {
    let stand = [0.45, 0.30, 0.08, 0.30];
    let mut rng = Lcg(13);
    let mut world = CameraWorld::new(WorldConfig::default());
    let mut t = settle_still(&mut world, &[stand], &["person"], &mut rng);
    let moves = [
      stand,
      [0.41, 0.28, 0.16, 0.32],
      [0.45, 0.40, 0.08, 0.20],
      [0.43, 0.30, 0.08, 0.30],
      [0.47, 0.30, 0.08, 0.30],
      [0.42, 0.25, 0.14, 0.35],
    ];
    let mut woke = 0;
    for i in 0..3_000usize {
      let b = moves[(i / 5) % moves.len()];
      woke += count(
        &world.ingest(t, &[jittered(b, &mut rng, "person")], None),
        "objectWoke",
      );
      t += 100.0;
    }
    assert_eq!(woke, 0, "exercise in place woke the person");
  }

  #[test]
  fn a_pet_on_a_lap_never_wakes() {
    let sit = [0.40, 0.40, 0.12, 0.24];
    let pet = [0.43, 0.52, 0.06, 0.05];
    let mut rng = Lcg(14);
    let mut world = CameraWorld::new(WorldConfig::default());
    let mut t = settle_still(&mut world, &[sit, pet], &["person", "animal"], &mut rng);
    let spots = [
      [0.41, 0.52, 0.06, 0.05],
      [0.47, 0.53, 0.06, 0.05],
      [0.44, 0.50, 0.05, 0.06],
    ];
    let mut woke = 0;
    for i in 0..3_000usize {
      let leg = (i / 15) % spots.len();
      let next = (leg + 1) % spots.len();
      let f = (i % 15) as f32 / 15.0;
      let p: [f32; 4] =
        std::array::from_fn(|k| spots[leg][k] + (spots[next][k] - spots[leg][k]) * f);
      let person = [sit[0], sit[1], sit[2] + 0.02 * f, sit[3]];
      let up = world.ingest(
        t,
        &[
          jittered(person, &mut rng, "person"),
          jittered(p, &mut rng, "animal"),
        ],
        None,
      );
      woke += up
        .events
        .iter()
        .filter(|e| matches!(e, SemanticEvent::ObjectWoke(s) if s.label == "person"))
        .count();
      t += 100.0;
    }
    assert_eq!(woke, 0, "a pet shifting on a lap woke the person");
  }

  #[test]
  fn a_pet_hopping_off_its_spot_and_back_never_oscillates() {
    // out in steps the engine follows, the last one clears the spot, then hidden: never a travel streak
    let spot = [0.43, 0.52, 0.06, 0.05];
    let mut rng = Lcg(24);
    let mut world = CameraWorld::new(WorldConfig::default());
    let mut t = settle_still(&mut world, &[spot], &["animal"], &mut rng);
    let mut oscillated = 0;
    let mut woke = 0;
    for _ in 0..48 {
      for k in 0..=4 {
        let b = [spot[0] + 0.0125 * k as f32, spot[1], spot[2], spot[3]];
        let up = world.ingest(t, &[jittered(b, &mut rng, "animal")], None);
        oscillated += woke_by(&up, WakeCause::Oscillation).len();
        woke += count(&up, "objectWoke");
        t += 100.0;
      }
      for _ in 0..20 {
        world.ingest(t, &[], None);
        t += 100.0;
      }
    }
    assert_eq!(
      oscillated, 0,
      "a pet hopping off its spot woke by oscillation"
    );
    assert_eq!(woke, 0, "setup: the hops never built a travel streak");
  }

  #[test]
  fn a_view_flap_on_a_fragment_anchor_never_wakes() {
    // walkout multi_hq 263-274 s: recovered on a fragment at the edge, then fragment and body flap
    let fragment = [0.869, 0.525, 0.057, 0.118];
    let views = [
      fragment,
      [0.869, 0.486, 0.093, 0.196],
      [0.918, 0.461, 0.065, 0.246],
      [0.92, 0.431, 0.08, 0.306],
      [0.869, 0.437, 0.131, 0.238],
      [0.915, 0.512, 0.085, 0.226],
    ];
    let mut rng = Lcg(15);
    let mut world = CameraWorld::new(WorldConfig::default());
    let mut t = settle_still(&mut world, &[fragment], &["person"], &mut rng);
    let mut woke = 0;
    for i in 0..1_200usize {
      let b = views[((i * 7 + (rng.next() * 3.0) as usize) / 4) % views.len()];
      woke += count(
        &world.ingest(t, &[jittered(b, &mut rng, "person")], None),
        "objectWoke",
      );
      t += 100.0;
    }
    assert_eq!(woke, 0, "a view flap around a fragment anchor woke");
  }

  #[test]
  fn someone_running_past_a_sitter_does_not_wake_the_sitter() {
    let sit = [0.40, 0.40, 0.10, 0.22];
    let mut rng = Lcg(16);
    let mut world = CameraWorld::new(WorldConfig::default());
    let mut t = settle_still(&mut world, &[sit], &["person"], &mut rng);
    let sitter: Vec<u32> = world.tracks.keys().copied().collect();
    let start = t;
    let mut sitter_woke = false;
    for _ in 0..1_200 {
      let phase = ((t - start) / 3_000.0 * std::f64::consts::TAU).sin() as f32;
      let passer = [0.40 + 0.15 * phase, 0.47, 0.06, 0.15];
      let up = world.ingest(
        t,
        &[
          jittered(sit, &mut rng, "person"),
          jittered(passer, &mut rng, "person"),
        ],
        None,
      );
      sitter_woke |= up
        .events
        .iter()
        .any(|e| matches!(e, SemanticEvent::ObjectWoke(s) if sitter.contains(&s.track_id)));
      t += 100.0;
    }
    assert!(!sitter_woke, "the sitter woke on someone running past");
  }

  #[test]
  fn someone_moving_in_front_of_a_sitter_never_oscillates_the_sitter() {
    // the sitter is hidden while covered, the one in front is seen on half the frames
    for seed in 1..=20u32 {
      let sit = [0.40, 0.40, 0.10, 0.22];
      let mut rng = Lcg(100 + seed);
      let mut world = CameraWorld::new(WorldConfig::default());
      let mut t = settle_still(&mut world, &[sit], &["person"], &mut rng);
      let sitter: Vec<u32> = world.tracks.keys().copied().collect();
      let start = t;
      for _ in 0..3_000 {
        let phase = ((t - start) / 4_000.0 * std::f64::consts::TAU).sin() as f32;
        let front = [0.42 + 0.14 * phase, 0.44, 0.07, 0.17];
        let mut seen = Vec::new();
        if box_iou(&front, &sit) <= 0.15 {
          seen.push(jittered(sit, &mut rng, "person"));
        }
        if rng.next() < 0.5 {
          seen.push(jittered(front, &mut rng, "person"));
        }
        let up = world.ingest(t, &seen, None);
        let oscillated = woke_by(&up, WakeCause::Oscillation);
        assert!(
          !oscillated.iter().any(|id| sitter.contains(id)),
          "seed {seed}: the sitter's identity woke by oscillation on the one in front at {:.1} s",
          (t - start) / 1000.0
        );
        t += 100.0;
      }
    }
  }

  #[test]
  fn single_excursions_with_rest_between_never_wake() {
    let sit = [0.40, 0.40, 0.10, 0.22];
    let mut rng = Lcg(17);
    let mut world = CameraWorld::new(WorldConfig::default());
    let mut t = settle_still(&mut world, &[sit], &["person"], &mut rng);
    let mut woke = 0;
    for k in 0..30 {
      for i in (0..=4).chain((0..=4).rev()) {
        let dx = 0.04 * i as f32 * if k % 2 == 0 { 1.0 } else { -1.0 };
        let b = [sit[0] + dx, sit[1], sit[2], sit[3]];
        woke += count(
          &world.ingest(t, &[jittered(b, &mut rng, "person")], None),
          "objectWoke",
        );
        t += 100.0;
      }
      for _ in 0..200 {
        woke += count(
          &world.ingest(t, &[jittered(sit, &mut rng, "person")], None),
          "objectWoke",
        );
        t += 100.0;
      }
    }
    assert_eq!(
      woke, 0,
      "single excursions with rest in between woke the sitter"
    );
  }

  #[test]
  fn a_sitter_back_outside_the_spot_after_a_break_never_oscillates() {
    let sit = [0.40, 0.40, 0.10, 0.22];
    // close enough for re-association to resume the sitter, far enough to clear the spot
    let beside = [sit[0] + 0.09, sit[1] + 0.08, sit[2], sit[3]];
    // break length, hidden instead of not decoded, first sighting back below the sure margin
    for (gap, hidden, unsure) in [
      (10_000.0, false, false),
      (240_000.0, true, false),
      (10_000.0, false, true),
    ] {
      let mut rng = Lcg(24);
      let mut world = CameraWorld::new(WorldConfig::default());
      world.set_min_confidence(0.8);
      let mut t = settle_still(&mut world, &[sit], &["person"], &mut rng);
      let mut woke = 0;
      let mut recovered = 0;
      // one excursion, back on the spot for a moment, then the break
      for i in [1, 2, 3, 4, 3, 2, 1, 0] {
        let b = [sit[0] + 0.04 * i as f32, sit[1], sit[2], sit[3]];
        woke += count(
          &world.ingest(t, &[jittered(b, &mut rng, "person")], None),
          "objectWoke",
        );
        t += 100.0;
      }
      let resume = t + gap;
      while hidden && t < resume {
        woke += count(&world.ingest(t, &[], None), "objectWoke");
        t += 100.0;
      }
      t = resume;
      for k in 0..5 {
        let mut d = jittered(beside, &mut rng, "person");
        if unsure && k == 0 {
          d.confidence = 0.82;
        }
        let up = world.ingest(t, &[d], None);
        woke += count(&up, "objectWoke");
        recovered += count(&up, "objectRecovered");
        t += 100.0;
      }
      assert_eq!(recovered, 1, "setup: the sitter resumes after {gap} ms");
      assert_eq!(
        woke, 0,
        "one excursion before a {gap} ms break and one after woke the sitter"
      );
    }
  }

  #[test]
  fn a_sitter_who_sits_back_differently_never_wakes_on_one_more_reach_out() {
    let sit = [0.40, 0.40, 0.10, 0.22];
    // centre still inside the old spot, IoU about 0.54: too far for the anchor to follow
    let shifted = [0.43, 0.40, 0.10, 0.22];
    for (rest_s, seed) in [(60u32, 21u32), (600, 22), (1_800, 23)] {
      let mut rng = Lcg(seed);
      let mut world = CameraWorld::new(WorldConfig::default());
      let mut t = settle_still(&mut world, &[sit], &["person"], &mut rng);
      let mut woke = 0;
      let reach =
        |world: &mut CameraWorld, t: &mut f64, from: [f32; 4], dir: f32, rng: &mut Lcg| {
          let mut n = 0;
          for i in [1, 2, 3, 4, 3, 2, 1] {
            let b = [from[0] + dir * 0.04 * i as f32, from[1], from[2], from[3]];
            n += count(
              &world.ingest(*t, &[jittered(b, rng, "person")], None),
              "objectWoke",
            );
            *t += 100.0;
          }
          n
        };
      woke += reach(&mut world, &mut t, sit, 1.0, &mut rng);
      for _ in 0..rest_s * 10 {
        woke += count(
          &world.ingest(t, &[jittered(shifted, &mut rng, "person")], None),
          "objectWoke",
        );
        t += 100.0;
      }
      woke += reach(&mut world, &mut t, shifted, -1.0, &mut rng);
      for _ in 0..100 {
        woke += count(
          &world.ingest(t, &[jittered(shifted, &mut rng, "person")], None),
          "objectWoke",
        );
        t += 100.0;
      }
      assert_eq!(woke, 0, "two reach-outs {rest_s} s apart woke the sitter");
    }
  }

  // one trip out of the spot and back as a sparse detector sees it: out in steps the engine can follow, then hidden
  fn out_and_back(
    world: &mut CameraWorld,
    t: &mut f64,
    rng: &mut Lcg,
    rest: [f32; 4],
    reach_out: f32,
  ) -> Vec<WorldUpdate> {
    let mut ups = Vec::new();
    for k in 0..=4 {
      let dx = reach_out * k as f32 / 4.0;
      let b = [rest[0] + dx, rest[1], rest[2], rest[3]];
      ups.push(world.ingest(*t, &[jittered(b, rng, "person")], None));
      *t += 100.0;
    }
    for _ in 0..20 {
      ups.push(world.ingest(*t, &[], None));
      *t += 100.0;
    }
    ups
  }

  struct BackAndForth {
    woke_in_cycle: Option<usize>,
    settled_after: usize,
    woke: Vec<TrackSnapshot>,
  }

  fn back_and_forth(
    world: &mut CameraWorld,
    mut t: f64,
    rng: &mut Lcg,
    rest: [f32; 4],
  ) -> BackAndForth {
    let sitter: Vec<u32> = world.tracks.keys().copied().collect();
    let mut out = BackAndForth {
      woke_in_cycle: None,
      settled_after: 0,
      woke: Vec::new(),
    };
    for cycle in 0..48 {
      for up in out_and_back(world, &mut t, rng, rest, 0.12) {
        for e in &up.events {
          match e {
            SemanticEvent::ObjectWoke(s) => {
              out.woke.push(s.clone());
              out.woke_in_cycle.get_or_insert(cycle);
            }
            SemanticEvent::ObjectSettled(s)
              if out.woke_in_cycle.is_some() && sitter.contains(&s.track_id) =>
            {
              out.settled_after += 1
            }
            _ => {}
          }
        }
      }
    }
    out
  }

  #[test]
  fn a_settled_sitter_who_starts_moving_back_and_forth_wakes_and_stays_active() {
    let rest = [0.30, 0.40, 0.07, 0.20];
    let mut rng = Lcg(18);
    let mut world = CameraWorld::new(WorldConfig::default());
    world.set_min_confidence(0.7);
    let t = settle_still(&mut world, &[rest], &["person"], &mut rng);
    let sitter: Vec<u32> = world.tracks.keys().copied().collect();
    let run = back_and_forth(&mut world, t, &mut rng, rest);
    assert_eq!(
      run.woke_in_cycle,
      Some(1),
      "the second trip out wakes the sitter"
    );
    assert_eq!(
      run.woke.iter().map(|s| s.track_id).collect::<Vec<_>>(),
      sitter,
      "the sitter's own identity wakes, once"
    );
    assert_eq!(
      run.settled_after, 0,
      "the sitter settled again while moving back and forth"
    );
    let woke = &run.woke[0];
    assert_eq!(woke.wake_cause, Some(WakeCause::Oscillation));
    let distance = woke.anchor_distance.expect("anchor distance on a wake");
    assert!(distance > 0.5, "woke inside the spot: {distance}");
    assert_eq!(woke.far_from_anchor, Some(distance > FAR_FROM_ANCHOR));
  }

  #[test]
  fn a_fade_settled_sitter_moving_back_and_forth_wakes() {
    let rest = [0.30, 0.40, 0.07, 0.20];
    let mut rng = Lcg(19);
    let mut world = CameraWorld::new(WorldConfig::default());
    let mut t = 0.0;
    for _ in 0..20 {
      world.ingest(t, &[jittered(rest, &mut rng, "person")], None);
      t += 100.0;
    }
    let mut settled = 0;
    for _ in 0..60 {
      settled += count(&world.ingest(t, &[], None), "objectSettled");
      t += 100.0;
    }
    assert_eq!(settled, 1, "setup: fade settle");
    let run = back_and_forth(&mut world, t, &mut rng, rest);
    assert_eq!(
      run.woke_in_cycle,
      Some(1),
      "the second trip out wakes the sitter"
    );
    assert_eq!(run.woke[0].wake_cause, Some(WakeCause::Oscillation));
  }

  #[test]
  fn a_sitter_who_woke_and_sat_down_again_counts_afresh() {
    let rest = [0.30, 0.40, 0.07, 0.20];
    let mut rng = Lcg(18);
    let mut world = CameraWorld::new(WorldConfig::default());
    world.set_min_confidence(0.7);
    let mut t = settle_still(&mut world, &[rest], &["person"], &mut rng);
    let mut woke = 0;
    for _ in 0..2 {
      for up in out_and_back(&mut world, &mut t, &mut rng, rest, 0.12) {
        woke += count(&up, "objectWoke");
      }
    }
    assert_eq!(woke, 1, "setup: the second trip out woke the sitter");
    let mut settled = false;
    for _ in 0..1_000 {
      settled = count(
        &world.ingest(t, &[jittered(rest, &mut rng, "person")], None),
        "objectSettled",
      ) > 0;
      t += 100.0;
      if settled {
        break;
      }
    }
    assert!(settled, "setup: the sitter settled again");
    // straight into one trip out, before the rest at home could clear anything
    let mut after = 0;
    for up in out_and_back(&mut world, &mut t, &mut rng, rest, 0.12) {
      after += count(&up, "objectWoke");
    }
    for _ in 0..50 {
      after += count(
        &world.ingest(t, &[jittered(rest, &mut rng, "person")], None),
        "objectWoke",
      );
      t += 100.0;
    }
    assert_eq!(
      after, 0,
      "one trip out after sitting down again woke the sitter"
    );
  }

  #[test]
  fn gentle_swaying_in_place_stays_settled() {
    let rest = [0.30, 0.40, 0.07, 0.20];
    let mut rng = Lcg(20);
    let mut world = CameraWorld::new(WorldConfig::default());
    let mut t = settle_still(&mut world, &[rest], &["person"], &mut rng);
    let mut woke = 0;
    for _ in 0..48 {
      for up in out_and_back(&mut world, &mut t, &mut rng, rest, 0.05) {
        woke += count(&up, "objectWoke");
      }
    }
    assert_eq!(woke, 0, "swaying within the body's own footprint woke");
  }

  #[test]
  fn swaying_a_quarter_of_the_body_never_wakes() {
    let rest = [0.30, 0.40, 0.07, 0.20];
    for period_ms in [2_000.0, 4_000.0] {
      let mut rng = Lcg(31);
      let mut world = CameraWorld::new(WorldConfig::default());
      let mut t = settle_still(&mut world, &[rest], &["person"], &mut rng);
      let start = t;
      let mut woke = 0;
      for _ in 0..3_000 {
        let phase = ((t - start) / period_ms * std::f64::consts::TAU).sin() as f32;
        let b = [rest[0] + 0.05 * phase, rest[1], rest[2], rest[3]];
        let seen = if rng.next() < 0.9 {
          vec![jittered(b, &mut rng, "person")]
        } else {
          Vec::new()
        };
        woke += count(&world.ingest(t, &seen, None), "objectWoke");
        t += 100.0;
      }
      assert_eq!(
        woke, 0,
        "a sway of a quarter body height woke at a {period_ms} ms period"
      );
    }
  }

  #[test]
  fn a_swaying_false_person_at_the_threshold_never_oscillates() {
    // a web in front of the lens read as a person just at the user threshold
    let web = [0.70, 0.10, 0.03, 0.05];
    let mut rng = Lcg(41);
    let mut world = CameraWorld::new(WorldConfig::default());
    world.set_min_confidence(0.7);
    let mut t = 0.0;
    while t <= WorldConfig::default().settle_person_ms + 1_000.0 {
      let mut d = jittered(web, &mut rng, "person");
      d.confidence = 0.72;
      world.ingest(t, &[d], None);
      t += 100.0;
    }
    let start = t;
    let mut oscillated = 0;
    for k in 0..6_000 {
      let gust = (k / 200) % 3 == 0;
      let phase = if gust {
        ((t - start) / 1_500.0 * std::f64::consts::TAU).sin() as f32
      } else {
        0.0
      };
      let mut seen = Vec::new();
      if rng.next() < 0.6 {
        let mut d = jittered(
          [
            web[0] + 0.02 * phase,
            web[1] + 0.006 * phase,
            web[2],
            web[3],
          ],
          &mut rng,
          "person",
        );
        d.confidence = 0.6 + 0.12 * rng.next();
        seen.push(d);
      }
      oscillated += woke_by(&world.ingest(t, &seen, None), WakeCause::Oscillation).len();
      t += 100.0;
    }
    assert_eq!(
      oscillated, 0,
      "a false person at the threshold woke by oscillation"
    );
  }

  #[test]
  fn a_bystander_never_wakes_while_the_camera_pans() {
    // the pan residual never clears a thin bystander's spot; a seated one it clears,
    // and the next pan forgets the departure again
    for (p, clears) in [
      ([0.40, 0.40, 0.04, 0.10], false),
      ([0.40, 0.40, 0.10, 0.12], true),
    ] {
      let mut departed = false;
      // shift, share of the shift the pose report carries, ticks the report lags
      for (shift, ratio, lag) in [
        (0.12f32, 0.5f32, 0usize),
        (0.12, 0.5, 1),
        (0.10, 0.7, 1),
        (0.08, 0.5, 0),
        (0.12, 1.0, 1),
        (0.16, 0.5, 0),
        (0.12, 0.33, 0),
      ] {
        // ticks at each end: 3 s, then a pan every second and every half second
        for hold in [30usize, 10, 5] {
          for seed in 51..=55 {
            let mut rng = Lcg(seed);
            let mut world = CameraWorld::new(WorldConfig::default());
            let mut t = 0.0;
            while t <= WorldConfig::default().settle_person_ms + 1_000.0 {
              world.ingest(t, &[jittered(p, &mut rng, "person")], Some((0.0, 0.0)));
              t += 100.0;
            }
            let mut reported = 0.0f32;
            let mut pending: Vec<f32> = Vec::new();
            let mut woke = 0;
            for cycle in 0..20 {
              let off = if cycle % 2 == 0 { shift } else { 0.0 };
              for k in 0..hold {
                if k == 0 {
                  pending.push(off * ratio);
                }
                if !pending.is_empty() && k >= lag {
                  reported = pending.remove(0);
                }
                let b = [p[0] + off, p[1], p[2], p[3]];
                woke += count(
                  &world.ingest(t, &[jittered(b, &mut rng, "person")], Some((reported, 0.0))),
                  "objectWoke",
                );
                departed |= world.tracks.values().any(|tr| tr.departures > 0);
                t += 100.0;
              }
            }
            assert_eq!(
              woke, 0,
              "a pan of {shift} reported at {ratio} with lag {lag}, held {hold} ticks, woke the bystander {p:?} (seed {seed})"
            );
          }
        }
      }
      assert_eq!(departed, clears, "bystander {p:?}: a departure counted");
    }
  }

  #[test]
  fn a_walk_off_reports_a_travel_wake() {
    let sit = [0.40, 0.40, 0.10, 0.22];
    let mut rng = Lcg(61);
    let mut world = CameraWorld::new(WorldConfig::default());
    let mut t = settle_still(&mut world, &[sit], &["person"], &mut rng);
    let mut woke = Vec::new();
    for i in 1..=20 {
      let b = [sit[0] + 0.02 * i as f32, sit[1], sit[2], sit[3]];
      for e in world
        .ingest(t, &[jittered(b, &mut rng, "person")], None)
        .events
      {
        if let SemanticEvent::ObjectWoke(s) = e {
          woke.push(s);
        }
      }
      t += 200.0;
    }
    assert_eq!(woke.len(), 1, "one wake for one walk-off");
    assert_eq!(woke[0].wake_cause, Some(WakeCause::Travel));
    assert!(woke[0].anchor_distance.is_some_and(|d| d > 0.5));
    assert!(woke[0].far_from_anchor.is_some());
  }

  #[test]
  fn walking_into_the_room_reports_a_depth_wake() {
    let config = WorldConfig::default();
    let mut world = CameraWorld::new(WorldConfig::default());
    let near = [0.84, 0.23, 0.16, 0.76];
    let mut t = 0.0;
    while t <= config.settle_person_ms + 1_000.0 {
      world.ingest(t, &[lerp_box(near, near, 0.0)], None);
      t += 200.0;
    }
    let far = [0.90, 0.38, 0.06, 0.38];
    let mut causes = Vec::new();
    for i in 1..=20 {
      for e in world
        .ingest(t, &[lerp_box(near, far, i as f32 / 20.0)], None)
        .events
      {
        if let SemanticEvent::ObjectWoke(s) = e {
          causes.push(s.wake_cause);
        }
      }
      t += 200.0;
    }
    assert_eq!(causes, vec![Some(WakeCause::Depth)]);
  }

  #[test]
  fn a_handover_is_reported_on_its_tick_only() {
    let sit = [0.40, 0.40, 0.10, 0.22];
    let mut rng = Lcg(71);
    let mut world = CameraWorld::new(WorldConfig::default());
    let mut t = settle_still(&mut world, &[sit], &["person"], &mut rng);
    let id = *world.tracks.keys().next().unwrap();
    let handed = |up: &WorldUpdate| {
      up.tracked
        .iter()
        .find(|s| s.track_id == id)
        .map(|s| (s.handover, s.handover_tier, s.unseen_ms))
    };
    // hidden longer than the engine keeps a lost track, shorter than the still grace
    let last = t - 100.0;
    for _ in 0..40 {
      world.ingest(t, &[], None);
      t += 100.0;
    }
    let up = world.ingest(t, &[jittered(sit, &mut rng, "person")], None);
    let (kind, tier, unseen) = handed(&up).expect("the sitter is picked up again");
    assert_eq!(kind, Some(Handover::Reassociated));
    assert_eq!(tier, Some(MatchTier::Overlap));
    assert!(
      unseen.is_some_and(|u| (u - (t - last)).abs() < 1.0),
      "unseen {unseen:?}"
    );
    t += 100.0;
    let up = world.ingest(t, &[jittered(sit, &mut rng, "person")], None);
    assert_eq!(
      handed(&up),
      Some((None, None, None)),
      "the engine continues, nothing was handed over"
    );
    // asleep past the still grace: the recovery carries the handover on event and snapshot
    t += 100.0;
    while t < world.tracks[&id].last_seen_ms + WorldConfig::default().still_lost_grace_ms + 1_000.0
    {
      world.ingest(t, &[], None);
      t += 100.0;
    }
    assert!(world.tracks[&id].dormant, "setup: asleep");
    let up = world.ingest(t, &[jittered(sit, &mut rng, "person")], None);
    let recovered = up.events.iter().find_map(|e| match e {
      SemanticEvent::ObjectRecovered(s) if s.track_id == id => Some(s.handover),
      _ => None,
    });
    assert_eq!(recovered, Some(Some(Handover::Recovered)));
    assert_eq!(handed(&up).map(|h| h.0), Some(Some(Handover::Recovered)));
  }

  #[test]
  fn a_handover_names_the_tier_it_matched_on() {
    let sit = [0.40, 0.40, 0.10, 0.22];
    for (seen, tier) in [
      ([0.48, 0.40, 0.10, 0.22], MatchTier::Near),
      ([0.42, 0.42, 0.06, 0.08], MatchTier::Contained),
    ] {
      let mut rng = Lcg(72);
      let mut world = CameraWorld::new(WorldConfig::default());
      let mut t = settle_still(&mut world, &[sit], &["person"], &mut rng);
      let id = *world.tracks.keys().next().unwrap();
      for _ in 0..40 {
        world.ingest(t, &[], None);
        t += 100.0;
      }
      let up = world.ingest(t, &[jittered(seen, &mut rng, "person")], None);
      let handed = up
        .tracked
        .iter()
        .find(|s| s.track_id == id)
        .map(|s| (s.handover, s.handover_tier));
      assert_eq!(
        handed,
        Some((Some(Handover::Reassociated), Some(tier))),
        "picked up at {seen:?}"
      );
    }
  }

  #[test]
  fn a_newcomer_confirmed_on_its_pickup_was_never_handed_over() {
    let mut world = CameraWorld::new(WorldConfig::default());
    world.set_min_confidence(0.5);
    world.ingest(0.0, &[box_at(0.40, 0.40, 0.10, 0.22, "person")], None);
    let mut t = 100.0;
    while t < 2_000.0 {
      world.ingest(t, &[], None);
      t += 100.0;
    }
    let up = world.ingest(t, &[box_at(0.495, 0.40, 0.10, 0.22, "person")], None);
    assert_eq!(
      up.created.len(),
      1,
      "setup: the pickup confirms the newcomer"
    );
    let entered = up
      .tracked
      .iter()
      .find(|s| s.track_id == up.created[0])
      .map(|s| (s.handover, s.handover_tier, s.unseen_ms));
    assert_eq!(entered, Some((None, None, None)));
  }

  #[test]
  fn a_newcomer_confirming_where_a_body_vanished_before_it_is_that_body() {
    let mut world = CameraWorld::new(WorldConfig::default());
    let mut t = 0.0;
    while t <= 2_000.0 {
      world.ingest(t, &[person(0.2)], None);
      t += 200.0;
    }
    let id = *world.tracks.keys().next().unwrap();
    while t < 4_000.0 {
      world.ingest(t, &[], None);
      t += 200.0;
    }
    // found again a little off, too far for a pickup, then back on its spot
    let mut created = Vec::new();
    let mut handed = None;
    for x in [0.32, 0.29, 0.26, 0.23, 0.2] {
      let up = world.ingest(t, &[person(x)], None);
      created.extend(up.created);
      handed = handed.or(
        up.tracked
          .iter()
          .find(|s| s.track_id == id)
          .and_then(|s| s.handover),
      );
      t += 200.0;
    }
    assert!(created.is_empty(), "the body entered again as {created:?}");
    assert_eq!(handed, Some(Handover::Reassociated));
    assert_eq!(world.tracks.len(), 1);
  }

  #[test]
  fn a_car_driving_onto_a_parked_cars_spot_is_not_that_car() {
    let parked = [0.50, 0.40, 0.15, 0.10];
    let mut rng = Lcg(73);
    let mut world = CameraWorld::new(WorldConfig::default());
    let mut t = settle_still(&mut world, &[parked], &["vehicle"], &mut rng);
    let id = *world.tracks.keys().next().unwrap();
    assert_eq!(
      world.tracks[&id].state,
      TrackState::Stationary,
      "setup: parked"
    );
    for _ in 0..100 {
      world.ingest(t, &[], None);
      t += 100.0;
    }
    // born a car length off the spot, on it when it confirms, then on its way
    let mut created = Vec::new();
    let mut handed = None;
    let mut x: f32 = 0.30;
    while x < 0.8 {
      let up = world.ingest(t, &[box_at(x, 0.40, 0.15, 0.10, "vehicle")], None);
      created.extend(up.created);
      handed = handed.or(
        up.tracked
          .iter()
          .find(|s| s.track_id == id)
          .and_then(|s| s.handover),
      );
      x += 0.04;
      t += 100.0;
    }
    assert_eq!(created.len(), 1, "the passing car never entered");
    assert_eq!(handed, None, "the parked car was handed to the passing one");
    assert!(world.tracks.contains_key(&id), "the parked car is gone");
  }

  #[test]
  fn a_newcomer_confirming_where_a_flicker_died_was_never_handed_over() {
    let mut world = CameraWorld::new(WorldConfig::default());
    world.ingest(0.0, &[person(0.2)], None);
    let mut t = 200.0;
    while t < 2_000.0 {
      world.ingest(t, &[], None);
      t += 200.0;
    }
    let mut entered = Vec::new();
    for x in [0.32, 0.29, 0.26, 0.23] {
      let up = world.ingest(t, &[person(x)], None);
      entered.extend(
        up.tracked
          .iter()
          .filter(|s| up.created.contains(&s.track_id))
          .map(|s| s.handover),
      );
      t += 200.0;
    }
    assert_eq!(entered, [None]);
  }

  #[test]
  fn a_newcomer_seen_beside_a_body_never_takes_it_over() {
    let mut world = CameraWorld::new(WorldConfig::default());
    world.set_min_confidence(0.7);
    let seen = |x: f32, confidence: f32| {
      let mut d = person(x);
      d.confidence = confidence;
      d
    };
    let mut t = 0.0;
    while t <= 2_000.0 {
      world.ingest(t, &[person(0.2)], None);
      t += 200.0;
    }
    let id = *world.tracks.keys().next().unwrap();
    // a second person comes in while the first is still there, too faint to
    // confirm, and walks onto the first one's spot once that one is gone
    world.ingest(t, &[person(0.2), seen(0.45, 0.75)], None);
    t += 200.0;
    world.ingest(t, &[person(0.2), seen(0.43, 0.5)], None);
    t += 100.0;
    // longer than the engine keeps a lost track, shorter than the depart grace
    for i in 1..=40 {
      world.ingest(t, &[seen(0.43 - 0.23 * i as f32 / 40.0, 0.5)], None);
      t += 100.0;
    }
    assert_eq!(
      world.tracks[&id].state,
      TrackState::Active,
      "setup: the first one is still held"
    );
    let up = world.ingest(t, &[seen(0.2, 0.9)], None);
    assert_eq!(up.created.len(), 1, "the second person never entered");
    assert_ne!(up.created[0], id);
    let first = up.tracked.iter().find(|s| s.track_id == id);
    assert!(first.is_none(), "the first one was handed to the second");
  }
}
