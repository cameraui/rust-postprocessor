use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::iou::box_iou;
use crate::line_crossing::{DetectionLineInput, LineDirectionFilter};
use crate::semantic::{SemanticEvent, TrackSnapshot, TrackState};
use crate::zone_filter::{ZoneFilterMode, ZoneInput, ZoneMatchType};

pub use crate::types::{CameraMotion, Detection};

const CHURN_WINDOW_MS: f64 = 120_000.0;
const CHURN_IOU: f32 = 0.3;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReplayTick {
  pub t_ms: f64,
  pub detections: Vec<Detection>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub camera_motion: Option<CameraMotion>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReplayZone {
  #[serde(default)]
  pub labels: Vec<String>,
  pub filter: String,
  pub match_type: String,
  #[serde(default)]
  pub is_privacy_mask: bool,
  pub points: Vec<[f64; 2]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReplayLine {
  pub name: String,
  pub direction: String,
  #[serde(default)]
  pub labels: Vec<String>,
  pub points: Vec<[f64; 2]>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ReplayConfig {
  pub zones: Option<Vec<ReplayZone>>,
  pub min_confidence: Option<f32>,
  pub lines: Option<Vec<ReplayLine>>,
  pub aspect_ratio: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ReplayItem {
  Config { config: ReplayConfig },
  Tick(ReplayTick),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LatencyStats {
  pub mean_us: f64,
  pub p50_us: u64,
  pub p95_us: u64,
  pub max_us: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReplaySummary {
  pub ticks: u64,
  pub detections_in: u64,
  pub tracks_created: u64,
  pub tracks_removed: u64,
  pub identity_churn: u64,
  pub events_by_type: BTreeMap<String, u64>,
  pub tick_latency: LatencyStats,
  #[serde(default, skip_serializing_if = "Vec::is_empty")]
  pub events: Vec<TimedEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TimedEvent {
  pub t_ms: f64,
  #[serde(flatten)]
  pub event: SemanticEvent,
}

pub struct EngineTick {
  pub tracked: Vec<(TrackSnapshot, bool)>,
  pub created: Vec<u32>,
  pub removed: Vec<u32>,
  pub extra_events: Vec<SemanticEvent>,
}

pub fn run_world(ticks: &[ReplayTick], keep_events: bool) -> ReplaySummary {
  let items: Vec<ReplayItem> = ticks.iter().cloned().map(ReplayItem::Tick).collect();
  run_world_items(&items, keep_events)
}

pub fn run_world_items(items: &[ReplayItem], keep_events: bool) -> ReplaySummary {
  let mut world = crate::world::CameraWorld::new(crate::world::WorldConfig::default());
  let mut aspect: f32 = 16.0 / 9.0;
  run_steps(
    items,
    move |item| {
      let tick = match item {
        ReplayItem::Config { config } => {
          apply_config(&mut world, config, &mut aspect);
          return None;
        }
        ReplayItem::Tick(tick) => tick,
      };
      let update = world.ingest(
        tick.t_ms,
        &tick.detections,
        tick.camera_motion.map(|m| (m.x as f32, m.y as f32)),
      );
      Some(EngineTick {
        tracked: update.tracked.into_iter().map(|s| (s, false)).collect(),
        created: update.created,
        removed: update.removed,
        extra_events: update
          .events
          .into_iter()
          .filter(|e| {
            matches!(
              e.kind(),
              "objectSettled" | "objectWoke" | "bestShotUpdated" | "objectLost" | "objectRecovered"
            )
          })
          .collect(),
      })
    },
    keep_events,
  )
}

fn apply_config(world: &mut crate::world::CameraWorld, config: &ReplayConfig, aspect: &mut f32) {
  if let Some(a) = config.aspect_ratio {
    *aspect = a;
  }
  if let Some(zones) = &config.zones {
    world.set_zones(zones.iter().filter_map(to_zone_input).collect());
  }
  if let Some(min) = config.min_confidence {
    world.set_min_confidence(min);
  }
  if let Some(lines) = &config.lines {
    world.set_lines(lines.iter().filter_map(to_line_input).collect(), *aspect);
  }
}

fn to_zone_input(z: &ReplayZone) -> Option<ZoneInput> {
  Some(ZoneInput {
    labels: z.labels.clone(),
    filter: match z.filter.as_str() {
      "include" => ZoneFilterMode::Include,
      "exclude" => ZoneFilterMode::Exclude,
      _ => return None,
    },
    match_type: match z.match_type.as_str() {
      "intersect" => ZoneMatchType::Intersect,
      "contain" => ZoneMatchType::Contain,
      _ => return None,
    },
    is_privacy_mask: z.is_privacy_mask,
    points: z.points.clone(),
  })
}

fn to_line_input(l: &ReplayLine) -> Option<DetectionLineInput> {
  if l.points.len() != 2 {
    return None;
  }
  Some(DetectionLineInput {
    name: l.name.clone(),
    direction: match l.direction.as_str() {
      "both" => LineDirectionFilter::Both,
      "a-to-b" => LineDirectionFilter::AToB,
      "b-to-a" => LineDirectionFilter::BToA,
      _ => return None,
    },
    labels: l.labels.clone(),
    points: [l.points[0], l.points[1]],
  })
}

pub(crate) fn run_steps(
  items: &[ReplayItem],
  mut step: impl FnMut(&ReplayItem) -> Option<EngineTick>,
  keep_events: bool,
) -> ReplaySummary {
  let mut last_seen: HashMap<u32, TrackSnapshot> = HashMap::new();
  let mut lost_state: HashMap<u32, bool> = HashMap::new();
  let mut departures: Vec<(f64, TrackSnapshot)> = Vec::new();

  let mut summary = ReplaySummary {
    ticks: 0,
    detections_in: 0,
    tracks_created: 0,
    tracks_removed: 0,
    identity_churn: 0,
    events_by_type: BTreeMap::new(),
    tick_latency: LatencyStats {
      mean_us: 0.0,
      p50_us: 0,
      p95_us: 0,
      max_us: 0,
    },
    events: Vec::new(),
  };
  let mut latencies_us: Vec<u64> = Vec::new();
  let emit = |summary: &mut ReplaySummary, t_ms: f64, event: SemanticEvent| {
    *summary
      .events_by_type
      .entry(event.kind().to_string())
      .or_insert(0) += 1;
    if keep_events {
      summary.events.push(TimedEvent { t_ms, event });
    }
  };

  for item in items {
    let started = Instant::now();
    let Some(result) = step(item) else { continue };
    let ReplayItem::Tick(tick) = item else {
      continue;
    };
    summary.ticks += 1;
    summary.detections_in += tick.detections.len() as u64;
    latencies_us.push(started.elapsed().as_micros() as u64);

    for event in result.extra_events {
      emit(&mut summary, tick.t_ms, event);
    }

    for (snapshot, is_lost) in &result.tracked {
      let was_lost = lost_state.insert(snapshot.track_id, *is_lost);
      match (was_lost, *is_lost) {
        (Some(false), true) => emit(
          &mut summary,
          tick.t_ms,
          SemanticEvent::ObjectLost(snapshot.clone()),
        ),
        (Some(true), false) => emit(
          &mut summary,
          tick.t_ms,
          SemanticEvent::ObjectRecovered(snapshot.clone()),
        ),
        _ => {}
      }
      last_seen.insert(snapshot.track_id, snapshot.clone());
    }

    for id in &result.created {
      summary.tracks_created += 1;
      let Some(snapshot) = last_seen.get(id) else {
        continue;
      };
      departures.retain(|(t, _)| tick.t_ms - t <= CHURN_WINDOW_MS);
      let churned = departures.iter().any(|(_, gone)| {
        gone.label == snapshot.label
          && box_iou(
            &[gone.x, gone.y, gone.width, gone.height],
            &[snapshot.x, snapshot.y, snapshot.width, snapshot.height],
          ) >= CHURN_IOU
      });
      if churned {
        summary.identity_churn += 1;
      }
      emit(
        &mut summary,
        tick.t_ms,
        SemanticEvent::ObjectEntered(snapshot.clone()),
      );
    }

    for id in &result.removed {
      summary.tracks_removed += 1;
      lost_state.remove(id);
      let Some(mut snapshot) = last_seen.remove(id) else {
        continue;
      };
      snapshot.state = TrackState::Departed;
      departures.push((tick.t_ms, snapshot.clone()));
      emit(
        &mut summary,
        tick.t_ms,
        SemanticEvent::ObjectDeparted(snapshot),
      );
    }
  }

  summary.tick_latency = latency_stats(&mut latencies_us);
  summary
}

pub fn read_jsonl(input: &str) -> Result<Vec<ReplayTick>, serde_json::Error> {
  input
    .lines()
    .map(str::trim)
    .filter(|line| !line.is_empty())
    .map(serde_json::from_str)
    .collect()
}

pub fn read_jsonl_items(input: &str) -> Result<Vec<ReplayItem>, serde_json::Error> {
  input
    .lines()
    .map(str::trim)
    .filter(|line| !line.is_empty())
    .map(serde_json::from_str)
    .collect()
}

fn latency_stats(latencies_us: &mut [u64]) -> LatencyStats {
  if latencies_us.is_empty() {
    return LatencyStats {
      mean_us: 0.0,
      p50_us: 0,
      p95_us: 0,
      max_us: 0,
    };
  }
  latencies_us.sort_unstable();
  let sum: u64 = latencies_us.iter().sum();
  let at = |q: f64| latencies_us[((latencies_us.len() - 1) as f64 * q) as usize];
  LatencyStats {
    mean_us: sum as f64 / latencies_us.len() as f64,
    p50_us: at(0.5),
    p95_us: at(0.95),
    max_us: latencies_us[latencies_us.len() - 1],
  }
}
