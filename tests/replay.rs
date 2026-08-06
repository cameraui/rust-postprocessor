use cameraui_rust_postprocessor::replay::{
  run_world, Detection, LatencyStats, ReplaySummary, ReplayTick,
};

const TICK_MS: f64 = 200.0;

fn canon(mut summary: ReplaySummary) -> String {
  summary.tick_latency = LatencyStats {
    mean_us: 0.0,
    p50_us: 0,
    p95_us: 0,
    max_us: 0,
  };
  serde_json::to_string(&summary).unwrap()
}

fn det(x: f32, y: f32, w: f32, h: f32, confidence: f32, label: &str) -> Detection {
  Detection {
    x,
    y,
    width: w,
    height: h,
    confidence,
    label: label.to_string(),
  }
}

fn tick(index: usize, detections: Vec<Detection>) -> ReplayTick {
  ReplayTick {
    t_ms: index as f64 * TICK_MS,
    detections,
    camera_motion: None,
  }
}

fn parked_car_flicker() -> Vec<ReplayTick> {
  let mut ticks = Vec::new();
  for session in 0..6 {
    let base = session * 250;
    for i in 0..12 {
      let confidence = if i % 2 == 0 { 0.55 } else { 0.72 };
      ticks.push(tick(
        base + i,
        vec![det(0.4, 0.1, 0.25, 0.12, confidence, "vehicle")],
      ));
    }
    for i in 12..250 {
      ticks.push(tick(base + i, Vec::new()));
    }
  }
  ticks
}

fn gate_visit() -> Vec<ReplayTick> {
  let mut ticks = Vec::new();
  for i in 0..30 {
    let x = 0.05 + i as f32 * 0.01;
    ticks.push(tick(i, vec![det(x, 0.4, 0.2, 0.15, 0.85, "vehicle")]));
  }
  for i in 30..180 {
    ticks.push(tick(i, vec![det(0.35, 0.4, 0.2, 0.15, 0.85, "vehicle")]));
  }
  for i in 180..220 {
    let x = 0.35 + (i - 180) as f32 * 0.012;
    ticks.push(tick(i, vec![det(x, 0.4, 0.2, 0.15, 0.85, "vehicle")]));
  }
  ticks
}

fn passing_car() -> Vec<ReplayTick> {
  let mut ticks = Vec::new();
  for i in 0..120 {
    let mut detections = vec![det(0.45, 0.5, 0.22, 0.16, 0.8, "vehicle")];
    if (20..80).contains(&i) {
      let x = -0.1 + (i - 20) as f32 * 0.015;
      detections.push(det(x, 0.45, 0.25, 0.18, 0.85, "vehicle"));
    }
    ticks.push(tick(i, detections));
  }
  ticks
}

#[test]
fn deterministic_across_runs() {
  for scenario in [parked_car_flicker(), gate_visit(), passing_car()] {
    let a = run_world(&scenario, true);
    let b = run_world(&scenario, true);
    assert_eq!(canon(a), canon(b));
  }
}

#[test]
fn world_baselines() {
  let flicker = run_world(&parked_car_flicker(), true);
  println!(
    "world parked_car_flicker: {}",
    serde_json::to_string_pretty(&flicker).unwrap()
  );
  assert_eq!(
    flicker.tracks_created, 1,
    "one car, one identity across all decode gaps"
  );
  assert_eq!(flicker.identity_churn, 0);

  let gate = run_world(&gate_visit(), true);
  println!(
    "world gate_visit: {}",
    serde_json::to_string_pretty(&gate).unwrap()
  );
  assert_eq!(gate.tracks_created, 1);
  assert_eq!(gate.identity_churn, 0);
  assert_eq!(
    gate.events_by_type.get("objectSettled"),
    Some(&1),
    "waiting at the gate settles"
  );
  assert_eq!(
    gate.events_by_type.get("objectWoke"),
    Some(&1),
    "driving on wakes"
  );

  let passing = run_world(&passing_car(), true);
  println!(
    "world passing_car: {}",
    serde_json::to_string_pretty(&passing).unwrap()
  );
  assert_eq!(passing.tracks_created, 2);
  assert_eq!(passing.identity_churn, 0);

  let a = run_world(&parked_car_flicker(), true);
  let b = run_world(&parked_car_flicker(), true);
  assert_eq!(canon(a), canon(b));
}

#[test]
fn jsonl_roundtrip() {
  let scenario = gate_visit();
  let jsonl: String = scenario
    .iter()
    .map(|t| serde_json::to_string(t).unwrap() + "\n")
    .collect();
  let parsed = cameraui_rust_postprocessor::replay::read_jsonl(&jsonl).unwrap();
  let a = run_world(&scenario, true);
  let b = run_world(&parsed, true);
  assert_eq!(canon(a), canon(b));
}
