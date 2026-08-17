use std::process::ExitCode;

use cameraui_rust_postprocessor::replay::{read_jsonl_items, run_world_items_with};
use cameraui_rust_postprocessor::world::WorldConfig;

fn main() -> ExitCode {
  let mut path = None;
  let mut keep_events = false;
  let mut config = WorldConfig::default();

  let args: Vec<String> = std::env::args().skip(1).collect();
  let mut i = 0;
  while i < args.len() {
    match args[i].as_str() {
      "--events" => keep_events = true,
      "--world" => {
        let Some(pair) = args.get(i + 1) else {
          eprintln!("--world needs key=value");
          return ExitCode::FAILURE;
        };
        let Some((key, value)) = pair.split_once('=') else {
          eprintln!("--world needs key=value, got {pair}");
          return ExitCode::FAILURE;
        };
        let Ok(value) = value.parse::<f64>() else {
          eprintln!("--world {key}: not a number: {value}");
          return ExitCode::FAILURE;
        };
        match key {
          "gapResetMs" => config.gap_reset_ms = value,
          "departGraceMs" => config.depart_grace_ms = value,
          "stillLostGraceMs" => config.still_lost_grace_ms = value,
          "settleDefaultMs" => config.settle_default_ms = value,
          "settleVehicleMs" => config.settle_vehicle_ms = value,
          "settlePersonMs" => config.settle_person_ms = value,
          "stationarySpeed" => config.stationary_speed = value as f32,
          "reassocIou" => config.reassoc_iou = value as f32,
          "wakeTicks" => config.wake_ticks = value as u32,
          "confirmMs" => config.confirm_ms = value,
          "maxDormant" => config.max_dormant = value as usize,
          _ => {
            eprintln!("--world: unknown key {key}");
            return ExitCode::FAILURE;
          }
        }
        i += 1;
      }
      _ => path = Some(args[i].clone()),
    }
    i += 1;
  }
  let Some(path) = path else {
    eprintln!("usage: replay <ticks.jsonl> [--events] [--world key=value]...");
    return ExitCode::FAILURE;
  };

  let input = match std::fs::read_to_string(&path) {
    Ok(input) => input,
    Err(error) => {
      eprintln!("{path}: {error}");
      return ExitCode::FAILURE;
    }
  };
  let items = match read_jsonl_items(&input) {
    Ok(items) => items,
    Err(error) => {
      eprintln!("{path}: {error}");
      return ExitCode::FAILURE;
    }
  };

  let summary = run_world_items_with(&items, keep_events, config);
  println!(
    "{}",
    serde_json::to_string_pretty(&summary).expect("summary serializes")
  );
  ExitCode::SUCCESS
}
