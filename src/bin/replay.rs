use std::process::ExitCode;

use cameraui_rust_postprocessor::replay::{read_jsonl_items, run_world_items};

fn main() -> ExitCode {
  let mut path = None;
  let mut keep_events = false;

  for arg in std::env::args().skip(1) {
    match arg.as_str() {
      "--events" => keep_events = true,
      _ => path = Some(arg),
    }
  }
  let Some(path) = path else {
    eprintln!("usage: replay <ticks.jsonl> [--events]");
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

  let summary = run_world_items(&items, keep_events);
  println!(
    "{}",
    serde_json::to_string_pretty(&summary).expect("summary serializes")
  );
  ExitCode::SUCCESS
}
