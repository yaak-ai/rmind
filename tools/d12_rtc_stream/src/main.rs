mod controller {
    include!(env!("RTC_CONTROLLER_SOURCE"));
}

use controller::{RtcConfig, RtcController, RtcError};
use serde::Deserialize;
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};

#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum Event {
    Request { now_ns: u64, observation_ns: u64, generation: u64 },
    Complete { now_ns: u64, generation: u64, request_id: u64, actions: Vec<Vec<f32>> },
    Action { now_ns: u64, generation: u64 },
    Reset { generation: u64 },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    let config: RtcConfig = serde_json::from_str(&lines.next().ok_or("missing config")??)?;
    let mut controller = RtcController::new(config)?;
    let mut stdout = io::stdout().lock();
    for line in lines {
        let event: Event = serde_json::from_str(&line?)?;
        let result: Result<Value, RtcError> = match event {
            Event::Request { now_ns, observation_ns, generation } => controller
                .request(now_ns, observation_ns, generation).map(|value| json!(value)),
            Event::Complete { now_ns, generation, request_id, actions } => controller
                .complete(now_ns, generation, request_id, actions).map(|value| json!(value)),
            Event::Action { now_ns, generation } => controller
                .action(now_ns, generation).map(|value| json!(value)),
            Event::Reset { generation } => { controller.reset(generation); Ok(Value::Null) }
        };
        let output = match result {
            Ok(value) => json!({"ok": value, "forecast": controller.forecast()}),
            Err(error) => json!({"error": error, "forecast": controller.forecast()})
        };
        writeln!(stdout, "{}", output)?;
        stdout.flush()?;
    }
    Ok(())
}
