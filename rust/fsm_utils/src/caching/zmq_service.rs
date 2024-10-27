// Copyright 2024 Nathan Hoos
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#![cfg(feature = "python_bindings")]
use crate::caching::{MODULE_STATE, CachedFSM};
use std::sync::{
    mpsc::{self, Receiver, Sender},
    Mutex,
};
use std::thread;
use zmq::{Context, SocketType};
use anyhow::{anyhow, bail, Result};
use std::fs;

const UNIX_ADDRESS: &str = "ipc:///tmp/faster_outlines_cache_reciever.ipc";
const FALLBACK_ADRESS: &str = "tcp://127.0.0.1:5555";
const UNIX_FILE_PATH: &str = "/tmp/faster_outlines_cache_reciever.ipc";

struct ZMQReciever {
    stop_sender: Option<Sender<()>>,
}

impl ZMQReciever {
    pub fn new() -> Self {
        ZMQReciever {
            stop_sender: None,
        }
    }

    pub fn start_cache_service(&mut self, context: &Context) -> Result<Sender<()>> {
        let (stop_tx, stop_rx): (Sender<()>, Receiver<()>) = mpsc::channel();

        let zmq_context = context.clone();

        let connection_address = if cfg!(unix) {
            UNIX_ADDRESS
        } else {
            FALLBACK_ADRESS
        };

        thread::spawn(move || {
            let socket = zmq_context.socket(SocketType::REP).unwrap();
            socket.bind(connection_address).unwrap();
            println!("Cache service started at: {}", connection_address);

            loop {
                let mut poll_items = [socket.as_poll_item(zmq::POLLIN)];

                let _poll_result = zmq::poll(&mut poll_items, 10).unwrap(); // 0.01s

                if poll_items[0].is_readable() {
                    match socket.recv_msg(0) {
                        Ok(msg) => {
                            let received: Result<CachedFSM, _> = serde_json::from_slice(&msg);
                            match received {
                                Ok(fsm) => {
                                    MODULE_STATE
                                        .fsm_cache
                                        .lock()
                                        .unwrap()
                                        .put(fsm.hash.clone(), fsm.into());
                                    socket.send("Inserted", 0).unwrap();
                                }
                                Err(e) => {
                                    eprintln!("Failed to deserialize FSM: {:?}", e);
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("Failed to receive message: {:?}", e);
                        }
                    }
                }

                if stop_rx.try_recv().is_ok() {
                    println!("Stopping cache service...");
                    break;
                }
            }

            let _ = socket.unbind(connection_address);
            println!("Cache service stopped.");

            if cfg!(unix) {
                if let Err(e) = fs::remove_file(UNIX_FILE_PATH) {
                    eprintln!("Failed to remove IPC file: {:?}", e);
                } else {
                    println!("IPC file cleaned up: {}", UNIX_FILE_PATH);
                }
            }let _ = socket.unbind(connection_address);
            println!("Cache service stopped.");

            if cfg!(unix) {
                if let Err(e) = fs::remove_file(UNIX_FILE_PATH) {
                    eprintln!("Failed to remove IPC file: {:?}", e);
                } else {
                    println!("IPC file cleaned up: {}", UNIX_FILE_PATH);
                }
            }
        });

        self.stop_sender = Some(stop_tx.clone()); 
        Ok(stop_tx)
    }
}

static CACHE_STOP_TX: Mutex<Option<Sender<()>>> = Mutex::new(None);

pub fn start_zmq_thread() -> Result<()> {
    let mut cache_service = ZMQReciever::new();
    let context = Context::new();

    let stop_sender = cache_service.start_cache_service(&context)?;
    let mut global_reciever_stop_tx = CACHE_STOP_TX.lock().unwrap();
    *global_reciever_stop_tx = Some(stop_sender);

    println!("faster_outlines cache service started.");
    Ok(())
}

pub fn stop_zmq_thread() -> Result<()> {
    let mut global_reciever_stop_tx = CACHE_STOP_TX.lock().unwrap();

    if let Some(stop_sender) = global_reciever_stop_tx.take() {
        stop_sender.send(()).map_err(|e| anyhow!("Failed to send stop signal: {:?}", e))?;
        println!("faster_outlines cache service stopped.");
        Ok(())
    } else {
        bail!("Cache service is not running")
    }
}

pub fn check_zmq_service_running() -> Result<(bool, String)> {
    let is_running = CACHE_STOP_TX.lock().unwrap().is_some();
    let address = if cfg!(unix) {
        UNIX_ADDRESS
    } else {
        FALLBACK_ADRESS
    };

    Ok((is_running, address.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustc_hash::FxHashMap;
    use zmq::SocketType;

    fn create_sample_fsm() -> CachedFSM {
        let mut state_map = FxHashMap::default();
        state_map.insert(1, 2);
        CachedFSM {
            states_to_token_maps: vec![state_map],
            first_state: 1,
            finals: vec![2],
            hash: 12345,
        }
    }
    /// Tests that the service correctly starts, stops,
    /// and inserts fsm's correctly.
    #[test]
    fn test_send_receive_fsm() {
        assert!(start_zmq_thread().is_ok());

        let context = zmq::Context::new();
        let socket = context.socket(SocketType::REQ).unwrap();
        let address = if cfg!(unix) { UNIX_ADDRESS } else { FALLBACK_ADRESS };
        socket.connect(address).unwrap();

        let fsm = create_sample_fsm();
        let serialized_fsm = serde_json::to_vec(&fsm).unwrap();

        socket.send(serialized_fsm, 0).unwrap();

        let response = socket.recv_string(0).unwrap().unwrap();
        assert_eq!(response, "Inserted");
        println!("fsm inserted");

        let cache = MODULE_STATE.fsm_cache.lock().unwrap();
        assert!(cache.contains(&fsm.hash), "FSM should be cached");
        println!("FSM found in cache");
        assert!(stop_zmq_thread().is_ok());
    }
}