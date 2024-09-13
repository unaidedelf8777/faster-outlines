use once_cell::sync::Lazy;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::env; 
use std::thread;
use std::cmp::{min, max};

mod lazy_index;
mod tokenizer_index;
mod types;
mod caching;

use tokenizer_index::create_fsm_index_end_to_end_py;
use crate::lazy_index::{LazyFSMIndex, Write, Generate};
use crate::caching::MODULE_STATE;

#[pymodule]
fn fsm_utils(m: &Bound<'_, PyModule>) -> PyResult<()> {

    let num_threads = if let Ok(threads) = env::var("FASTER_OUTLINES_NUM_THREADS") {
        println!("Faster Outlines num threads set to: {:?}", threads);
        threads
    } else {
        // Get the number of available threads on the machine
        let available_threads = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        // Scale the thread count to 1/4 of available threads, capped at 16
        let scaled_threads = match available_threads {
            1..=4 => 1,
            5..=8 => 2,
            _ => min(max(available_threads / 4, 2), 16),
        };

        scaled_threads.to_string()
    };

    env::set_var("RAYON_NUM_THREADS", num_threads);

    // Ensure MODULE_STATE / cache is initialized
    Lazy::force(&MODULE_STATE);

    m.add_function(wrap_pyfunction!(create_fsm_index_end_to_end_py, m)?)?;
    m.add_class::<LazyFSMIndex>()?;
    m.add_class::<Write>()?;
    m.add_class::<Generate>()?;
    Ok(())
}