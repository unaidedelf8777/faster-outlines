use once_cell::sync::Lazy;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::env; 

mod lazy_index;
mod tokenizer_index;
mod types;
mod caching;
mod environment;

use tokenizer_index::create_fsm_index_end_to_end_py;
use environment::NUM_THREADS;
use crate::lazy_index::{LazyFSMIndex, Write, Generate};
use crate::caching::MODULE_STATE;

#[pymodule]
fn fsm_utils(m: &Bound<'_, PyModule>) -> PyResult<()> {
    
    env::set_var("RAYON_NUM_THREADS", &*NUM_THREADS);

    Lazy::force(&MODULE_STATE);

    m.add_function(wrap_pyfunction!(create_fsm_index_end_to_end_py, m)?)?;
    m.add_class::<LazyFSMIndex>()?;
    m.add_class::<Write>()?;
    m.add_class::<Generate>()?;
    Ok(())
}