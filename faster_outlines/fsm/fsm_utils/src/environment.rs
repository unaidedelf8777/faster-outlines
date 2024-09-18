use std::env;
use once_cell::sync::Lazy;

pub static FSM_CACHE_SIZE: Lazy<usize> = Lazy::new(|| {
    env::var("FASTER_OUTLINES_CACHE_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100)
});

pub static NUM_THREADS: Lazy<String> = Lazy::new(|| {
    match env::var("FASTER_OUTLINES_NUM_THREADS") {
        Ok(threads) => {
            println!("Faster Outlines num threads set to: {:?}", threads);
            threads
        },
        Err(_) => "1".to_string(),
    }
});

pub static DISABLE_CACHE: Lazy<bool> = Lazy::new(|| {
    match env::var("FASTER_OUTLINES_DISABLE_CACHE") {
        Ok(val) => {
            let val_lower = val.to_lowercase();
            let is_disabled = val_lower == "1" || val_lower == "true" || val_lower == "yes";
            if is_disabled {
                println!("Cache is disabled via DISABLE_CACHE environment variable.");
            }
            is_disabled
        },
        Err(_) => false, 
    }
});