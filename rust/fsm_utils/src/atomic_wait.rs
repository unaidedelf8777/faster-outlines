// This is basically identical to https://docs.rs/atomic-wait/latest/atomic_wait/
// just modified so that it works with AtomicBool instead of AtomicU32
// This doesnt have docs because I dont know CPP, so this is foreign to me.
// Someone feel free to document it though, would be appreciated.
#![allow(dead_code)]
#[cfg(target_os = "linux")]
pub mod platform {
    use core::sync::atomic::{AtomicBool, Ordering};
    use libc;

    // These need to wait in a loop,
    // because futex's while very performant,
    // can also return spiratically / when the kernel decides.
    // so we have to deal with that by checking before returning.
    #[inline]
    pub fn wait(a: &AtomicBool, expected: bool) {
        while a.load(Ordering::SeqCst) == expected {
            let expected_int = if expected { 1 } else { 0 };
            unsafe {
                libc::syscall(
                    libc::SYS_futex,
                    a as *const _ as *const i32,
                    libc::FUTEX_WAIT | libc::FUTEX_PRIVATE_FLAG,
                    expected_int,
                    core::ptr::null::<libc::timespec>(),
                );
            }
        }
    }
    

    #[inline]
    pub fn wake_one(ptr: *const AtomicBool) {
        unsafe {
            libc::syscall(
                libc::SYS_futex,
                ptr as *const _ as *const i32,
                libc::FUTEX_WAKE | libc::FUTEX_PRIVATE_FLAG,
                1i32,
            );
        }
    }

    #[inline]
    pub fn wake_all(ptr: *const AtomicBool) {
        unsafe {
            libc::syscall(
                libc::SYS_futex,
                ptr as *const _ as *const i32,
                libc::FUTEX_WAKE | libc::FUTEX_PRIVATE_FLAG,
                i32::MAX,
            );
        }
    }
}

#[cfg(target_os = "freebsd")]
pub mod platform {
    use core::sync::atomic::{AtomicBool, Ordering};
    use libc;

    // These need to wait in a loop,
    // because futex's while very performant,
    // can also return spiratically / when the kernel decides.
    // so we have to deal with that by checking before returning.
    #[inline]
    pub fn wait(a: &AtomicBool, expected: bool) {
        while a.load(Ordering::SeqCst) == expected {
            let expected_int = if expected { 1 } else { 0 };
            let ptr: *const AtomicBool = a;
            unsafe {
                libc::_umtx_op(
                    ptr as *mut libc::c_void,
                    libc::UMTX_OP_WAIT_UINT_PRIVATE,
                    expected_int as libc::c_ulong,
                    core::ptr::null_mut(),
                    core::ptr::null_mut(),
                );
            }
        }
    }


    #[inline]
    pub fn wake_one(ptr: *const AtomicBool) {
        unsafe {
            libc::_umtx_op(
                ptr as *mut libc::c_void,
                libc::UMTX_OP_WAKE_PRIVATE,
                1 as libc::c_ulong,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
            );
        };
    }

    #[inline]
    pub fn wake_all(ptr: *const AtomicBool) {
        unsafe {
            libc::_umtx_op(
                ptr as *mut libc::c_void,
                libc::UMTX_OP_WAKE_PRIVATE,
                i32::MAX as libc::c_ulong,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
            );
        };
    }
}

// No windows for now, since I believe there are other deps which dont support it anyway.


#[cfg(test)]
mod tests {
    use super::platform;
    use core::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_wait_and_wake_one() {
        let atomic_bool = Arc::new(AtomicBool::new(false));
        let barrier = Arc::new(Barrier::new(2));
        let atomic_clone = atomic_bool.clone();
        let barrier_clone = barrier.clone();

        // Spawn a thread that waits for the atomic to become true.
        let handle = thread::spawn(move || {
            // Signal that the thread is about to wait.
            barrier_clone.wait();
            platform::wait(&atomic_clone, false);
            assert!(atomic_clone.load(Ordering::SeqCst), "Thread did not detect correct value");
        });

        // Wait for the thread to reach the barrier, ensuring it is now waiting.
        barrier.wait();

        // Simulate a short delay to ensure the thread is waiting.
        thread::sleep(Duration::from_millis(100));
        assert!(!atomic_bool.load(Ordering::SeqCst), "Atomic value was set prematurely");

        // Set the atomic to true and wake one thread.
        atomic_bool.store(true, Ordering::SeqCst);
        platform::wake_one(Arc::as_ptr(&atomic_bool) as *const AtomicBool);

        // Wait for the thread to finish.
        handle.join().expect("Thread panicked");
    }

    #[test]
    fn test_wait_and_wake_all() {
        let atomic_bool = Arc::new(AtomicBool::new(false));
        let barrier = Arc::new(Barrier::new(6)); // One for each thread + main thread
        let mut handles = vec![];

        // Spawn multiple threads that wait for the atomic to become true.
        for _ in 0..5 {
            let atomic_clone = atomic_bool.clone();
            let barrier_clone = barrier.clone();
            handles.push(thread::spawn(move || {
                // Signal that the thread is about to wait.
                barrier_clone.wait();
                platform::wait(&atomic_clone, false);
                assert!(atomic_clone.load(Ordering::SeqCst), "Thread did not detect correct value");
            }));
        }

        // Wait for all threads to reach the barrier, ensuring they are now waiting.
        barrier.wait();

        // Simulate a short delay to ensure the threads are waiting.
        thread::sleep(Duration::from_millis(100));
        assert!(!atomic_bool.load(Ordering::SeqCst), "Atomic value was set prematurely");

        // Set the atomic to true and wake all threads.
        atomic_bool.store(true, Ordering::SeqCst);
        platform::wake_all(Arc::as_ptr(&atomic_bool) as *const AtomicBool);

        // Wait for all threads to finish.
        for handle in handles {
            handle.join().expect("Thread panicked");
        }
    }
}

