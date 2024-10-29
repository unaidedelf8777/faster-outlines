// This is basically identical to https://docs.rs/atomic-wait/latest/atomic_wait/
// just modified so that it works with AtomicBool instead of AtomicU32
#![allow(dead_code)]
#[cfg(target_os = "linux")]
pub mod platform {
    use core::sync::atomic::AtomicBool;
    use libc;

    #[inline]
    pub fn wait(a: &AtomicBool, expected: bool) {
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
    use core::sync::atomic::AtomicBool;
    use libc;

    #[inline]
    pub fn wait(a: &AtomicBool, expected: bool) {
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
        };
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
