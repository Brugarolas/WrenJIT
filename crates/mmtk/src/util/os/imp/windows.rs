use crate::util::address::Address;
use crate::util::os::memory::*;
use std::io::Result;

/// Windows implementation of the `OS` trait.
pub struct Windows;

impl OSMemory for Windows {
    // Windows uses 4 KiB pages, independently of its allocation granularity.
    const LOG_BYTES_IN_PAGE: u8 = 12;

    fn dzmmap(
        start: Address,
        size: usize,
        strategy: MmapStrategy,
        annotation: &MmapAnnotation<'_>,
    ) -> MmapResult<Address> {
        use std::io;
        use windows_sys::Win32::System::Memory::{
            VirtualAlloc, VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, MEM_FREE, MEM_RESERVE,
        };

        let ptr: *mut u8 = start.to_mut_ptr();
        let mmap_error = |error| MmapError::new(start, size, annotation, error);

        // Has to COMMIT immediately if:
        // - not MAP_NORESERVE
        // - and protection is not NoAccess
        let commit = strategy.reserve && !matches!(strategy.prot, MmapProtection::NoAccess);

        // Scan the region [ptr, ptr + size) to understand its current state
        unsafe {
            let mut addr = ptr;
            let end = ptr.add(size);

            let mut saw_free = false;
            let mut saw_reserved = false;
            let mut saw_committed = false;

            while addr < end {
                let mut mbi: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
                let q = VirtualQuery(
                    addr as *const _,
                    &mut mbi,
                    std::mem::size_of::<MEMORY_BASIC_INFORMATION>(),
                );
                if q == 0 {
                    return Err(mmap_error(io::Error::last_os_error()));
                }

                let region_base = mbi.BaseAddress as *mut u8;
                let region_size = mbi.RegionSize;
                let region_end = region_base.add(region_size);

                // Calculate the intersection of [addr, end) and [region_base, region_end)
                let _sub_begin = if addr > region_base {
                    addr
                } else {
                    region_base
                };
                let _sub_end = if end < region_end { end } else { region_end };

                match mbi.State {
                    MEM_FREE => saw_free = true,
                    MEM_RESERVE => saw_reserved = true,
                    MEM_COMMIT => saw_committed = true,
                    _ => {
                        return Err(mmap_error(io::Error::other(
                            "Unexpected memory state in mmap_fixed",
                        )));
                    }
                }

                // Jump to the next region (VirtualQuery always returns "continuous regions with the same attributes")
                addr = region_end;
            }

            // 1. All FREE: make a new mapping in the region
            // 2. All RESERVE/COMMIT: treat as an existing mapping, can just COMMIT or succeed directly
            // 3. MIX of FREE + others: not allowed (semantically similar to MAP_FIXED_NOREPLACE)
            if saw_free && (saw_reserved || saw_committed) {
                return Err(mmap_error(io::Error::from_raw_os_error(
                    windows_sys::Win32::Foundation::ERROR_INVALID_ADDRESS as i32,
                )));
            }

            if saw_free && !saw_reserved && !saw_committed {
                // All FREE: make a new mapping in the region
                let mut allocation_type = MEM_RESERVE;
                if commit {
                    allocation_type |= MEM_COMMIT;
                }

                let res = VirtualAlloc(
                    ptr as *mut _,
                    size,
                    allocation_type,
                    strategy.prot.get_native_flags(),
                );
                if res.is_null() {
                    return Err(mmap_error(io::Error::last_os_error()));
                }

                Ok(start)
            } else {
                // This behavior is similar to mmap with MAP_FIXED on Linux.
                // If the region is already mapped, we just ensure the required commitment.
                // If commit is not needed, we just return Ok.
                if commit {
                    let res = VirtualAlloc(
                        ptr as *mut _,
                        size,
                        MEM_COMMIT,
                        strategy.prot.get_native_flags(),
                    );
                    if res.is_null() {
                        return Err(mmap_error(io::Error::last_os_error()));
                    }
                }
                Ok(start)
            }
        }
    }

    fn dzmmap_anywhere(
        size: usize,
        align: usize,
        strategy: MmapStrategy,
        annotation: &MmapAnnotation<'_>,
    ) -> MmapResult<Address> {
        Self::mmap_aligned(Address::ZERO, size, align, strategy, annotation)
    }

    fn dzmmap_preferred(
        start: Address,
        size: usize,
        align: usize,
        strategy: MmapStrategy,
        annotation: &MmapAnnotation<'_>,
    ) -> MmapResult<Address> {
        Self::mmap_aligned(start, size, align, strategy, annotation)
    }

    fn munmap(start: Address, size: usize) -> Result<()> {
        use windows_sys::Win32::System::Memory::*;
        // Using MEM_DECOMMIT will decommit the memory but leave the address space reserved.
        // This is the safest way to emulate munmap on Windows, as MEM_RELEASE would free
        // the entire allocation, which could be larger than the requested size.
        let res = unsafe { VirtualFree(start.to_mut_ptr(), size, MEM_DECOMMIT) };
        if res == 0 {
            // If decommit fails, we try to release the memory. This might happen if the memory was
            // only reserved.
            let res_release = unsafe { VirtualFree(start.to_mut_ptr(), 0, MEM_RELEASE) };
            if res_release == 0 {
                Err(std::io::Error::last_os_error())
            } else {
                Ok(())
            }
        } else {
            Ok(())
        }
    }

    fn set_memory_access(start: Address, size: usize, prot: MmapProtection) -> Result<()> {
        use windows_sys::Win32::System::Memory::*;
        let mut old_protect = 0;
        let res = unsafe {
            VirtualProtect(
                start.to_mut_ptr(),
                size,
                prot.get_native_flags(),
                &mut old_protect,
            )
        };
        if res == 0 {
            Err(std::io::Error::last_os_error())
        } else {
            Ok(())
        }
    }

    fn panic_if_unmapped(start: Address, size: usize) {
        warn!(
            "Check if {} of size {} is mapped is ignored on Windows",
            start, size
        );
    }

    fn is_mmap_oom(os_errno: i32) -> bool {
        os_errno == windows_sys::Win32::Foundation::ERROR_NOT_ENOUGH_MEMORY as i32
            || os_errno == windows_sys::Win32::Foundation::ERROR_INVALID_ADDRESS as i32
    }
}

impl Windows {
    fn mmap_aligned(
        preferred_start: Address,
        size: usize,
        align: usize,
        strategy: MmapStrategy,
        annotation: &MmapAnnotation<'_>,
    ) -> MmapResult<Address> {
        use std::ptr::{null, null_mut};
        use windows_sys::Win32::System::Memory::{
            MemExtendedParameterAddressRequirements, VirtualAlloc2, MEM_ADDRESS_REQUIREMENTS,
            MEM_COMMIT, MEM_EXTENDED_PARAMETER, MEM_EXTENDED_PARAMETER_0, MEM_EXTENDED_PARAMETER_1,
            MEM_RESERVE,
        };
        use windows_sys::Win32::System::SystemInformation::{GetSystemInfo, SYSTEM_INFO};

        let mmap_error = |error| MmapError::new(preferred_start, size, annotation, error);
        if !align.is_power_of_two()
            || align % Self::BYTES_IN_PAGE != 0
            || size == 0
            || size % Self::BYTES_IN_PAGE != 0
        {
            return Err(mmap_error(std::io::Error::from_raw_os_error(
                windows_sys::Win32::Foundation::ERROR_INVALID_PARAMETER as i32,
            )));
        }

        let mut info = SYSTEM_INFO::default();
        // SAFETY: info points to a writable SYSTEM_INFO for the duration of the call.
        unsafe { GetSystemInfo(&mut info) };
        let align = align.max(info.dwAllocationGranularity as usize);
        let mut allocation_type = MEM_RESERVE;
        if strategy.reserve && !matches!(strategy.prot, MmapProtection::NoAccess) {
            allocation_type |= MEM_COMMIT;
        }
        let prot = strategy.prot.get_native_flags();

        // A hint must not reuse or replace an existing reservation, even if strategy.replace
        // is set. MEM_RESERVE fails atomically if any part of the requested range is occupied.
        if !preferred_start.is_zero() && preferred_start.is_aligned_to(align) {
            // SAFETY: the OS validates the address range; no existing memory is dereferenced.
            // A non-null base address must be used without nonzero address requirements.
            let ptr = unsafe {
                VirtualAlloc2(
                    null_mut(),
                    preferred_start.to_ptr(),
                    size,
                    allocation_type,
                    prot,
                    null_mut(),
                    0,
                )
            };
            if !ptr.is_null() {
                return Ok(Address::from_mut_ptr(ptr));
            }
        }

        // VirtualAlloc2 (Windows 10+) can enforce alignment without overallocating and
        // trimming: unlike munmap, MEM_RELEASE cannot free part of a normal reservation.
        let mut requirements = MEM_ADDRESS_REQUIREMENTS {
            Alignment: align,
            ..Default::default()
        };
        let mut parameter = MEM_EXTENDED_PARAMETER {
            Anonymous1: MEM_EXTENDED_PARAMETER_0 {
                // Type occupies the low 8 bits; the remaining reserved bits must be zero.
                _bitfield: MemExtendedParameterAddressRequirements as u64,
            },
            Anonymous2: MEM_EXTENDED_PARAMETER_1 {
                Pointer: (&mut requirements as *mut MEM_ADDRESS_REQUIREMENTS).cast(),
            },
        };
        // SAFETY: parameter and its address requirements remain valid throughout the call.
        // The null process/base pointers request a new allocation in the current process.
        let ptr = unsafe {
            VirtualAlloc2(
                null_mut(),
                null(),
                size,
                allocation_type,
                prot,
                &mut parameter,
                1,
            )
        };
        if ptr.is_null() {
            Err(mmap_error(std::io::Error::last_os_error()))
        } else {
            // Newly committed pages are zero-filled by Windows. Huge-page hints and OS
            // mapping annotations are ignored, as permitted by OSMemory.
            Ok(Address::from_mut_ptr(ptr))
        }
    }
}

impl MmapProtection {
    fn get_native_flags(&self) -> u32 {
        use windows_sys::Win32::System::Memory::*;
        match self {
            Self::ReadWrite => PAGE_READWRITE,
            Self::ReadWriteExec => PAGE_EXECUTE_READWRITE,
            Self::NoAccess => PAGE_NOACCESS,
        }
    }
}

use crate::util::os::process::*;

impl OSProcess for Windows {
    type ProcessIDType = u32;
    type ThreadIDType = u32;

    fn get_process_memory_maps() -> Result<String> {
        // Windows-specific implementation to get process memory maps
        Ok("get_process_memory_maps not implemented for Windows".to_string())
    }

    fn get_process_id() -> Result<Self::ProcessIDType> {
        use windows_sys::Win32::System::Threading::GetCurrentProcessId;
        Ok(unsafe { GetCurrentProcessId() })
    }

    fn get_thread_id() -> Result<Self::ThreadIDType> {
        use windows_sys::Win32::System::Threading::GetCurrentThreadId;
        Ok(unsafe { GetCurrentThreadId() })
    }

    fn get_total_num_cpus() -> CoreNum {
        unsafe {
            windows_sys::Win32::System::Threading::GetActiveProcessorCount(
                windows_sys::Win32::System::Threading::ALL_PROCESSOR_GROUPS,
            ) as CoreNum
        }
    }

    fn bind_current_thread_to_core(core_id: CoreId) {
        unsafe {
            windows_sys::Win32::System::Threading::SetThreadAffinityMask(
                windows_sys::Win32::System::Threading::GetCurrentThread(),
                1 << core_id,
            );
        }
    }

    fn bind_current_thread_to_cpuset(core_ids: &[CoreId]) {
        let mut mask = 0;
        for cpu in core_ids {
            mask |= 1 << cpu;
        }
        unsafe {
            windows_sys::Win32::System::Threading::SetThreadAffinityMask(
                windows_sys::Win32::System::Threading::GetCurrentThread(),
                mask,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::heap::layout::vm_layout::BYTES_IN_CHUNK;
    use crate::util::test_util::serial_test;
    use windows_sys::Win32::System::Memory::{
        VirtualFree, VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE,
    };

    const ANNOTATION: MmapAnnotation<'static> = MmapAnnotation::Misc {
        name: "windows_mmap_test",
    };

    struct Mapping(Address);

    impl Drop for Mapping {
        fn drop(&mut self) {
            // SAFETY: these tests own the entire reservation returned by the allocator.
            // Release it directly because Windows::munmap normally only decommits pages.
            assert_ne!(unsafe { VirtualFree(self.0.to_mut_ptr(), 0, MEM_RELEASE) }, 0);
        }
    }

    fn query(start: Address) -> MEMORY_BASIC_INFORMATION {
        let mut info = MEMORY_BASIC_INFORMATION::default();
        // SAFETY: info is writable and VirtualQuery only inspects the supplied address.
        assert_ne!(
            unsafe { VirtualQuery(start.to_ptr(), &mut info, std::mem::size_of_val(&info)) },
            0
        );
        info
    }

    #[test]
    fn mmap_anywhere_reserves_aligned_range_and_can_commit_it() {
        serial_test(|| {
            // The size is deliberately not a multiple of the larger alignments.
            let size = BYTES_IN_CHUNK + Windows::BYTES_IN_PAGE;
            for align in [Windows::BYTES_IN_PAGE, 64 * 1024, BYTES_IN_CHUNK] {
                let mapping = Mapping(
                    Windows::dzmmap_anywhere(size, align, MmapStrategy::QUARANTINE, &ANNOTATION)
                        .unwrap(),
                );
                assert!(mapping.0.is_aligned_to(align));
                let info = query(mapping.0);
                assert_eq!(Address::from_mut_ptr(info.AllocationBase), mapping.0);
                assert_eq!(info.RegionSize, size);
                assert_eq!(info.State, MEM_RESERVE);
                assert_eq!(query(mapping.0 + size - 1).State, MEM_RESERVE);

                assert_eq!(
                    Windows::dzmmap(mapping.0, size, MmapStrategy::TEST, &ANNOTATION).unwrap(),
                    mapping.0
                );
                assert_eq!(query(mapping.0).State, MEM_COMMIT);
                // SAFETY: the entire reservation is now committed and writable.
                unsafe {
                    let bytes = std::slice::from_raw_parts_mut(mapping.0.to_mut_ptr::<u8>(), size);
                    assert!(bytes.iter().all(|byte| *byte == 0));
                    bytes[0] = 1;
                    bytes[size - 1] = 2;
                }
            }
        });
    }

    #[test]
    fn mmap_anywhere_respects_commit_and_protection() {
        serial_test(|| {
            for (strategy, expected_state) in [
                (MmapStrategy::TEST.reserve(false), MEM_RESERVE),
                (MmapStrategy::TEST.prot(MmapProtection::NoAccess), MEM_RESERVE),
                (MmapStrategy::TEST, MEM_COMMIT),
                (
                    MmapStrategy::TEST.prot(MmapProtection::ReadWriteExec),
                    MEM_COMMIT,
                ),
            ] {
                let mapping = Mapping(
                    Windows::dzmmap_anywhere(
                        Windows::BYTES_IN_PAGE,
                        Windows::BYTES_IN_PAGE,
                        strategy,
                        &ANNOTATION,
                    )
                    .unwrap(),
                );
                let info = query(mapping.0);
                assert_eq!(info.State, expected_state);
                if expected_state == MEM_COMMIT {
                    assert_eq!(info.Protect, strategy.prot.get_native_flags());
                    // SAFETY: both committed strategies grant read/write access to this page.
                    unsafe {
                        assert_eq!(*mapping.0.to_ptr::<u8>(), 0);
                        *mapping.0.to_mut_ptr::<u8>() = 42;
                    }
                }
            }
        });
    }

    #[test]
    fn mmap_preferred_uses_available_hint() {
        serial_test(|| {
            let mapping = Mapping(
                Windows::dzmmap_anywhere(
                    Windows::BYTES_IN_PAGE,
                    BYTES_IN_CHUNK,
                    MmapStrategy::QUARANTINE,
                    &ANNOTATION,
                )
                .unwrap(),
            );
            let hint = mapping.0;
            drop(mapping);
            let mapping = Mapping(
                Windows::dzmmap_preferred(
                    hint,
                    Windows::BYTES_IN_PAGE,
                    BYTES_IN_CHUNK,
                    MmapStrategy::QUARANTINE,
                    &ANNOTATION,
                )
                .unwrap(),
            );
            assert_eq!(mapping.0, hint);
            assert_eq!(query(mapping.0).State, MEM_RESERVE);
        });
    }

    #[test]
    fn mmap_preferred_falls_back_without_changing_existing_memory() {
        serial_test(|| {
            let original = Mapping(
                Windows::dzmmap_anywhere(
                    BYTES_IN_CHUNK,
                    BYTES_IN_CHUNK,
                    MmapStrategy::TEST,
                    &ANNOTATION,
                )
                .unwrap(),
            );
            // SAFETY: original owns a committed read/write mapping.
            unsafe { *original.0.to_mut_ptr::<u8>() = 42 };
            // Exercise both an occupied aligned hint and a hint that is only page-aligned.
            for hint in [original.0, original.0 + Windows::BYTES_IN_PAGE] {
                let mapping = Mapping(
                    Windows::dzmmap_preferred(
                        hint,
                        BYTES_IN_CHUNK,
                        BYTES_IN_CHUNK,
                        MmapStrategy::TEST,
                        &ANNOTATION,
                    )
                    .unwrap(),
                );
                assert!(mapping.0.is_aligned_to(BYTES_IN_CHUNK));
                assert_ne!(mapping.0, original.0);
                // SAFETY: both mappings are still committed and readable.
                unsafe {
                    assert_eq!(*original.0.to_ptr::<u8>(), 42);
                    assert_eq!(*mapping.0.to_ptr::<u8>(), 0);
                }
            }
        });
    }

    #[test]
    fn mmap_errors_preserve_context() {
        let page = Windows::BYTES_IN_PAGE;
        // A page-aligned size too large for the address space also exercises an OS failure.
        for (size, align) in [
            (0, page),
            (page + 1, page),
            (page, 0),
            (page, 3 * page),
            (usize::MAX & !(page - 1), page),
        ] {
            let hint = Address::ZERO + BYTES_IN_CHUNK;
            let error = Windows::dzmmap_preferred(
                hint,
                size,
                align,
                MmapStrategy::QUARANTINE,
                &ANNOTATION,
            )
            .unwrap_err();
            assert_eq!(error.error_address, hint);
            assert_eq!(error.bytes, size);
            assert_eq!(error.annotation, ANNOTATION.to_string());
            assert!(error.error.raw_os_error().is_some());
        }
    }
}
