#[macro_export]
macro_rules! rtoy_request_discrete_gpu {
    () => {
        #[cfg(target_os = "windows")]
        #[used]
        #[no_mangle]
        #[allow(non_upper_case_globals)]
        pub static NvOptimusEnablement: i32 = 1;

        #[cfg(target_os = "windows")]
        #[used]
        #[no_mangle]
        #[allow(non_upper_case_globals)]
        pub static AmdPowerXpressRequestHighPerformance: i32 = 1;

        #[cfg(target_os = "windows")]
        #[allow(unused_attributes)]
        #[link_args = "/EXPORT:NvOptimusEnablement,DATA /EXPORT:AmdPowerXpressRequestHighPerformance,DATA"]
        extern {}
    };
}
