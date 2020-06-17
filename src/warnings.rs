lazy_static! {
    static ref RTOY_WARNINGS: std::sync::Mutex<Vec<String>> =
        std::sync::Mutex::new(Default::default());
}

pub fn rtoy_show_warning(text: String) {
    RTOY_WARNINGS.lock().unwrap().push(text);
}

pub fn with_drain_warnings(callback: impl Fn(&mut Vec<String>)) {
    let mut warnings = RTOY_WARNINGS.lock().unwrap();
    callback(&mut warnings);
    warnings.clear();
}
