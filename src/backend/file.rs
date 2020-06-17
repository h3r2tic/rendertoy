use notify::{DebouncedEvent, RecommendedWatcher, RecursiveMode, Watcher};

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::channel;
use std::sync::Mutex;
use std::thread;
use std::time::Duration;

struct FileWatcher {
    watcher: RecommendedWatcher,
    callbacks: HashMap<PathBuf, Box<dyn Fn() + Sync + Send>>,
}

impl FileWatcher {
    fn new() -> FileWatcher {
        let (tx, rx) = channel();
        let watcher: RecommendedWatcher = Watcher::new(tx, Duration::from_millis(100)).unwrap();

        thread::spawn(move || loop {
            match rx.recv() {
                Ok(DebouncedEvent::Write(path)) => {
                    if let Some(ref callback) = FILE_WATCHER.lock().unwrap().callbacks.get(&path) {
                        callback();
                    }
                    //println!("Detected file modification: {:?}", path)
                }
                Err(e) => tracing::error!("watch error: {:?}", e),
                _ => (),
            }
        });

        FileWatcher {
            watcher,
            callbacks: HashMap::new(),
        }
    }

    fn watch<F: Fn() + Sync + Send + 'static>(&mut self, path: &str, callback: F) {
        let path = Path::new(path).canonicalize().unwrap();
        if !self.callbacks.contains_key(&path) {
            //println!("Watching file {:?}", path);
            self.watcher
                .watch(path.clone(), RecursiveMode::NonRecursive)
                .unwrap();
        }

        self.callbacks.insert(path, Box::new(callback));
    }
}

lazy_static! {
    static ref FILE_WATCHER: Mutex<FileWatcher> = Mutex::new(FileWatcher::new());
}

pub(crate) fn watch_file<F: Fn() + Sync + Send + 'static>(path: &str, callback: F) {
    FILE_WATCHER.lock().unwrap().watch(path, callback);
}
