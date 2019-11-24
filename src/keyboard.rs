use std::collections::HashMap;
pub use winit::{ElementState, KeyboardInput, VirtualKeyCode};

pub struct KeyState {
    ticks: u32,
    seconds: f32,
}

pub struct KeyboardState {
    keys_down: HashMap<VirtualKeyCode, KeyState>,
    events: Vec<KeyboardInput>,
}

impl KeyboardState {
    pub fn new() -> Self {
        Self {
            keys_down: HashMap::new(),
            events: Vec::new(),
        }
    }

    pub fn is_down(&self, key: VirtualKeyCode) -> bool {
        self.get_down(key).is_some()
    }

    pub fn get_down(&self, key: VirtualKeyCode) -> Option<&KeyState> {
        self.keys_down.get(&key)
    }

    pub fn iter_events(&self) -> impl Iterator<Item = &KeyboardInput> {
        self.events.iter()
    }

    pub(crate) fn update(&mut self, events: Vec<KeyboardInput>, dt: f32) {
        self.events = events;

        for event in &self.events {
            if let Some(vk) = event.virtual_keycode {
                if event.state == ElementState::Pressed {
                    self.keys_down.entry(vk).or_insert(KeyState {
                        ticks: 0,
                        seconds: 0.0,
                    });
                } else {
                    self.keys_down.remove(&vk);
                }
            }
        }

        for ks in self.keys_down.values_mut() {
            ks.ticks += 1;
            ks.seconds += dt;
        }
    }
}
