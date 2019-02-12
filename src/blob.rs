use super::package::get_cargo_package_dep_path;
use snoozy::*;
use std::fs::File;
use std::path::PathBuf;

#[derive(Hash, Debug)]
pub struct Blob {
    pub contents: Vec<u8>,
}

snoozy! {
    fn load_blob(ctx: &mut Context, path: &AssetPath) -> Result<Blob> {
        let mut buffer = Vec::new();

        let mut file_path: PathBuf = (*ctx.get(get_cargo_package_dep_path(path.crate_name.clone()))?).clone().into();
        file_path.push("assets");
        file_path.push(&path.asset_name);

        let file_path = &file_path.to_string_lossy().to_string();

        println!("Loading {}\n    -> {}", path, file_path);

        std::io::Read::read_to_end(&mut File::open(file_path)?, &mut buffer)?;
        crate::backend::file::watch_file(file_path, ctx.get_invalidation_trigger());

        Ok(Blob { contents: buffer })
    }
}

#[derive(Serialize, Debug, Clone)]
pub struct AssetPath {
    pub crate_name: String,
    pub asset_name: String,
}

impl std::fmt::Display for AssetPath {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}::{}", self.crate_name, self.asset_name)
    }
}

pub fn rendertoy_asset_path(crate_name: &str, partial_path: &str) -> AssetPath {
    if let Some(crate_end) = partial_path.find("::") {
        let crate_name = partial_path.chars().take(crate_end).collect();
        let asset_name = partial_path.chars().skip(crate_end + 2).collect();

        AssetPath {
            crate_name,
            asset_name,
        }
    } else {
        AssetPath {
            crate_name: crate_name.to_string(),
            asset_name: partial_path.to_string(),
        }
    }
}

#[macro_export]
macro_rules! asset {
    ($asset_path:expr) => {
        rendertoy_asset_path(env!("CARGO_PKG_NAME"), $asset_path)
    };
}