use cargo_metadata::MetadataCommand;
use snoozy::*;
use std::collections::HashMap;

#[derive(Debug)]
pub struct CargoPackageMap {
    pub deps: HashMap<String, String>,
}

#[snoozy]
pub async fn load_cargo_package_map_snoozy(_ctx: Context) -> Result<CargoPackageMap> {
    let metadata = MetadataCommand::new()
        .manifest_path("./Cargo.toml")
        .exec()
        .unwrap();

    let deps = metadata
        .packages
        .iter()
        .map(|p| {
            assert_eq!(
                p.manifest_path.file_name().and_then(|p| p.to_str()),
                Some("Cargo.toml")
            );
            let path = p
                .manifest_path
                .parent()
                .unwrap()
                .to_string_lossy()
                .into_owned();
            (p.name.clone(), path)
        })
        .collect();

    Ok(CargoPackageMap { deps })
}

// Explicitly not serializable, so that it doesn't end up auto-cached
pub struct CargoDependencyPath(pub String);

#[snoozy]
pub async fn get_cargo_package_dep_path_snoozy(
    mut ctx: Context,
    package: &String,
) -> Result<CargoDependencyPath> {
    let map = ctx.get(load_cargo_package_map()).await?;

    if let Some(path) = map.deps.get(package) {
        Ok(CargoDependencyPath(path.clone()))
    } else {
        Err(format_err!("Package not found: {}", *package))
    }
}
