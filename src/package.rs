use cargo::core::source::MaybePackage;
use cargo::core::Workspace;
use cargo::ops::load_pkg_lockfile;
use cargo::util::{important_paths, Config};
use snoozy::*;

#[derive(Hash, Debug)]
pub struct CargoPackageMap {
    pub name: String,
    pub path: String,
    pub deps: Vec<(String, String)>,
}

#[snoozy]
pub fn load_cargo_package_map(_ctx: &mut Context) -> Result<CargoPackageMap> {
    let config = Config::default()?;
    let package_lock = config.acquire_package_cache_lock()?;

    let root = important_paths::find_root_manifest_for_wd(config.cwd())?;
    println!("Loading cargo package map from {}", root.to_string_lossy());

    let ws = Workspace::new(&root, &config)?;
    let package = ws.load(&root).unwrap();
    let resolve = load_pkg_lockfile(&ws)?.unwrap();

    let mut deps = Vec::new();

    //println!("package: {:?}\ndependencies:\n", package.package_id());
    for (dep, _) in resolve.deps(package.package_id()) {
        let mut source = dep.source_id().load(&config, &Default::default()).unwrap();
        source.update()?;

        if let MaybePackage::Ready(pkg) = source.download(dep).unwrap() {
            //println!("{} -> {}", pkg.name(), pkg.root().to_string_lossy());
            deps.push((
                pkg.name().to_string(),
                pkg.root().to_string_lossy().to_string(),
            ));
        }
    }

    let path = root.parent().unwrap().to_string_lossy().to_string();
    std::mem::drop(package_lock);

    Ok(CargoPackageMap {
        name: package.name().to_string(),
        path,
        deps,
    })
}

// Explicitly not serializable, so that it doesn't end up auto-cached
pub struct CargoDependencyPath(pub String);

#[snoozy]
pub fn get_cargo_package_dep_path(
    ctx: &mut Context,
    package: &String,
) -> Result<CargoDependencyPath> {
    let map = ctx.get(load_cargo_package_map())?;

    if *package == map.name {
        return Ok(CargoDependencyPath(map.path.clone()));
    }

    for (name, path) in map.deps.iter() {
        if name == package {
            return Ok(CargoDependencyPath(path.clone()));
        }
    }

    Err(format_err!("Package not found: {}", *package))
}
