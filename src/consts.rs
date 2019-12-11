use snoozy::*;

#[snoozy]
pub async fn const_f32_snoozy(_ctx: Context, value: &f32) -> Result<f32> {
    Ok(*value)
}

#[snoozy]
pub async fn const_u32_snoozy(_ctx: Context, value: &u32) -> Result<u32> {
    Ok(*value)
}
