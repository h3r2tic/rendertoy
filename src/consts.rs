use snoozy::*;

snoozy! {
    fn const_f32(_ctx: &mut Context, value: &f32) -> Result<f32> {
        Ok(*value)
    }
}
