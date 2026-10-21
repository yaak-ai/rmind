use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};
use std::io::{self, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let calls: usize = arguments.next().ok_or("expected call count")?.parse()?;
    let width: usize = arguments.next().unwrap_or_else(|| "150".into()).parse()?;
    let mut random = StdRng::seed_from_u64(287);
    let mut output = io::BufWriter::new(io::stdout().lock());
    for _ in 0..calls * width {
        let value: f32 = StandardNormal.sample(&mut random);
        output.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeded_prefix_matches_pinned_serving_versions() {
        let mut random = StdRng::seed_from_u64(287);
        let expected = [
            0x3e863259, 0x3ede202d, 0x3f61897e, 0xbee76fb3, 0xbface154, 0x3dbcfb1b, 0x3f8c316a,
            0x3ea0251e,
        ];
        for bits in expected {
            let value: f32 = StandardNormal.sample(&mut random);
            assert_eq!(value.to_bits(), bits);
        }
    }
}
