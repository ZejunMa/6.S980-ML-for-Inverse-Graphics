from jaxtyping import install_import_hook

# Add runtime type checking to all imports.
with install_import_hook(("src",), ("beartype", "beartype")):
    from src.components.positional_encoding import PositionalEncoding

    from .f32 import f32


def test_d_out():
    pe = PositionalEncoding(7)
    assert pe.d_out(1) == 14
    assert pe.d_out(2) == 28
    assert pe.d_out(3) == 42

    pe = PositionalEncoding(11)
    assert pe.d_out(1) == 22
    assert pe.d_out(2) == 44
    assert pe.d_out(3) == 66


def test_lowest_frequency():
    pe = PositionalEncoding(1)

    expected = f32([1, 0, -1])
    actual = pe(f32([0.25, 0.5, 0.75]))

    for entry in actual:
        for possible in expected:
            if entry.isclose(possible, atol=1e-3):
                break
        else:
            # no match found
            assert False
