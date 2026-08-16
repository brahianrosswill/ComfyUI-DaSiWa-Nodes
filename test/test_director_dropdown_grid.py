"""Source-level regression tests for the Director resolution/aspect dropdown menus."""
from pathlib import Path

JS = Path(__file__).resolve().parent.parent / "js" / "minimax_h3_director.js"


def _source() -> str:
    return JS.read_text(encoding="utf-8")


def test_wheel_over_open_menu_does_not_zoom_canvas():
    """Wheeling over an open, scrollable dropdown must not forward zoom to the canvas."""
    src = _source()
    # The timeline wheel handler must bail out while the pointer is inside an open res menu
    # that still has content to scroll (native scroll), instead of dispatching to app.canvas.
    assert 'closest(".ds-h3-res-menu.open")' in src
    assert "openMenu && openMenu.scrollHeight > openMenu.clientHeight" in src


def test_large_dropdowns_render_as_grid():
    """Dropdowns with many options must render as a CSS grid, not a 1-column list."""
    src = _source()
    assert 'options.length >= 6 ? "ds-h3-res-menu grid" : "ds-h3-res-menu"' in src
    assert "gridTemplateColumns" in src
    assert "repeat(${cols}, minmax(0, 1fr))" in src
    # Grid menus need room: capped at 420px with native scroll fallback if it overflows.
    assert 'menu.style.maxHeight = "420px"' in src
    # Grid layout CSS must actually be installed.
    assert ".ds-h3-res-menu.grid.open{display:grid;gap:2px}" in src


def test_menu_flips_up_and_clamps_to_viewport():
    """Menus must open above the button when there is not enough room below, and
    clamp horizontally so the full grid is never clipped by the view borders."""
    src = _source()
    assert 'menu.dataset.place = menuRect.height > spaceBelow && spaceAbove > spaceBelow ? "up" : "down"' in src
    assert 'left + menuRect.width > window.innerWidth - margin' in src
    assert '.ds-h3-res-menu[data-place="up"]{top:auto;bottom:calc(100% + 4px)}' in src
