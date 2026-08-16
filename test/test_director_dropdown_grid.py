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


def test_aspect_menu_grouped_by_orientation():
    """Aspect dropdown must render as columns: General / Horizontal / Square / Vertical,
    with horizontal and vertical ratios sorted ascending per column."""
    src = _source()
    assert 'title: "General"' in src
    assert 'title: "Horizontal"' in src
    assert 'title: "Square"' in src
    assert 'title: "Vertical"' in src
    assert 'aspectItems[w > h ? "horizontal" : w < h ? "vertical" : "square"].push(pair)' in src
    assert "aspectItems.horizontal.sort((a, b) => ratioOf(a[0]) - ratioOf(b[0]))" in src
    assert "aspectItems.vertical.sort((a, b) => ratioOf(a[0]) - ratioOf(b[0]))" in src


def test_resolution_menu_grouped_by_p_and_mp():
    """Resolution dropdown must render as columns: General / ###p / MP,
    sorted ascending within each column."""
    src = _source()
    assert 'title: "###p"' in src
    assert 'title: "MP"' in src
    assert "resNames.filter(n => !/MP/.test(n)).sort((a, b) => pVal(a) - pVal(b))" in src
    assert "resNames.filter(n => /MP/.test(n)).sort((a, b) => mpVal(a) - mpVal(b))" in src
    # 2K (2000) must sort between 1440p and 2160p: K-suffix expansion is required.
    assert 'String(n).replace(/K$/i, "000").replace(/p$/i, "")' in src


def test_grouped_menu_uses_column_layout_css():
    """Grouped dropdowns must use the .cols flex-column layout with titled columns."""
    src = _source()
    assert 'menu.className = hasGroups ? "ds-h3-res-menu cols"' in src
    assert ".ds-h3-res-menu.cols.open{display:flex;align-items:flex-start;gap:7px}" in src
    assert ".ds-h3-res-col{display:flex;flex-direction:column;gap:2px;min-width:84px}" in src
    assert ".ds-h3-res-col-title{" in src


def test_resolution_presets_snap_to_h3_transformer_patch_grid():
    """H3 patchifies its 16px VAE cells in 2×2 groups, requiring 32px edges."""
    src = _source()
    assert "const MINIMAX_MULTIPLE = 32;" in src
    assert "32px H3 grid" in src
