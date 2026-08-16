import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const sourcePath = new URL("../js/minimax_h3_director.js", import.meta.url);
let source = await readFile(sourcePath, "utf8");
source = source.replace(
    'import { app } from "../../../scripts/app.js";\nimport { api } from "../../../scripts/api.js";',
    "const app = { registerExtension() {} }; const api = {};"
);
source += "\nexport { mediaTypeFor, REPOSITORY_URL };";

assert.match(source, /lane\.ondrop = event => \{ if \(!supported\)/, "each timeline lane must own its direct drop handler");
assert.match(source, /acceptLaneDrop\(event, targetLane\)/, "supported lane drops must use the direct lane upload path");
assert.match(source, /resizer\.className = "ds-h3-prompt-field-resizer"/, "each prompt field must expose a bottom resize grabber");
assert.match(source, /wrapper\.appendChild\(resizer\)/, "the resize grabber must live inside the prompt field");
assert.match(source, /\.ds-h3\{box-sizing:border-box[^`]*background:transparent;border:0;border-radius:0;padding:0/, "the Director UI must render directly on the node without an outer panel");
assert.doesNotMatch(source, /promptPanelHeight/, "restoring a workflow must not reference the removed prompt-panel divider state");
assert.match(source, /FL2VA supports image references only; video and audio are unavailable\./, "unsupported drops must show an explicit FL2VA error");
assert.match(source, /ds-h3-timeline-lane \$\{targetLane\}[^`]*\$\{supported \? "" : " disabled"\}/, "unsupported lanes must be visibly blocked");
assert.match(source, /removeButton\.textContent = "Remove"/, "selected media must have a toolbar remove button");
assert.match(source, /close\.className = "ds-h3-clip-close"/, "selected media must have a clip-corner remove button");
assert.doesNotMatch(source, /Remove selected media item/, "the old prompt-panel remove button must be absent");
assert.match(source, /\.ds-h3-prompt-panel\{width:100%;box-sizing:border-box;border:0;border-radius:0;padding:0/, "prompt fields must not sit inside another framed panel");
assert.match(source, /ds-h3-status ds-h3-info-field/, "status messages must use their own info field");
assert.match(source, /Math\.min\(sourceDuration, 15\)/, "long uploaded media must default to a 15-second crop");
assert.match(source, /extractWaveform\(value, added\.id\)/, "audio uploads must decode a waveform");
assert.match(source, /waveform\.className = "ds-h3-waveform"/, "audio clips must render their waveform canvas");
assert.match(source, /let selectedLane = "visual"/, "the timeline must keep an active destination lane");
assert.match(source, /ds-h3-timeline-lane\.selected/, "the active lane must be visibly highlighted");
assert.match(source, /timeline\.addEventListener\("paste", async event =>/, "the timeline must accept clipboard paste events");
assert.match(source, /await acceptFile\(file, selectedLane\)/, "clipboard files must be routed to the selected lane");
assert.match(source, /lane selected\. Paste files here with Ctrl\+V\./, "lane selection must explain the clipboard interaction");
assert.doesNotMatch(source, /FL2VA: \$\{timelineSeconds\.toFixed\(2\)\}s/, "the obsolete time-scale hint must be absent");
assert.match(source, /audioSlotWidth = sourceDuration =>.*Math\.log2/, "audio slots must fit long source files logarithmically");
assert.match(source, /ds-h3-audio-crop-marker start/, "audio clips must draw a crop-start position marker");
assert.match(source, /ds-h3-audio-crop-marker end/, "audio clips must draw a crop-end position marker");
assert.match(source, /Math\.log1p\(9 \* peaks\[peakIndex\]\)/, "audio waveform amplitudes must use logarithmic scaling");
assert.doesNotMatch(source, /if \(item\.type === "audio"\) return; if \(event\.target !== clip\) return;/, "audio slots must be horizontally movable");
assert.match(source, /window\.open\(REPOSITORY_URL, "_blank", "noopener,noreferrer"\)/, "the help button must open the GitHub documentation safely");
assert.match(source, /const promptStyle = \(\) => \{ const v = builderState\?\.prompt_mode;/, "the builder must expose a validated prompt-style accessor");
assert.match(source, /styleButton\.className = "ds-h3-prompt-mode-btn"/, "the mode bar must render a prompt-style toggle button");
assert.match(source, /builderState\.prompt_mode = styleLabel === "simple" \? "structured" : "simple"/, "the toggle must flip between the simple and structured styles");
assert.match(source, /function buildSimpleForm\(panel\)/, "simple mode must render its own one-field prompt form");
assert.match(source, /createBuilderField\("Prompt", builderState\.simple_prompt/, "simple mode must bind the visible field to serialized simple_prompt state");
assert.match(source, /if \(styleLabel !== "simple"\) builderState\.simple_prompt = previewTextFor\(mode\(\), false\);/, "switching into simple mode must seed the single field from the current builder prompt");
assert.match(source, /\.ds-h3-prompt-mode-btn\{[^`]*box-shadow:0 0 8px rgba\(126,235,167/, "the prompt-style toggle must use the green-glow accent");
assert.match(source, /function showPromptPreview\(\) \{[\s\S]*?const promptText = previewTextFor\(m, hasExternalPrompt\(\)\);/, "the prompt preview must render the selected style");
assert.doesNotMatch(source, /builderState\.prompt_mode = styleLabel === "structured" \? "simple" : "structured"/, "the toggle must not invert the wrong way");
assert.match(source, /const resetBuilderState = \(\) => \{ builderState = DEFAULT_BUILDER_STATE\(mode\(\)\); builderState\.mode = mode\(\); \};/, "Clear must reset all serialized builder fields to the current mode defaults");
assert.match(source, /const clearAll = \(\) => \{ selectedId = null; resetBuilderState\(\);/, "Clear must reset prompt/text fields as well as selected media");
assert.match(source, /const hasContent = state\.items\.length \|\| state\.prompt_blocks\?\.length \|\| hasBuilderContent\(\) \|\| String\(promptWidget\?\.value \|\| ""\)\.trim\(\);/, "Clear must remain available when only builder text is filled");

const moduleUrl = `data:text/javascript;base64,${Buffer.from(source).toString("base64")}`;
const { mediaTypeFor, REPOSITORY_URL } = await import(moduleUrl);

for (const [name, type] of [
    ["reference.webp", "image"],
    ["reference.heic", "image"],
    ["reference.mkv", "video"],
    ["reference.m2ts", "video"],
    ["reference.flac", "audio"],
    ["reference.opus", "audio"],
]) {
    assert.equal(mediaTypeFor({ name, type: "" }), type, `${name} must classify by extension when the browser omits MIME type`);
}
assert.equal(mediaTypeFor({ name: "reference.bin", type: "" }), null, "unknown files must not be uploaded as media");
assert.equal(mediaTypeFor({ name: "renamed.bin", type: "audio/ogg" }), "audio", "known MIME types take precedence over extensions");
assert.equal(REPOSITORY_URL, "https://github.com/darksidewalker/ComfyUI-DaSiWa-Nodes/blob/main/docs/minimax_h3_director.md");
