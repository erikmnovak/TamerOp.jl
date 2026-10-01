# Presentation choices shared by static figures and live HTML inspectors.

const _VISUAL_COLOR_ROLES = (:source, :target, :both, :selected, :inactive,
    :support, :absent, :unrepresented, :foreground, :muted, :edge,
    :positive, :negative, :zero, :finite, :essential, :background, :surface, :border, :error, :missing)

# A semantic reference, not an encoded color name. Its meaning survives palettes.
struct _VisualRole
    kind::Symbol
    index::Int
    function _VisualRole(kind::Symbol, index::Int=0)
        kind in _VISUAL_COLOR_ROLES || kind === :categorical ||
            throw(ArgumentError("Unknown visual role $kind."))
        kind === :categorical ? index > 0 || throw(ArgumentError("A categorical role needs a positive index.")) :
            index == 0 || throw(ArgumentError("Only categorical roles take an index."))
        new(kind, index)
    end
end

const _ACCESSIBLE_VISUAL_COLORS = (
    source="#0072B2", target="#A84600", both="#8B4F83", selected="#202020",
    inactive="#6B6B6B", support="#007F5F", absent="#D9D9D9", unrepresented="#F0F0F0",
    foreground="#202020", muted="#595959", edge="#666666", positive="#A84600",
    negative="#0072B2", zero="#6B6B6B", finite="#0072B2", essential="#A84600", background="#FFFFFF", surface="#F7F7F7",
    border="#B3B3B3", error="#A51B16", missing="#E6E6E6")
const _GRAYSCALE_VISUAL_COLORS = (
    source="#202020", target="#666666", both="#404040", selected="#000000",
    inactive="#6B6B6B", support="#595959", absent="#D9D9D9", unrepresented="#F0F0F0",
    foreground="#202020", muted="#595959", edge="#666666", positive="#202020",
    negative="#666666", zero="#6B6B6B", finite="#202020", essential="#666666", background="#FFFFFF", surface="#F7F7F7",
    border="#B3B3B3", error="#202020", missing="#E6E6E6")
const _VISUAL_CATEGORY_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7",
    "#56B4E9", "#D55E00", "#999933", "#666666")
const _VISUAL_CATEGORY_GRAYS = ("#404040", "#909090", "#606060", "#B0B0B0", "#202020", "#D0D0D0")

"""
    VisualStyle(; palette=:accessible, fontsize=16, font="DejaVu Sans",
                mono_font="DejaVu Sans Mono", linewidth_scale=1,
                markersize_scale=1, gap=12, padding=16,
                colormap=nothing, colors=(;))

Shared appearance for `visualize`, `render`, `save_visual`, and `save_visuals`.
Pass `style=VisualStyle(...)` to one rendering/export call; it does not change
the module, its selection, a visualization specification, or global Makie themes.

`palette` is `:accessible` (blue/orange roles with non-color selection cues) or
`:grayscale`. `colors` overrides named roles, for example
`colors=(source=:darkblue, target=:darkorange)`. Roles remain independent of
colors, so coinciding custom colors do not erase source/target labels or shapes.
Custom colors accept Makie color names or hex strings; backends validate names.

`fontsize` sets the base pixel size; headings and layer text scale relative to
it. `linewidth_scale` and `markersize_scale` multiply layer choices, preserving
their relative emphasis; data-space marker extents remain mathematical sizes.
`gap` and `padding` are nonnegative pixel distances. `font` and `mono_font` name
fonts available to Makie; browsers use them when installed and otherwise fall
back to sans-serif and monospace fonts.

`colormap=nothing` retains each recipe's numerical colormap, except that
`:grayscale` uses `:grays`. A supplied colormap symbol overrides numerical
colormaps. Literal layer colors are retained in the accessible palette and
converted to luminance by grayscale renderers. Missing cells remain distinct
from numerical zero; styles never change data or numerical color limits.

Use `describe(style)` to inspect all effective choices. Unknown options, roles,
palettes and invalid sizes are rejected. Renderer style options do not belong
to `visual_spec`, which describes the mathematical picture.
"""
struct VisualStyle{C<:NamedTuple}
    palette::Symbol
    fontsize::Float64
    font::String
    mono_font::String
    linewidth_scale::Float64
    markersize_scale::Float64
    gap::Float64
    padding::Float64
    colormap::Union{Nothing,Symbol}
    colors::C
    function VisualStyle(; palette=:accessible, fontsize=16, font="DejaVu Sans",
                         mono_font="DejaVu Sans Mono", linewidth_scale=1,
                         markersize_scale=1, gap=12, padding=16,
                         colormap=nothing, colors=(;))
        palette isa Symbol && palette in (:accessible, :grayscale) || throw(ArgumentError("palette must be :accessible or :grayscale."))
        for (name, value, positive) in ((:fontsize, fontsize, true),
                (:linewidth_scale, linewidth_scale, true), (:markersize_scale, markersize_scale, true),
                (:gap, gap, false), (:padding, padding, false))
            value isa Real && !(value isa Bool) && isfinite(value) &&
                (positive ? value > 0 : value >= 0) && isfinite(Float64(value)) ||
                throw(ArgumentError("$name must be a finite $(positive ? "positive" : "nonnegative") number."))
        end
        for (name, value) in ((:font, font), (:mono_font, mono_font))
            value isa AbstractString && !isempty(strip(value)) &&
                !occursin(r"[\x00-\x1f\x7f;{}<>\"'\\]", value) ||
                throw(ArgumentError("$name must be a nonempty plain font name."))
        end
        colormap === nothing || colormap isa Symbol && !isempty(String(colormap)) ||
            throw(ArgumentError("colormap must be nothing or a nonempty symbol."))
        colors isa NamedTuple || throw(ArgumentError("colors must be named role overrides."))
        for (role, color) in pairs(colors)
            role in _VISUAL_COLOR_ROLES || throw(ArgumentError("Unknown color role $role."))
            (color isa Symbol && !isempty(String(color))) ||
                (color isa AbstractString && occursin(r"^#[0-9a-fA-F]{6}([0-9a-fA-F]{2})?$", color)) ||
                throw(ArgumentError("Color $role must be a nonempty color symbol or #RRGGBB/#RRGGBBAA string."))
        end
        resolved = merge(palette === :accessible ? _ACCESSIBLE_VISUAL_COLORS : _GRAYSCALE_VISUAL_COLORS, colors)
        new{typeof(resolved)}(palette, Float64(fontsize), String(font), String(mono_font),
            Float64(linewidth_scale), Float64(markersize_scale), Float64(gap), Float64(padding), colormap, resolved)
    end
end

describe(style::VisualStyle) = (; palette=style.palette, fontsize=style.fontsize,
    font=style.font, mono_font=style.mono_font, linewidth_scale=style.linewidth_scale,
    markersize_scale=style.markersize_scale, gap=style.gap, padding=style.padding,
    colormap=style.colormap, colors=style.colors)
Base.:(==)(a::VisualStyle, b::VisualStyle) = describe(a) == describe(b)
Base.isequal(a::VisualStyle, b::VisualStyle) = isequal(describe(a), describe(b))
Base.hash(style::VisualStyle, h::UInt) = hash(describe(style), h)
Base.show(io::IO, style::VisualStyle) = print(io, "VisualStyle(palette=:", style.palette,
    ", fontsize=", style.fontsize, ", font=", repr(style.font), ")")
Base.show(io::IO, ::MIME"text/plain", style::VisualStyle) = show(io, style)

_visual_color(style::VisualStyle, color::Symbol) = color
function _visual_color(style::VisualStyle, role::_VisualRole)
    role.kind === :categorical || return getproperty(style.colors, role.kind)
    colors = style.palette === :grayscale ? _VISUAL_CATEGORY_GRAYS : _VISUAL_CATEGORY_COLORS
    return colors[mod1(role.index, length(colors))]
end
_visual_color(style::VisualStyle, color::AbstractString) = color
_visual_colormap(style::VisualStyle, original::Symbol) = something(style.colormap,
    style.palette === :grayscale ? :grays : original)
_visual_marker(::Any) = :circle
_visual_marker(role::_VisualRole) = role.kind === :source ? :circle :
    role.kind === :target ? :rect : role.kind === :both ? :diamond :
    role.kind === :selected ? :utriangle : role.kind === :positive ? :utriangle :
    role.kind === :negative ? :dtriangle : role.kind === :essential ? :diamond : :circle
_visual_role_label(role::Symbol) = role in (:source, :target, :both, :selected, :inactive) ?
    " [$role]" : ""
