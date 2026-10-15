module TamerOpMiniQhullExt

import TamerOp, MiniQhull

function _tetrahedra(points)
    # No QJ: numerical perturbations would change the input filtration.
    cells = MiniQhull.delaunay(permutedims(points), "qhull d Qt Qbb Qc Qz")
    return [sort!(Int.(cells[:, i])) for i in axes(cells, 2)]
end

function __init__()
    TamerOp.DataIngestion._SPATIAL_DELAUNAY_IMPL[] = _tetrahedra
end

end
