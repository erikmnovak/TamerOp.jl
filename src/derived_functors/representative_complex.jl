# Public access to the stored coordinate complex, shared by algebra inspection.

"""
    representative_complex(E)

Return the coordinate complex in which `representative(E, degree, coords)`
lives. Projective and injective Ext retain their Hom cochain complex; unified
Ext uses its canonical model. Tor is reindexed as `C^(-s) = C_s`, so its
boundary lowers homological degree while increasing cohomological degree.
The extra incoming-boundary term is retained; it is not another computed
Ext/Tor group. No quotient bases or multiplication tables are requested.

Treat returned complexes as read-only. Ext returns its stored complex; Tor
copies its boundary matrices into a `CochainComplex`. Algebras use their
underlying additive model. All computations concern the recorded finite
category, without an automatic identification with ambient persistence Ext.
"""
representative_complex(E::Union{ExtSpaceProjective,ExtSpaceInjective}) = E.complex
representative_complex(E::ExtSpace) = representative_complex(
    E.canon === :projective ? projective_model(E) : injective_model(E))
representative_complex(A::ExtAlgebra) = representative_complex(underlying_ext_space(A))
representative_complex(A::TorAlgebra) = representative_complex(underlying_tor_space(A))
function representative_complex(T::Union{TorSpace{K},TorSpaceSecond{K}}) where {K}
    n = length(T.dims)-1
    ChainComplexes.CochainComplex{K}(-n,0,reverse(T.dims),
        [copy(b) for b in reverse(T.bd)]; field=provenance(T).field)
end
