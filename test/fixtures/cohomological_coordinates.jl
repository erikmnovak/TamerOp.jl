function coordinate_ring_fixture(;n=12)
    I=TamerOp.DataIngestion
    pts=[(cospi(2i/n),sinpi(2i/n)) for i in 0:n-1]
    edges=sort!(vcat([[i,i+1] for i in 1:n-1], [[1,n]]))
    simplices=[[[i] for i in 1:n],edges]
    G=I._materialize_simplicial_output(simplices,[(0,) for _ in 1:n+length(edges)],I.FiltrationSpec(kind=:rips))[1]
    source=I.filtered_complex(G;geometry=[[pts[v] for v in s] for ss in simplices for s in ss],
        source_ids=[Tuple(s) for ss in simplices for s in ss],construction="cycle graph")
    return source
end

function coordinate_torus_fixture(;nx=4,ny=5)
    I=TamerOp.DataIngestion
    index(i,j)=mod(i,nx)+1+nx*mod(j,ny)
    triangles=Set{NTuple{3,Int}}()
    for j in 0:ny-1,i in 0:nx-1
        a,b,c,d=index(i,j),index(i+1,j),index(i+1,j+1),index(i,j+1)
        push!(triangles,Tuple(sort([a,b,c])),Tuple(sort([a,c,d])))
    end
    edges=sort!(unique([sort([t[i],t[j]]) for t in triangles for i in 1:2 for j in i+1:3]))
    simplices=[[[i] for i in 1:nx*ny],edges,[collect(t) for t in sort!(collect(triangles))]]
    G=I._materialize_simplicial_output(simplices,[(0,) for _ in 1:sum(length,simplices)],I.FiltrationSpec(kind=:rips))[1]
    pts=[((2+cospi(2j/ny))*cospi(2i/nx),(2+cospi(2j/ny))*sinpi(2i/nx),sinpi(2j/ny)) for j in 0:ny-1 for i in 0:nx-1]
    source=I.filtered_complex(G;geometry=[[pts[v] for v in s] for ss in simplices for s in ss],
        source_ids=[Tuple(s) for ss in simplices for s in ss],construction="triangulated torus")
    alpha=zeros(Int,length(edges),2)
    for (e,(a,b)) in enumerate(edges)
        x1,y1=mod(a-1,nx),div(a-1,nx);x2,y2=mod(b-1,nx),div(b-1,nx)
        alpha[e,1]=x2-x1>nx/2 ? -1 : x1-x2>nx/2 ? 1 : 0
        alpha[e,2]=y2-y1>ny/2 ? -1 : y1-y2>ny/2 ? 1 : 0
    end
    return source,alpha
end
