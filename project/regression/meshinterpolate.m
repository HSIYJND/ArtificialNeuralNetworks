function [Xmesh, Ymesh, Zmesh, interpolant] = meshinterpolate(X,Y,Z,gridn)
% MESHINTERPOLATE Interpolate the data and evaluate the interpolant on a
% mesh
    
    % we create a new grid first, one that is regular
    Xlin = linspace(min(X),max(X),gridn);
    Ylin = linspace(min(Y),max(Y),gridn);
    
    % make a meshgrid from the linearly spaces vectors
    % i.e. go from vectors to matrices
    [Xmesh,Ymesh] = meshgrid(Xlin,Ylin);
    % create a simple interpolant to guide the eye
    interpolant = TriScatteredInterp(X,Y,Z);
    % map it on the meshgrid
    Zmesh = interpolant(Xmesh,Ymesh);
end