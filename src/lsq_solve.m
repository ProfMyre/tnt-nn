% ====================================================================
% Least squares feasible solution using a preconditioned conjugate  
% gradient least-squares solver.
% ====================================================================
% Minimize norm_2(b - A * x)
% ====================================================================
% Author: Erich Frahm, frahm@physics.umn.edu
%         Joseph Myre, myre@stthomas.edu
% ====================================================================

function [ score, x, residual, free_set, binding_set, ...
           AA, epsilon, del_hist, dels, loops, lsq_loops ] = lsq_solve ( A, b, lambda, ...
           AA, epsilon, free_set, binding_set, deletions_per_loop )

% ------------------------------------------------------------
% Put the lists in order.
% ------------------------------------------------------------
free_set = sort(free_set, 'descend');
binding_set = sort(binding_set, 'descend');
    
% ------------------------------------------------------------
% Reduce A to B.
% ------------------------------------------------------------
% B is a matrix that has all of the rows of A, but its
% columns are a subset of the columns of A. The free_set
% provides a map from the columns of B to the columns of A.
B = A(:,free_set);

% ------------------------------------------------------------
% Reduce AA to BB.
% ------------------------------------------------------------
% BB is a symmetric matrix that has a subset of rows and 
% columns of AA. The free_set provides a map from the rows
% and columns of BB to rows and columns of AA.
BB = AA(free_set,free_set);

% ------------------------------------------------------------
% Adjust with Tikhonov regularization parameter lambda.
% ------------------------------------------------------------
if (lambda > 0)
    for i=1:numel(free_set)
        B(i,i) = B(i,i) + lambda;
        BB(i,i) = BB(i,i) + (lambda*lambda);
    end
end

% ------------------------------------------------------------
% Loop until the solution is feasible.
% ------------------------------------------------------------
dels = 0;
loops = 0;
lsq_loops = 0;
del_hist = [];
while (1)
    
    loops = loops + 1;
    
    % ------------------------------------------------------------
    % Compute a feasible solution using the unconstrained 
    % least-squares solver of your choice.
    % ------------------------------------------------------------
    reduced_x = your_lsq_solver(B, b);
    
    % ------------------------------------------------------------
    % Get a list of variables that must be deleted.
    % ------------------------------------------------------------
    deletion_set = [];
    for i=1:numel(free_set)
        if (reduced_x(i) <= 0) 
            deletion_set(end+1) = i;
        end
    end
    
    % ------------------------------------------------------------
    % If the current solution is feasible then quit.
    % ------------------------------------------------------------
    if (numel(deletion_set) == 0) 
        break;
    end
    
    % ------------------------------------------------------------
    % Sort the possible deletions by their reduced_x values to 
    % find the worst violators.
    % ------------------------------------------------------------
    x_score = reduced_x(deletion_set);
    [ x_list set_index ] = sort(x_score, 'ascend');
    deletion_set = deletion_set(set_index);
    
    % ------------------------------------------------------------
    % Limit the number of deletions per loop.
    % ------------------------------------------------------------
    if (numel(deletion_set) > deletions_per_loop)
        deletion_set(deletions_per_loop+1:end) = [];
    end
    deletion_set = sort(deletion_set, 'descend');
    del_hist = union(del_hist, deletion_set);
    dels = dels + numel(deletion_set);
    
    % ------------------------------------------------------------
    % Move the variables from "free" to "binding".
    % ------------------------------------------------------------
    binding_set = [ binding_set free_set(deletion_set) ];
    free_set(deletion_set) = [];
    
    % ------------------------------------------------------------
    % Reduce A to B.
    % ------------------------------------------------------------
    % B is a matrix that has all of the rows of A, but its
    % columns are a subset of the columns of A. The free_set
    % provides a map from the columns of B to the columns of A.
    clear B;
    B = A(:,free_set);
    
    % ------------------------------------------------------------
    % Reduce AA to BB.
    % ------------------------------------------------------------
    % BB is a symmetric matrix that has a subset of rows and 
    % columns of AA. The free_set provides a map from the rows
    % and columns of BB to rows and columns of AA.
    clear BB;
    BB = AA(free_set,free_set);
    
    % ------------------------------------------------------------
    % Adjust with Tikhonov regularization parameter lambda.
    % ------------------------------------------------------------
    if (lambda > 0) 
        for i=1:numel(free_set)
            B(i,i) = B(i,i) + lambda;
            BB(i,i) = BB(i,i) + (lambda*lambda);
        end
    end
    
end

%
% Clear out the B and BB vars to save memory
%
% clear B;
% clear BB;

% ------------------------------------------------------------
% Unscramble the column indices to get the full (unreduced) x.
% ------------------------------------------------------------
x = zeros(n,1);
x(free_set) = reduced_x;

% ------------------------------------------------------------
% Compute the full (unreduced) residual.
% ------------------------------------------------------------
residual = b - (A * x);

% ------------------------------------------------------------
% Compute the norm of the residual.
% ------------------------------------------------------------
score = sqrt(dot(residual,residual));

return;
end