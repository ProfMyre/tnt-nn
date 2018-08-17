% ===============================================================
% TNT-NN: A Fast Active Set Method for Solving Large Non-Negative 
% Least Squares Problems.
% ===============================================================
% Minimize norm2(b - A * x) with constraints: x(i) >= 0
% ===============================================================
% Authors:	Erich Frahm, frahm@physics.umn.edu
%		    Joseph Myre, myre@stthomas.edu
% ===============================================================

function [ x AA status, OuterLoop, TotalInnerLoops ] = ...
    tntnn ( A, b, lambda, rel_tol, AA, use_AA, red_c, exp_c)
    
show_hist = 0;

% Check number of inputs.
if(nargin >8)
    error('TooManyInputs', ...
        'requires at most 6 optional inputs');
end

% Fill in unset optional values.
switch nargin
    case 2
        lambda = 0;
        rel_tol = 0;
        AA = 0;
        use_AA = 0;
        red_c = 0.2;
        exp_c = 1.2;
    case 3
        rel_tol = 0;
        AA = 0;
        use_AA = 0;
        red_c = 0.2;
        exp_c = 1.2;
    case 4
        AA = 0;
        use_AA = 0;
        red_c = 0.2;
        exp_c = 1.2;
    case 5
        use_AA = 0;
        red_c = 0.2;
        exp_c = 1.2;
    case 6
        red_c = 0.2;
        exp_c = 1.2;
    case 7
        exp_c = 1.2;
end

% ===============================================================
% Set verbosity to: 
% 0 for no output
% 1 for output in a hist matrix file
% 2 for output printed to the console
% ===============================================================
verbose = 0;
if (verbose > 0) 
    hist = zeros(10000,6); % preallocate some space
    hist_index = 0;     
end

x = 0;
status = 3; % unknown failure
iteration = -1;

% Get the input matrix size.
[m n] = size(A);

% Check the input vector size.
[mb nb] = size(b);
if ((mb ~= m) || (nb ~= 1))
    status = 2; % failure: vector is wrong size
    return;
end

% ===============================================================
% Compute A'A one time for use as a preconditioner with the LS 
% solver.  This is not necessary for all LS solvers.  Unless you 
% need it for preconditioning it is unlikely you actually need 
% this step.
% ===============================================================

if (use_AA == 1)
    if (size(AA) ~= [n n])
        status = 2; % failure: matrix is wrong size
        return;
    end
else
    %tic
    AA = A' * A; % one time
    %toc
end
AAsize = size(AA);

% ===============================================================
% AA is a symmetric and positive definite (probably) n x n matrix.
% If A did not have full rank, then AA is positive semi-definite.
% Also, if A is very ill-conditioned, then rounding errors can make 
% AA appear to be indefinite. Modify AA a little to make it more
% positive definite.
% ===============================================================
epsilon = 10 * eps(1) * norm(AA,1);
AA = AA + (epsilon * eye(n));

% ===============================================================
% In this routine A will never be changed, but AA might be adjusted
% with a larger "epsilon" if needed. Working copies called B and BB
% will be used to perform the computations using the "free" set 
% of variables.
% ===============================================================
% Pre-allocate some matlab space.
% ===============================================================
x = zeros(n,1);
binding_set = zeros(1,n);
free_set = zeros(1,n);
B  = zeros(m,n);
BB = zeros(n,n);
insertion_set = zeros(1,n);
residual = zeros(1,n);
gradient = zeros(1,n);

% ===============================================================
% Initialize sets.
% ===============================================================
for i=1:n
    free_set(i) = i;
end
binding_set = [];

% ===============================================================
% This sets up the unconstrained, core LS solver
% ===============================================================
[ score, x, residual, free_set, binding_set, ...
  AA, epsilon, dels, lps ] = lsq_solve(A, b, lambda, ...
  AA, epsilon, free_set, binding_set, n);
      
% ===============================================================
% Outer Loop.
% ===============================================================
OuterLoop = 0;
TotalInnerLoops = 0;
insertions = n;
while (1)
    
    OuterLoop = OuterLoop+1;
 
    % ===============================================================
    % Save this solution.
    % ===============================================================
    best_score = score;
    best_x = x;
    best_free_set = free_set;
    best_binding_set = binding_set;
    best_insertions = insertions;
    max_insertions = floor(exp_c * best_insertions);
    
    % ===============================================================
    % Compute the gradient of the "Normal Equations".
    % ===============================================================
    gradient = A' * residual;
     
    % ===============================================================
    % Check the gradient components.
    % ===============================================================
    insertions = 0;
    insertion_set = [];
    for i=1:numel(binding_set)
        if (gradient(binding_set(i)) > 0)
            insertions = insertions + 1;
            insertion_set(insertions) = i;
        end
    end
    
    % ===============================================================
    % Are we done ?
    % ===============================================================
    if (insertions == 0)
        % There were no changes that were feasible. 
        % We are done.
        status = 0; % success 
        if (verbose > 0) 
            hist_index = hist_index+1;
            hist(hist_index,:) = [0, 0, 0, 0, 0, 0];
            save('nnlsq_hist.mat', 'hist');    
            if (show_hist > 0)
                hist(1:hist_index,:)
            end
        end
        return;
    end
    
    % ===============================================================
    % Sort the possible insertions by their gradients to find the 
    % most attractive variables to insert.
    % ===============================================================
    grad_score = gradient(binding_set(insertion_set));
    [ grad_list set_index ] = sort(grad_score, 'descend');
    insertion_set = insertion_set(set_index);

    % ===============================================================
    % Inner Loop.
    % ===============================================================
    InnerLoop = 0;
    while (1)
    
        InnerLoop = InnerLoop+1;
        TotalInnerLoops = TotalInnerLoops+1;

        % ==============================================================
        % Adjust the number of insertions.
        % ==============================================================
        insertions = floor(red_c * insertions);
        if (insertions == 0)
            insertions = 1;
        end
        if (insertions > max_insertions)
            insertions = max_insertions;
        end
        insertion_set((insertions+1):end) = [];

        % ==============================================================
        % Move variables from "binding" to "free".
        % ==============================================================
        free_set = [ free_set binding_set(insertion_set) ];
        binding_set(insertion_set) = [];
        
        % ===============================================================
        % Compute a feasible solution using the unconstrained 
		% least-squares solver of your choice.
        % ===============================================================
        [ score, x, residual, free_set, binding_set, ...
          AA, epsilon, dels, lps ] = lsq_solve(A, b, lambda, ...
          AA, epsilon, free_set, binding_set, insertions);

        % ===============================================================
        % Accumulate history info for algorithm tuning.
        % ===============================================================
        % Each row has 6 values:
        % 1) Outer loop number
        % 2) Inner loop number
        % 3) Total number of inner loops
        % 4) Insertions in this inner loop
        % 5) Deletions required to make the insertions feasible
        % 6) Number of deletion loops required for these insertions
        % ===============================================================
        if (verbose > 1) 
            [OuterLoop,InnerLoop,TotalInnerLoops,insertions,dels,lps]
        end
        if (verbose > 0) 
            hist_index = hist_index+1;
            hist(hist_index,:) = ...
            [OuterLoop,InnerLoop,TotalInnerLoops,insertions,dels,lps];
        end
    
        % ===============================================================
        % Check for new best solution.
        % ===============================================================
        if (score < (best_score*(1 - rel_tol))) 
            break;
        end
        
        % ===============================================================
        % Restore the best solution.
        % ===============================================================
        score = best_score;
        x = best_x;
        free_set = best_free_set;
        binding_set = best_binding_set;
        max_insertions = floor(exp_c * best_insertions);
        
        % ===============================================================
        % Are we done ?
        % ===============================================================
        if (insertions == 1)
            % The best feasible change did not improve the score. 
            % We are done.
            status = 0; % success 
            if (verbose > 0) 
                hist_index = hist_index+1;
                hist(hist_index,:) = [1, 1, 1, 1, 1, 1];
                save('nnlsq_hist.mat', 'hist');
                if (show_hist > 0)
                    hist(1:hist_index,:)
                end
            end
            return;
        end
                
    end % Inner Loop
end % Outer Loop

return;
end

% ====================================================================
% Least squares feasible solution using a preconditioned conjugate  
% gradient least-squares solver.
% ====================================================================
% Minimize norm2(b - A * x)
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

% =============================================================
% Cholesky decomposition.
% =============================================================
[R,p] = chol(BB); % O(n^3/3)
while (p > 0)
    % It may be necessary to add to the diagonal of B'B to avoid 
    % taking the sqare root of a negative number when there are 
    % rounding errors on a nearly singular matrix. That's still OK 
    % because we just use the Cholesky factor as a preconditioner.
    epsilon = epsilon * 10;
    epsilon
    AA = AA + (epsilon * eye(n));
    BB = AA(free_set,free_set);
    if (lambda > 0)
        for i=1:numel(free_set)
            BB(i,i) = BB(i,i) + (lambda*lambda);
        end
    end
    clear R;
    [R,p] = chol(BB); % O(n^3/3)
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
    % Use PCGNR to find the unconstrained optimum in 
    % the "free" variables.
    % ------------------------------------------------------------
    [reduced_x k] = pcgnr(B,b,R);
    
    if( k > lsq_loops)
        lsq_loops = k;
    end
    
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
    
    % ------------------------------------------------------------
    % Compute R, the Cholesky factor.
    % ------------------------------------------------------------
    R = cholesky_delete(R,BB,deletion_set);
    
end

%
% Clear out the B and BB vars to save memory
%
% clear B;
% clear BB;

% ------------------------------------------------------------
% Unscramble the column indices to get the full (unreduced) x.
% ------------------------------------------------------------
[m n] = size(A);
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

% ====================================================================
% Iterative Methods for Sparse Linear Systems, Yousef Saad
% Algorithm 9.7 Left-Preconditioned CGNR
% http://www.cs.umn.edu/~saad/IterMethBook_2ndEd.pdf
% ====================================================================
% Author:   Erich Frahm, frahm@physics.umn.edu
%           Joseph Myre, myre@stthomas.edu
% ====================================================================

function [ x, k ] = pcgnr ( A, b, R )
    [ m n ] = size(A);
    x = zeros(n,1);
    r = b;
    r_hat = A' * r; % matrix_x_vector, O(mn)
    y = R' \ r_hat; % back_substitution, O(n^2)
    z = R \ y; % back_substitution, O(n^2)
    p = z;
    gamma = dot(z,r_hat);
    prev_rr = -1;
    for k = 1:n
        w = A * p; % matrix_x_vector, O(mn)
        ww = dot(w,w);
        if (ww == 0)
            return;
        end
        alpha = gamma/ww;
        x_prev = x;
        x = x + (alpha*p);
        r = b - (A * x); % matrix_x_vector, O(mn)
        r_hat = A' * r; % matrix_x_vector, O(mn)
        
        % ---------------------------------------------
        % Enforce continuous improvement in the score.
        % ---------------------------------------------
        rr = dot(r_hat,r_hat);
        if ((prev_rr >= 0) && (prev_rr <= rr))
            x = x_prev;
            return;
        end
        prev_rr = rr;
        % ---------------------------------------------
        
        y = R' \ r_hat; % back_substitution, O(n^2)
        z = R \ y; % back_substitution, O(n^2)
        gamma_new = dot(z,r_hat);
        beta = gamma_new / gamma;
        p = z + (beta * p);
        gamma = gamma_new;
        if (gamma == 0)
            return;
        end
    end
end

% ====================================================================
% Compute a new Cholesky factor after deletion of some variables.
% ====================================================================
% Author: Erich Frahm, frahm@physics.umn.edu
% ====================================================================
function R = cholesky_delete(R,BB,deletion_set)

[ m n ] = size(R);
[ r c ] = size(deletion_set);
num_deletions = max(r,c);

speed_fudge_factor = 0.001;
if (num_deletions > (speed_fudge_factor * n))

    % =============================================================
    % Full Cholesky decomposition of BB (on GPUs).
    % =============================================================
    [R,p] = chol(BB); % O(n^3/3)
    if (p > 0)
        % This should never happen because we have already added
        % a sufficiently large "epsilon" to AA to do the
        % nonnegativity tests required to create the deleted_set.
        dummy_var = fail_here1; % fail_here1 is not defined.
    end
    
else

    for i=1:num_deletions
        j = deletion_set(i);
        
        % =============================================================
        % This function is just a stripped version of Matlab's qrdelete.
        % Stolen from:
        % http://pmtksupport.googlecode.com/svn/trunk/lars/larsen.m
        % =============================================================
        R(:,j) = []; % remove column j
        n = size(R,2);
        for k = j:n
            p = k:k+1;
            [G,R(p,k)] = planerot(R(p,k)); % remove extra element in col
            if k < n
                R(p,k+1:n) = G*R(p,k+1:n); % adjust rest of row
            end
        end
        R(end,:) = []; % remove zero'ed out row
        
    end
end

return;
end

