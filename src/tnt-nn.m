% ===============================================================
% TNT-NN: A Fast Active Set Method for Solving Large Non-Negative 
% Least Squares Problems.
% ===============================================================
% Minimize norm2(b - A * x) with constraints: x(i) >= 0
% ===============================================================
% Authors:	Erich Frahm, frahm@physics.umn.edu
%		Joseph Myre, myre@stthomas.edu
% ===============================================================

function [ x AA status, OuterLoop, TotalInnerLoops ] = ...
    tnt-nn ( A, b, lambda, rel_tol, AA, use_AA, red_c, exp_c)
    
show_hist = 0;

% Check number of inputs.
if(nargin >8)
    error('TooManyInputs', ...
        'requires at most 3 optional inputs');
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
