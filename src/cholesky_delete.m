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