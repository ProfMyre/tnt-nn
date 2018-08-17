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
    y = cs_ltsolve(R', r_hat);
    z = cs_ltsolve(R, y);
    %y = R' \ r_hat; % back_substitution, O(n^2)
    %z = R \ y; % back_substitution, O(n^2)
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
